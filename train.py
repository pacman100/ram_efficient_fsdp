# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import gc
import os
import threading

import psutil
import evaluate
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers.modeling_utils import no_init_weights
from transformers.utils import ContextManagers
from transformers import (
    AutoModelForSequenceClassification,
    AutoConfig,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)
from typing import Set
import collections

import torch
from torch.nn.parameter import Parameter

from accelerate import init_empty_weights, Accelerator, DistributedType, FullyShardedDataParallelPlugin
from accelerate.utils import find_tied_parameters, retie_parameters
from accelerate.hooks import set_module_tensor_to_device

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

########################################################################
# This is a fully working simple example to use Accelerate
#
# This example trains a Bert base model on GLUE MRPC
# in any of the following settings (with the same script):
#   - single CPU or single GPU
#   - multi GPUS (using PyTorch distributed mode)
#   - (multi) TPUs
#   - fp16 (mixed-precision) or fp32 (normal precision)
#   - FSDP
#
# This example also demonstrates the checkpointing and sharding capabilities
#
# To run it in each of these various modes, follow the instructions
# in the readme for examples:
# https://github.com/huggingface/accelerate/tree/main/examples
#
########################################################################


MAX_GPU_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32


# New Code #
# Converting Bytes to Megabytes
def b2mb(x):
    return int(x / 2**20)


# New Code #
# This context manager is used to track the peak memory usage of the process
class TorchTracemalloc:
    def __enter__(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()  # reset the peak gauge to zero
        self.begin = torch.cuda.memory_allocated()
        self.process = psutil.Process()

        self.cpu_begin = self.cpu_mem_used()
        self.peak_monitoring = True
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()
        return self

    def cpu_mem_used(self):
        """get resident set size memory for the current process"""
        return self.process.memory_info().rss

    def peak_monitor_func(self):
        self.cpu_peak = -1

        while True:
            self.cpu_peak = max(self.cpu_mem_used(), self.cpu_peak)

            # can't sleep or will not catch the peak right (this comment is here on purpose)
            # time.sleep(0.001) # 1msec

            if not self.peak_monitoring:
                break

    def __exit__(self, *exc):
        self.peak_monitoring = False

        gc.collect()
        torch.cuda.empty_cache()
        self.end = torch.cuda.memory_allocated()
        self.peak = torch.cuda.max_memory_allocated()
        self.used = b2mb(self.end - self.begin)
        self.peaked = b2mb(self.peak - self.begin)

        self.cpu_end = self.cpu_mem_used()
        self.cpu_used = b2mb(self.cpu_end - self.cpu_begin)
        self.cpu_peaked = b2mb(self.cpu_peak - self.cpu_begin)
        # print(f"delta used/peak {self.used:4d}/{self.peaked:4d}")


# For testing only
if os.environ.get("TESTING_MOCKED_DATALOADERS", None) == "1":
    from accelerate.test_utils.training import mocked_dataloaders

    get_dataloaders = mocked_dataloaders  # noqa: F811


def meta_safe_apply(self, fn, ignored_modules: Set, module_name: str):
    """Applies the function recursively to a module's children and the module itself.

    This variant allows us to ignore modules to apply the function.
    The function is a slightly modified version of the one from PyTorch:
    https://github.com/pytorch/pytorch/blob/v1.13.0/torch/nn/modules/module.py#L637

    Args:
        self: the module to apply fn to.
        fn: the function called to each submodule
        ignored_modules: a set of names of modules to not apply fn.
        module_name: the current module's name.
    """
    for name, module in self.named_children():
        module_name_list = [module_name, name]
        if module_name == "":
            module_name_list = [name]
        curr_module_name = concatenate_strings(module_name_list)
        meta_safe_apply(module, fn, ignored_modules, curr_module_name)

    def compute_should_use_set_data(tensor, tensor_applied):
        if torch._has_compatible_shallow_copy_type(tensor, tensor_applied):
            # If the new tensor has compatible tensor type as the existing tensor,
            # the current behavior is to change the tensor in-place using `.data =`,
            # and the future behavior is to overwrite the existing tensor. However,
            # changing the current behavior is a BC-breaking change, and we want it
            # to happen in future releases. So for now we introduce the
            # `torch.__future__.get_overwrite_module_params_on_conversion()`
            # global flag to let the user control whether they want the future
            # behavior of overwriting the existing tensor or not.
            return not torch.__future__.get_overwrite_module_params_on_conversion()
        else:
            return False

    for key, param in self._parameters.items():
        curr_name = concatenate_strings([module_name, key])
        if param is None or curr_name in ignored_modules:
            continue
        # Tensors stored in modules are graph leaves, and we don't want to
        # track autograd history of `param_applied`, so we have to use
        # `with torch.no_grad():`
        with torch.no_grad():
            param_applied = fn(param)
        should_use_set_data = compute_should_use_set_data(param, param_applied)
        if should_use_set_data:
            param.data = param_applied
            out_param = param
        else:
            assert isinstance(param, Parameter)
            assert param.is_leaf
            out_param = Parameter(param_applied, param.requires_grad)
            self._parameters[key] = out_param

        if param.grad is not None:
            with torch.no_grad():
                grad_applied = fn(param.grad)
            should_use_set_data = compute_should_use_set_data(param.grad, grad_applied)
            if should_use_set_data:
                assert out_param.grad is not None
                out_param.grad.data = grad_applied
            else:
                assert param.grad.is_leaf
                out_param.grad = grad_applied.requires_grad_(param.grad.requires_grad)

    for key, buf in self._buffers.items():
        if buf is not None:
            self._buffers[key] = fn(buf)
    return self


def concatenate_strings(str_list, delim="."):
    """Concatenates a list of strings together with a delimiter in between the strings in the list.

    Args:
        str_list: a list of string to join.
        delim: the delimiter to separate all strings
    """
    return delim.join(str_list)


def get_fsdp_param_init_fn(accelerator):
    def fsdp_init_fn(module):
        # A dictionary of all tied parameter pointers to module names
        tied_pointers = {}

        # Goes through all modules finding which weights have the same pointers
        for name, mod in module.named_modules():
            for attr in ["weight", "bias"]:
                if hasattr(mod, attr):
                    mod_attr = getattr(mod, attr)
                    if mod_attr is None:
                        continue
                    ptr = id(mod_attr)
                    ptr_attr = (ptr, attr)
                    name_list = tied_pointers.get(ptr_attr, [])
                    name_list.append(name)
                    tied_pointers[ptr_attr] = name_list

        # Creates a dictionary of module names that should be tied together
        tied_mod_names = collections.defaultdict(list)
        # Creates a set of modules we should not initialize
        should_not_init_params = set()
        for ptr_attr_type, mod_names in tied_pointers.items():
            # No modules for this pointer are tied
            if len(mod_names) == 1:
                continue
            _, attr_type = ptr_attr_type
            first = next(mod_names.__iter__())
            for elem in mod_names:
                should_not_init_params.add(".".join([elem, attr_type]))
                tied_mod_names[(first, attr_type)].append(elem)
            # Make sure at least one of the tied parameters is initialized
            should_not_init_params.remove(".".join([first, attr_type]))

        meta_safe_apply(
            module,
            lambda t: torch.empty_like(t, device=f"cuda:{torch.cuda.current_device()}"),
            should_not_init_params,
            module_name="",
        )

        if len(tied_mod_names) > 0:
            warnings.warn(
                (
                    "The passed in model appears to have tied weights. In order to "
                    "support effective weight tying, the tied modules need to be "
                    "in the same FSDP module. If the weights are not properly tied "
                    "it can lead to loss spikes. We have tried our best to ensure "
                    "the tied weights are in the same FSDP module."
                )
            )

        # Redoes weight tying
        for name_attr, tied_names in tied_mod_names.items():
            name, attr = name_attr
            src_mod = module.get_submodule(name)
            # We need to make sure the source and destination
            # modules end up in the same FSDP module otherwise
            # with sharding weight tying gets violated
            src_mod._fsdp_wrap = False  # type: ignore
            src_params = getattr(src_mod, attr)
            for tied_name in tied_names:
                dest_mod = module.get_submodule(tied_name)
                dest_mod._fsdp_wrap = False  # type: ignore
                setattr(dest_mod, attr, src_params)

        #         if hasattr(obj, 'param_init_fn') and isinstance(obj.param_init_fn, Callable):
        #             module.apply(obj.param_init_fn)
        #         elif hasattr(module, 'reset_parameters') and isinstance(module.reset_parameters, Callable):
        #             module.reset_parameters()
        #         else:
        #             raise ValueError(
        #                 f'Object `{obj_name}` does not have a ``param_init_fn`` or a ``reset_parameters`` function. '
        #                 'This leaves parameters without initialization. Please add a ``param_init_fn`` or ``reset_parameters`` '
        #                 f'to module `{obj_name}`.')
        #         for key, value in module._parameters.items():
        #             set_module_tensor_to_device(module, key, accelerator.device, torch.empty(*value.size(), dtype=value.dtype))
        # print(f"{accelerator.process_index=} {key=} {value}")
        # module.to_empty(device=f"cuda:{accelerator.process_index}")
        #         meta_safe_apply(module,
        #                         lambda t: torch.empty_like(t, device=f'cuda:{torch.cuda.current_device()}'),
        #                         should_not_init_params,
        #                         module_name='')
        print(torch.cuda.current_device())

    return fsdp_init_fn


def load_model_from_pretrained_only_on_rank0(accelerator, cls, model_name_or_path):

    if accelerator.is_main_process:
        model = cls.from_pretrained(model_name_or_path, return_dict=True)
        param_init_fn = None
    else:
        with torch.device("meta"):
            config = AutoConfig.from_pretrained(model_name_or_path)
            model = cls.from_config(config)
        param_init_fn = lambda x: x.to_empty(device=torch.cuda.current_device(), recurse=False)
    model.train()
    return model, param_init_fn


def training_function(config, args):
    # For testing only
    if os.environ.get("TESTING_MOCKED_DATALOADERS", None) == "1":
        config["num_epochs"] = 2

    # New Code #
    # Pass the advanced FSDP settings not part of the accelerate config by creating fsdp_plugin
    fsdp_plugin = FullyShardedDataParallelPlugin(
        use_orig_params=True,
        forward_prefetch=False,
        sync_module_states=True,
    )

    # Initialize accelerator
    if args.with_tracking:
        accelerator = Accelerator(
            cpu=args.cpu,
            mixed_precision=args.mixed_precision,
            log_with="wandb",
            project_dir=args.logging_dir,
            fsdp_plugin=fsdp_plugin,
        )
    else:
        accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
    accelerator.print(accelerator.distributed_type)

    if hasattr(args.checkpointing_steps, "isdigit"):
        if args.checkpointing_steps == "epoch":
            checkpointing_steps = args.checkpointing_steps
        elif args.checkpointing_steps.isdigit():
            checkpointing_steps = int(args.checkpointing_steps)
        else:
            raise ValueError(
                f"Argument `checkpointing_steps` must be either a number or `epoch`. `{args.checkpointing_steps}` passed."
            )
    else:
        checkpointing_steps = None
    # Sample hyper-parameters for learning rate, batch size, seed and a few other HPs
    lr = config["lr"]
    num_epochs = int(config["num_epochs"])
    seed = int(config["seed"])
    batch_size = int(config["batch_size"])

    # We need to initialize the trackers we use, and also store our configuration
    if args.with_tracking:
        experiment_config = vars(args)
        accelerator.init_trackers("fsdp_glue_no_trainer", experiment_config)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    datasets = load_dataset("glue", "mrpc")
    metric = evaluate.load("glue", "mrpc")

    def tokenize_function(examples):
        # max_length=None => use the model max length (it's actually the default)
        outputs = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=None)
        return outputs

    # Apply the method we just defined to all the examples in all the splits of the dataset
    # starting with the main process first:
    with accelerator.main_process_first():
        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=["idx", "sentence1", "sentence2"],
        )

    # We also rename the 'label' column to 'labels' which is the expected name for labels by the models of the
    # transformers library
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    # If the batch size is too big we use gradient accumulation
    gradient_accumulation_steps = 1
    if batch_size > MAX_GPU_BATCH_SIZE and accelerator.distributed_type != DistributedType.TPU:
        gradient_accumulation_steps = batch_size // MAX_GPU_BATCH_SIZE
        batch_size = MAX_GPU_BATCH_SIZE

    def collate_fn(examples):
        # On TPU it's best to pad everything to the same length or training will be very slow.
        max_length = 128 if accelerator.distributed_type == DistributedType.TPU else None
        # When using mixed precision we want round multiples of 8/16
        if accelerator.mixed_precision == "fp8":
            pad_to_multiple_of = 16
        elif accelerator.mixed_precision != "no":
            pad_to_multiple_of = 8
        else:
            pad_to_multiple_of = None

        return tokenizer.pad(
            examples,
            padding="longest",
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        )

    # Instantiate dataloaders.
    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn, batch_size=batch_size
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], shuffle=False, collate_fn=collate_fn, batch_size=EVAL_BATCH_SIZE
    )

    set_seed(seed)

    # Instantiate the model (we build the model here so that the seed also control new weights initialization)
    # loading the model only on rank 0
    param_init_fn = None
    if args.ram_efficient:
        model, param_init_fn = load_model_from_pretrained_only_on_rank0(
            accelerator, AutoModelForSequenceClassification, args.model_name_or_path
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, return_dict=True)

    # New Code #
    # For FSDP feature, it is highly recommended and efficient to prepare the model before creating optimizer

    # first, provide the `param_init_fn` for fsdp_config
    accelerator.state.fsdp_plugin.param_init_fn = param_init_fn  # get_fsdp_param_init_fn(accelerator)
    print(f"{accelerator.process_index=} {model.bert.pooler.dense.weight=}")
    print(f"{accelerator.process_index=} {model.classifier.weight=}")
    model = accelerator.prepare(model)
    accelerator.print(model)
    with FSDP.summon_full_params(model):
        print(f"{accelerator.process_index=} {model.bert.pooler.dense.weight=}")
        print(f"{accelerator.process_index=} {model.classifier.weight=}")

    # Instantiate optimizer
    # New Code #
    # For FSDP feature, at present it doesn't support multiple parameter groups,
    # so we need to create a single parameter group for the whole model
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr, weight_decay=2e-4)

    # Instantiate scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=10,
        num_training_steps=(len(train_dataloader) * num_epochs) // gradient_accumulation_steps,
    )

    # New Code #
    # For FSDP feature, prepare everything except the model as we have already prepared the model
    # before creating the optimizer
    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
    # prepare method.
    optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    overall_step = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            num_epochs -= int(training_difference.replace("epoch_", ""))
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            num_epochs -= resume_step // len(train_dataloader)
            # If resuming by step, we also need to know exactly how far into the DataLoader we went
            resume_step = (num_epochs * len(train_dataloader)) - resume_step

    # Now we train the model
    for epoch in range(num_epochs):
        # New Code #
        # context manager to track the peak memory usage during the training epoch
        with TorchTracemalloc() as tracemalloc:
            model.train()
            if args.with_tracking:
                total_loss = 0
            #             with FSDP.summon_full_params(model):
            #                 print(f"{accelerator.process_index=} {model.classifier.weight=}")
            for step, batch in enumerate(train_dataloader):
                # We need to skip steps until we reach the resumed step
                if args.resume_from_checkpoint and epoch == 0:
                    if resume_step is not None and step < resume_step:
                        pass
                # We could avoid this line since we set the accelerator with `device_placement=True`.
                batch.to(accelerator.device)
                outputs = model(**batch)
                loss = outputs.loss
                loss = loss / gradient_accumulation_steps
                # print(loss)
                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float()
                accelerator.backward(loss)
                if step % gradient_accumulation_steps == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    # accelerator.print(lr_scheduler.get_lr())
                #                     with FSDP.summon_full_params(model):
                #                         print(f"{accelerator.process_index=} {model.classifier.weight=}")

                overall_step += 1

                if isinstance(checkpointing_steps, int):
                    output_dir = f"step_{overall_step}"
                    if overall_step % checkpointing_steps == 0:
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        accelerator.save_state(output_dir)
        # New Code #
        # Printing the GPU memory usage details such as allocated memory, peak memory, and total memory usage
        accelerator.print("Memory before entering the train : {}".format(b2mb(tracemalloc.begin)))
        accelerator.print("Memory consumed at the end of the train (end-begin): {}".format(tracemalloc.used))
        accelerator.print("Peak Memory consumed during the train (max-begin): {}".format(tracemalloc.peaked))
        accelerator.print(
            "Total Peak Memory consumed during the train (max): {}".format(
                tracemalloc.peaked + b2mb(tracemalloc.begin)
            )
        )
        # Logging the peak memory usage of the GPU to the tracker
        if args.with_tracking:
            accelerator.log(
                {
                    "train_total_peak_memory": tracemalloc.peaked + b2mb(tracemalloc.begin),
                },
                step=epoch,
            )

        # New Code #
        # context manager to track the peak memory usage during the evaluation
        with TorchTracemalloc() as tracemalloc:
            # model.eval()
            for step, batch in enumerate(eval_dataloader):
                # We could avoid this line since we set the accelerator with `device_placement=True`.
                batch.to(accelerator.device)
                with torch.no_grad():
                    outputs = model(**batch)
                predictions = outputs.logits.argmax(dim=-1)
                predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))
                metric.add_batch(
                    predictions=predictions,
                    references=references,
                )

            eval_metric = metric.compute()
            # Use accelerator.print to print only on the main process.
            accelerator.print(f"epoch {epoch}:", eval_metric)
            if args.with_tracking:
                accelerator.log(
                    {
                        "accuracy": eval_metric["accuracy"],
                        "f1": eval_metric["f1"],
                        "train_loss": total_loss.item() / len(train_dataloader),
                    },
                    step=epoch,
                )

            if checkpointing_steps == "epoch":
                output_dir = f"epoch_{epoch}"
                if args.output_dir is not None:
                    output_dir = os.path.join(args.output_dir, output_dir)
                accelerator.save_state(output_dir)
        # New Code #
        # Printing the GPU memory usage details such as allocated memory, peak memory, and total memory usage
        accelerator.print("Memory before entering the eval : {}".format(b2mb(tracemalloc.begin)))
        accelerator.print("Memory consumed at the end of the eval (end-begin): {}".format(tracemalloc.used))
        accelerator.print("Peak Memory consumed during the eval (max-begin): {}".format(tracemalloc.peaked))
        accelerator.print(
            "Total Peak Memory consumed during the eval (max): {}".format(tracemalloc.peaked + b2mb(tracemalloc.begin))
        )
        # Logging the peak memory usage of the GPU to the tracker
        if args.with_tracking:
            accelerator.log(
                {
                    "eval_total_peak_memory": tracemalloc.peaked + b2mb(tracemalloc.begin),
                },
                step=epoch,
            )

    if args.with_tracking:
        accelerator.end_training()


def main():
    parser = argparse.ArgumentParser(description="Simple example of training script.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16", "fp8"],
        help="Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU.",
    )
    parser.add_argument("--cpu", action="store_true", help="If passed, will train on the CPU.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to load in all available experiment trackers from the environment and use them for logging.",
    )
    parser.add_argument(
        "--ram_efficient",
        action="store_true",
        help="Whether to load in all available experiment trackers from the environment and use them for logging.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Optional save directory where all checkpoint folders will be stored. Default is the current working directory.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="Location on where to store experiment tracking logs`",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    args = parser.parse_args()
    config = {"lr": 2e-5, "num_epochs": 3, "seed": 1, "batch_size": 16}
    training_function(config, args)


if __name__ == "__main__":
    main()
