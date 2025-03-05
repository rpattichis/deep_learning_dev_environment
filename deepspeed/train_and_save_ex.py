import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import os
from enum import Enum

import torch
from datasets import DatasetDict, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from transformers import HfArgumentParser, set_seed
from trl import SFTConfig, SFTTrainer
#from accelerate import Accelerate
from accelerate.utils import set_seed

# NOTE: if not using Meltemi, might need to modify this.
DEFAULT_MELTEMI_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
class MeltemiSpecialTokens(str, Enum):
    user = "<|user|>"
    assistant = "<|assistant|>"
    system = "<|system|>"
    eos_token = "</s>"
    pad_token = "<pad>"

    @classmethod
    def list(cls):
        return [c.value for c in cls]

def create_datasets(tokenizer, data_args, apply_chat_template=False, training=True):
    def preprocess(samples):
        batch = []
        for conversation in samples["text"]:
            batch.append(tokenizer.apply_chat_template(conversation, tokenize=False))
        return {"text": batch}   

    def format_chatbot_turns(sample):
        sample['text'] = [
            {"role": "user", "content": "Γράψε μιαν ιστορίαν γραμμένην στα κυπριακά."},
            {"role": "assistant", "content": sample['text']}   
        ]
        return sample

    raw_datasets = DatasetDict()
    for split in data_args.datasets.split(","):
        print(f"Current dataset path: {data_args.datapath + split}")
        dataset = None
        if 'csv' in split:
            dataset = load_dataset('csv', data_files=data_args.datapath + split)
        else:
            # Try if dataset on a Hub repo
            try: 
                dataset = load_dataset(data_args.dataset_name, split=split)
            except:
                raise Exception("Couldn't load dataset!")

        if "train" in split:
            raw_datasets["train"] = dataset['train']
        elif "val" in split:
            raw_datasets["val"] = dataset['train']
        elif "test" in split:
            raw_datasets["test"] = dataset['train']
        else:
            raise ValueError(f"Split type {split} not recognized as one of train, val, or test.")

    print("Done loading CSVs...")
    #accelerator = Accelerate()
    # truncating from the left if necessary
    if data_args.format_chatbot_turns:
        #with accelerator.main_process_first():
        raw_datasets = raw_datasets.map(
            format_chatbot_turns, 
            #batched=True,
            #num_proc=torch.cuda.device_count()
        )
        print("Done formatting chatbot turns...")
    if apply_chat_template:
        #with accelerator.main_process_first():
        raw_datasets = raw_datasets.map(
            preprocess,
            #batched=True,
            #num_proc=torch.cuda.device_count(),
            remove_columns=raw_datasets[data_args.datasets.split(",")[0].split('.')[0]].column_names, # edit this later
        ) 
        print("Done applying chat template...")

    train_data = raw_datasets["train"] if "train" in data_args.datasets else None
    valid_data = raw_datasets["val"] if "val" in data_args.datasets else None
    test_data = raw_datasets["test"] if "test" in data_args.datasets else None

    if training:
        print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")
        print(f"A sample of train dataset: {train_data[0]}")

    return train_data, valid_data, test_data

def create_and_prepare_model(args, training=True):
    bnb_config = None
    quant_storage_dtype = None

    # this is for model quantization
    if args.use_4bit_quantization:
        compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
        quant_storage_dtype = getattr(torch, args.bnb_4bit_quant_storage_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=args.use_4bit_quantization,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.use_nested_quant,
            bnb_4bit_quant_storage=quant_storage_dtype,
        )

        if compute_dtype == torch.float16 and args.use_4bit_quantization:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
                print("=" * 80)
        elif args.use_8bit_quantization:
            bnb_config = BitsAndBytesConfig(load_in_8bit=args.use_8bit_quantization)

    torch_dtype = (
        quant_storage_dtype if quant_storage_dtype and quant_storage_dtype.is_floating_point else torch.float16
    )

    print("Loading model using .from_pretrained() ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        quantization_config=bnb_config,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
        torch_dtype=torch_dtype,
    )

    print("Setting up LoRA config...")
    peft_config = None
    chat_template = None
    if training and args.use_peft_lora:
        peft_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=args.lora_target_modules.split(",")
            if args.lora_target_modules != "all-linear"
            else args.lora_target_modules,
        )

    print("Setting up Meltemi (tokenizer template)...")
    special_tokens = None
    chat_template = None
    if args.chat_template_format == "meltemi":
        special_tokens = MeltemiSpecialTokens
        chat_template = DEFAULT_MELTEMI_CHAT_TEMPLATE

    if special_tokens is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            pad_token=special_tokens.pad_token.value,
            # bos_token=special_tokens.bos_token.value, # there is no bos for meltemi
            eos_token=special_tokens.eos_token.value,
            additional_special_tokens=special_tokens.list(),
            trust_remote_code=True
        )
        tokenizer.chat_template = chat_template
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        tokenizer.pad_token = "<pad>"

    return model, peft_config, tokenizer

# Define and parse arguments.
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    cache_dir: str = field(
        metadata={"help": "Path to store model loaded from HuggingFace hub."}
    )
    chat_template_format: Optional[str] = field(
        default="none",
        metadata={
            "help": "meltemi|none. Pass `none` if the dataset is already formatted with the chat template."
        },
    )
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    lora_target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj",
        metadata={"help": "comma separated list of target modules to apply LoRA layers to"},
    )
    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="float16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_storage_dtype: Optional[str] = field(
        default="uint8",
        metadata={"help": "Quantization storage dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )
    use_flash_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables Flash attention for training."},
    )
    use_peft_lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables PEFT LoRA for training."},
    )
    use_8bit_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 8bit."},
    )
    use_4bit_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 4bit."},
    )
    use_reentrant: Optional[bool] = field(
        default=False,
        metadata={"help": "Gradient Checkpointing param. Refer to the related docs."},
    )
    save_model: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether we should save the fine-tuned model."}
    )


@dataclass
class DataTrainingArguments:
    datapath: str = field(
        metadata={"help": "The path of where the data files are stored."}
    )
    datasets: str = field(
        metadata={"help": "The datasets to use, seperated by commas."},
    )
    append_concat_token: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, appends `eos_token_id` at the end of each sample being packed."},
    )
    add_special_tokens: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, tokenizers adds special tokens to each sample being packed."},
    )
    format_chatbot_turns: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not the data needs to be chunked into chatbot turns / prompts."}
    )
    use_packing: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not multiiple examples should be packed together."}
    )
    dataset_text_column: Optional[str] = field(
        default="text",
        metadata={"help": "The dataset column to train the model on."}
    )


def main(model_args, data_args, training_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    # model
    model, peft_config, tokenizer = create_and_prepare_model(model_args, training=True)
    print("Done loading model!")

    # gradient ckpt
    model.config.use_cache = not training_args.gradient_checkpointing
    training_args.gradient_checkpointing = training_args.gradient_checkpointing
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": model_args.use_reentrant}

    training_args.dataset_kwargs = {
        "append_concat_token": data_args.append_concat_token,
        "add_special_tokens": data_args.add_special_tokens,
    }

    training_args.packing = data_args.use_packing 

    if data_args.dataset_text_column:
        training_args.dataset_text_field = data_args.dataset_text_column 
    print("Done setting training arguments!")

    # datasets
    train_dataset, eval_dataset, test_dataset = create_datasets(
        tokenizer,
        data_args,
        apply_chat_template=model_args.chat_template_format != "none",
        training=True
    )
    print(f"Done creating the datasets!")

    # trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config
    )
    trainer.accelerator.print(f"{trainer.model}")
    if hasattr(trainer.model, "print_trainable_parameters"):
        trainer.model.print_trainable_parameters()

    # train
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)

    # if the user wants to save the model
    if model_args.save_model:
        unwrapped_model = accelerator.unwrap_model(model)

        # New Code #
        # Saves the whole/unpartitioned fp16 model when in ZeRO Stage-3 to the output directory if
        # `stage3_gather_16bit_weights_on_model_save` is True in DeepSpeed Config file or
        # `zero3_save_16bit_model` is True in DeepSpeed Plugin.
        # For Zero Stages 1 and 2, models are saved as usual in the output directory.
        # The model name saved is `pytorch_model.bin`
        unwrapped_model.save_pretrained(
            args.output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model),
        )

if __name__ == "__main__":
    # verify GPUs that are available -- this should match with the num_processes used in num_processes
    print(f"Number of GPUs available: {torch.cuda.device_count()}")

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, SFTConfig))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(f"")
    main(model_args, data_args, training_args)