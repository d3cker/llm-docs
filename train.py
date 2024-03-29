import os
model_name = "meta-llama/Llama-2-7b-chat-hf"

# The instruction dataset to use
dataset_name = "chrishayuk/test"

# Fine-tuned model name
new_model = "./llama-2-7b-lora"
new_model_a = "./llama-2-7b-merged"

# Output directory where the model predictions and checkpoints will be stored
output_dir = "./results"

# Number of training epochs
num_train_epochs = 20

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
    logging,
)

# load the quantized settings, we're doing 4 bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    # use the gpu
    device_map={"": 0}
)

# don't use the cache
model.config.use_cache = False

print("\n#### Loading tokenizer \n")

# Load the tokenizer from the model (llama2)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("\n#### Display reference prompt and response \n")

logging.set_verbosity(logging.CRITICAL)
prompt = "Write a Hello Chris program in psion opl"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
result = pipe(f"[INST] {prompt} [/INST]\n")
print(result[0]['generated_text'])

print("\n#### Loading dataset \n")

from datasets import load_dataset
# Load the dataset
dataset = load_dataset(dataset_name, split="train")

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

# Set training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,      # uses the number of epochs earlier
    per_device_train_batch_size=4,          # 4 seems reasonable
    gradient_accumulation_steps=2,          # 2 is fine, as we're a small batch
    optim="paged_adamw_32bit",              # default optimizer
    save_steps=0,                           # we're not gonna save
    logging_steps=10,                       # same value as used by Meta
    learning_rate=2e-4,                     # standard learning rate
    weight_decay=0.001,                     # standard weight decay 0.001
    fp16=False,                             # set to true for A100
    bf16=False,                             # set to true for A100
    max_grad_norm=0.3,                      # standard setting
    max_steps=-1,                           # needs to be -1, otherwise overrides epochs
    warmup_ratio=0.03,                      # standard warmup ratio
    group_by_length=True,                   # speeds up the training
    lr_scheduler_type="cosine",           # constant seems better than cosine
    report_to="tensorboard"
)

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,                # use our lora peft config
    dataset_text_field="text",
    max_seq_length=None,                    # no max sequence length
    tokenizer=tokenizer,                    # use the llama tokenizer
    args=training_arguments,                # use the training arguments
    packing=False,                          # don't need packing
)

print("\n#### Training model \n")

# Train model
trainer.train()

# Save trained model
print("\n#### Saving pretrained model\n")

trainer.model.save_pretrained(new_model)

print("\n#### Verify tunning result\n")

logging.set_verbosity(logging.CRITICAL)
prompt = "Write a Hello Chris program in psion opl"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
#result = pipe(f"[")
result = pipe(f"[INST] {prompt} [/INST]")
print(result[0]['generated_text'])

del model
del pipe
del trainer
import gc
gc.collect()
gc.collect()

print("\n#### Load and merge\n")

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map={"": 0}
)

model = PeftModel.from_pretrained(base_model, new_model)
model = model.merge_and_unload()

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("\n#### Saving merged models.\n")

model.save_pretrained(new_model_a)
tokenizer.save_pretrained(new_model_a)
