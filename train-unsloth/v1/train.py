import dataclasses
import sys
from datetime import datetime
import os
import json
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from transformers.utils import logging
import torch

logging.set_verbosity_info()

sys.path.insert(1, '../../dataset/')
sys.path.insert(1, '../')

import loader_llm_21
import custom_sfttrainer

################################################################################################
#                                               CONFIGURATION
################################################################################################

# BASE_MODEL = "unsloth/tinyllama"
# BASE_MODEL = "/home/tiny_epoch_base/"
# BASE_MODEL = "unsloth/mistral-7b-v0.2-bnb-4bit"
BASE_MODEL = "/home/mistral-test-1/"

max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.
LORA_R = 32  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
CPU_COUNT = 8

RUN_ID = datetime.now().strftime("mistral-%y%m%d_%H%M")
MODEL_OUTPUT_DIR = os.path.abspath("./models/" + RUN_ID + "/")

TRAINING_ARGUMENTS = TrainingArguments(
    per_device_train_batch_size=64,
    gradient_accumulation_steps=1,
    warmup_steps=100,
    # max_steps = 60,
    num_train_epochs=2,
    learning_rate=5e-6,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=1,
    evaluation_strategy="steps",
    eval_steps=100,
    save_steps=100,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=0xAFFE,
    save_total_limit=15,
    output_dir=MODEL_OUTPUT_DIR + "/trainer_out/",
)

print(json.dumps({
    "BASE_MODEL": BASE_MODEL,
    "RUN_ID": RUN_ID,
    "MODEL_OUTPUT_DIR": MODEL_OUTPUT_DIR
}, indent=4))

print(json.dumps(dataclasses.asdict(TRAINING_ARGUMENTS), indent=4))

os.environ['WANDB_NAME'] = RUN_ID
os.environ['WANDB_PROJECT'] = "novel-ai-01"

################################################################################################

from unsloth import FastLanguageModel
import torch
from datetime import datetime

print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", ],
    lora_alpha=16,
    lora_dropout=0,  # Currently only supports dropout = 0
    bias="none",  # Currently only supports bias = "none"
    use_gradient_checkpointing=True,
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)

if not os.path.exists("./tokenized_train"):
    import prepare_dataset

    prepare_dataset.generate_tokenized_dataset(
        BASE_MODEL,
        "./tokenized_train",
        "./tokenized_test",
        test_size=0.02,
        seed=None,
    )

dataset_train = Dataset.load_from_disk("./tokenized_train")
dataset_test = Dataset.load_from_disk("./tokenized_test")

trainer = custom_sfttrainer.CustomSFTTrainer(
    # trainer = SFTTrainer(
    # train_dataset = dataset['train'],
    # eval_dataset = dataset['test'],
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset_train,
    eval_dataset=dataset_test,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=CPU_COUNT,
    packing=False,  # Packs short sequences together to save time!
    args=TRAINING_ARGUMENTS,
    report_to="wandb"
)

# GPU STATS
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

try:
    trainer_stats = trainer.train()

    # AFTER TRAINING GPU STATS

    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
except KeyboardInterrupt:
    print("Interrupted by user. Saving model and uploading it...")
finally:
    model.save_pretrained(MODEL_OUTPUT_DIR + "/model-save-pretrained")
    model.save_pretrained_merged(MODEL_OUTPUT_DIR + "/model-pretrained-merged")
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR + "/model-pretrained-merged")

    #### Upload result

    sys.path.insert(1, '../../util')

    import upload
    upload.upload_files(MODEL_OUTPUT_DIR, "/models/auto-upload-" + RUN_ID + "/")

