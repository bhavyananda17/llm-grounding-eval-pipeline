import os
import json
import torch
import gc
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset
from peft import LoraConfig, get_peft_model

print("=" * 80)
print("SCRIPT 03: FINETUNE WITH LORA")
print("=" * 80)

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
BASE_MODEL_DIR = os.path.join(MODELS_DIR, "base_model")
LORA_WEIGHTS_DIR = os.path.join(MODELS_DIR, "lora_weights")

os.makedirs(LORA_WEIGHTS_DIR, exist_ok=True)

print("\n[STEP 1/3] Loading base model from models/base_model/...")
try:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    print("✓ Model loaded on CPU")
except Exception as e:
    print(f"✗ ERROR loading model: {e}")
    raise

print("\n[STEP 2/3] Loading QA dataset from data/qa_dataset.json...")
try:
    with open(os.path.join(DATA_DIR, "qa_dataset.json"), "r", encoding="utf-8") as f:
        qa_data = json.load(f)
    print(f"✓ Loaded {len(qa_data)} QA pairs")
except Exception as e:
    print(f"✗ ERROR loading QA dataset: {e}")
    raise

# Prepare training data
print("\nPreparing training data...")
train_texts = []
for qa in qa_data:
    # Format: Q: {question}\nA: {answer}
    text = f"Q: {qa['question']}\nA: {qa['answer']}"
    train_texts.append({"text": text})

print(f"✓ Prepared {len(train_texts)} training examples")

# Create HuggingFace Dataset
dataset = Dataset.from_dict({"text": [item["text"] for item in train_texts]})

# Tokenize function (with padding=True for consistent batch length)
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=256,
        padding="max_length"
    )

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
print(f"✓ Tokenized dataset")

# Add labels column (required for causal language modeling)
def add_labels(examples):
    examples["labels"] = examples["input_ids"].copy()
    return examples

print("Adding labels column...")
tokenized_dataset = tokenized_dataset.map(add_labels, batched=True)
print("✓ Labels added")

print("\n[STEP 3/3] Configuring LoRA and training...")

# LoRA Config (locked)
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

print(f"LoRA Config:")
print(f"  - r: 8")
print(f"  - lora_alpha: 16")
print(f"  - target_modules: ['q_proj', 'v_proj']")
print(f"  - lora_dropout: 0.05")
print(f"  - bias: none")

# Apply LoRA to model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Data collator for dynamic padding
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Training Arguments (locked)
training_args = TrainingArguments(
    output_dir=LORA_WEIGHTS_DIR,
    learning_rate=2e-4,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    remove_unused_columns=False,
    optim="adamw_torch"
)

print(f"\nTraining Config:")
print(f"  - learning_rate: 2e-4")
print(f"  - batch_size: 4")
print(f"  - epochs: 3")
print(f"  - max_length: 256")
print(f"  - logging_steps: 10")
print(f"  - optimizer: adamw_torch (CPU-compatible)")
print(f"  - output_dir: {LORA_WEIGHTS_DIR}")

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

print("\n" + "=" * 80)
print("STARTING TRAINING")
print("=" * 80)

try:
    trainer.train()
    print("\n✓ Training completed successfully")
except Exception as e:
    print(f"✗ ERROR during training: {e}")
    raise

# Save LoRA adapter
print(f"\nSaving LoRA adapter to {LORA_WEIGHTS_DIR}...")
model.save_pretrained(LORA_WEIGHTS_DIR)
tokenizer.save_pretrained(LORA_WEIGHTS_DIR)
print(f"✓ Adapter saved")

# Cleanup
del model
del trainer
gc.collect()
print("✓ Memory cleaned up")

print("\n" + "=" * 80)
print("SCRIPT 03 COMPLETE")
print("=" * 80)
print(f"Summary:")
print(f"  - QA pairs used: {len(qa_data)}")
print(f"  - LoRA adapter saved to: {LORA_WEIGHTS_DIR}")
print(f"  - Training completed: 3 epochs, batch_size=4, max_length=256")
print("=" * 80)
