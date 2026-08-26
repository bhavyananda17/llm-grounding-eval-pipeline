"""
Step 0: Download the base model.

Fetches Qwen2.5-1.5B-Instruct from the HuggingFace Hub and saves it to
models/base_model/ so that every later stage loads the exact same local
snapshot (base generation, LoRA fine-tuning, RAG generation).
"""

import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

PROJECT_DIR = Path(__file__).resolve().parent.parent
BASE_MODEL_DIR = PROJECT_DIR / "models" / "base_model"
BASE_MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print(f"STEP 0: Downloading {MODEL_NAME}")
print("=" * 80)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float32, device_map="cpu"
)

tokenizer.save_pretrained(BASE_MODEL_DIR)
model.save_pretrained(BASE_MODEL_DIR)

print(f"\n[OK] Base model saved to: {BASE_MODEL_DIR}")
