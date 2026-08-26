import os
import json
import torch
import gc
import time
import random
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from peft import PeftModel

print("=" * 80)
print("SCRIPT 04: GENERATE RESPONSES - Base, RAG, Fine-tuned")
print("=" * 80)

ARMS_TO_RUN = ["base", "rag", "finetuned"]
SEED = 42

# Seed everything
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
BASE_MODEL_DIR = os.path.join(MODELS_DIR, "base_model")
LORA_WEIGHTS_DIR = os.path.join(MODELS_DIR, "lora_weights")

print("\n[STEP 1/5] Loading base model on CPU...")
try:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    print("✓ Base model loaded")
except Exception as e:
    print(f"✗ ERROR loading base model: {e}")
    raise

print("\n[STEP 2/5] Loading LoRA adapter...")
try:
    lora_model = PeftModel.from_pretrained(base_model, LORA_WEIGHTS_DIR)
    print("✓ LoRA adapter loaded")
except Exception as e:
    print(f"✗ ERROR loading LoRA adapter: {e}")
    raise

print("\n[STEP 3/5] Loading FAISS index and chunks...")
try:
    index_path = os.path.join(DATA_DIR, "faiss_index.bin")
    chunks_path = os.path.join(DATA_DIR, "faiss_chunks.json")
    
    index = faiss.read_index(index_path)
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)
    chunks = chunks_data
    
    assert index.ntotal == len(chunks), f"Index mismatch: {index.ntotal} vectors vs {len(chunks)} chunks"
    
    print(f"✓ FAISS index loaded: {index.ntotal} chunks")
except Exception as e:
    print(f"✗ ERROR loading FAISS: {e}")
    raise

print("\n[STEP 4/5] Loading embedding model for RAG retrieval...")
try:
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    print("✓ Embedding model loaded")
except Exception as e:
    print(f"✗ ERROR loading embedding model: {e}")
    raise

print("\n[STEP 5/5] Loading QA dataset...")
try:
    qa_path = os.path.join(DATA_DIR, "qa_dataset.json")
    with open(qa_path, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)
    print(f"✓ Loaded {len(qa_pairs)} QA pairs")
except Exception as e:
    print(f"✗ ERROR loading QA dataset: {e}")
    raise

print("\n[STEP 6/6] Loading existing responses (used to backfill arms not in ARMS_TO_RUN)...")
output_path = os.path.join(DATA_DIR, "responses.json")
if os.path.exists(output_path):
    with open(output_path, "r", encoding="utf-8") as f:
        existing_responses_list = json.load(f)
    existing_responses = {r["qa_pair_id"]: r for r in existing_responses_list}
    print(f"✓ Loaded {len(existing_responses)} existing responses")
else:
    existing_responses = {}
    print("  No existing responses.json — generating all arms from scratch")

# Generation parameters (locked)
gen_params = {
    "max_new_tokens": 150,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id
}

print("\n" + "=" * 80)
print("STARTING RESPONSE GENERATION")
print("=" * 80)

responses_list = []
checkpoint_path = os.path.join(DATA_DIR, "responses_checkpoint.json")

for idx, qa in enumerate(qa_pairs, 1):
    question = qa["question"]
    expected_answer = qa["answer"]
    qa_pair_id = qa["qa_pair_id"]
    
    print(f"\n[{idx}/{len(qa_pairs)}] Question {qa_pair_id}:")
    print(f"Q: {question[:60]}...")
    
    existing_rec = existing_responses.get(qa_pair_id, {})
    
    # GENERATION 1: BASE (no context)
    if "base" in ARMS_TO_RUN:
        print("  Generating BASE response...", end=" ")
        prompt_base = f"Q: {question}\nA:"
        inputs_base = tokenizer(prompt_base, return_tensors="pt", truncation=True, max_length=512)
        inputs_base = {k: v.to("cpu") for k, v in inputs_base.items()}
        
        t_start = time.time()
        with torch.no_grad():
            outputs_base = base_model.generate(**inputs_base, **gen_params)
        base_latency = time.time() - t_start
        input_length = inputs_base["input_ids"].shape[-1]
        base_response = tokenizer.decode(outputs_base[0][input_length:], skip_special_tokens=True).strip()
        print(f"✓ ({base_latency:.2f}s)")
    else:
        base_response = existing_rec.get("base_response", "")
        base_latency = existing_rec.get("base_latency_seconds", 0.0)
    
    # GENERATION 2: RAG (with context)
    if "rag" in ARMS_TO_RUN:
        print("  Generating RAG response...", end=" ")
        q_embedding = embedding_model.encode(question, convert_to_numpy=True).astype(np.float32)
        distances, indices = index.search(np.array([q_embedding]), k=3)
        context_chunks = [chunks[i] for i in indices[0]]
        context = "\n".join(context_chunks)
        
        prompt_rag = f"Context:\n{context}\n\nQ: {question}\nA:"
        inputs_rag = tokenizer(prompt_rag, return_tensors="pt", truncation=True, max_length=512)
        inputs_rag = {k: v.to("cpu") for k, v in inputs_rag.items()}
        
        t_start = time.time()
        with torch.no_grad():
            outputs_rag = base_model.generate(**inputs_rag, **gen_params)
        rag_latency = time.time() - t_start
        input_length = inputs_rag["input_ids"].shape[-1]
        rag_response = tokenizer.decode(outputs_rag[0][input_length:], skip_special_tokens=True).strip()
        print(f"✓ ({rag_latency:.2f}s)")
    else:
        rag_response = existing_rec.get("rag_response", "")
        rag_latency = existing_rec.get("rag_latency_seconds", 0.0)
    
    # GENERATION 3: FINE-TUNED (no context)
    if "finetuned" in ARMS_TO_RUN:
        print("  Generating FINE-TUNED response...", end=" ")
        prompt_ft = f"Q: {question}\nA:"
        inputs_ft = tokenizer(prompt_ft, return_tensors="pt", truncation=True, max_length=512)
        inputs_ft = {k: v.to("cpu") for k, v in inputs_ft.items()}
        
        t_start = time.time()
        with torch.no_grad():
            outputs_ft = lora_model.generate(**inputs_ft, **gen_params)
        ft_latency = time.time() - t_start
        input_length = inputs_ft["input_ids"].shape[-1]
        ft_response = tokenizer.decode(outputs_ft[0][input_length:], skip_special_tokens=True).strip()
        print(f"✓ ({ft_latency:.2f}s)")
    else:
        ft_response = existing_rec.get("finetuned_response", "")
        ft_latency = existing_rec.get("finetuned_latency_seconds", 0.0)
    
    # Store responses
    response_record = {
        "qa_pair_id": qa_pair_id,
        "question": question,
        "expected_answer": expected_answer,
        "base_response": base_response,
        "rag_response": rag_response,
        "finetuned_response": ft_response,
        "base_latency_seconds": round(base_latency, 3),
        "rag_latency_seconds": round(rag_latency, 3),
        "finetuned_latency_seconds": round(ft_latency, 3)
    }
    responses_list.append(response_record)
    
    # Print sample
    print(f"    Base: {base_response[:60]}...")
    print(f"    RAG:  {rag_response[:60]}...")
    print(f"    FT:   {ft_response[:60]}...")
    
    # Save checkpoint every 10 responses
    if idx % 10 == 0:
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(responses_list, f, ensure_ascii=False, indent=2)
        print(f"  ✓ Checkpoint saved ({idx} responses)")

# Save final responses
output_path = os.path.join(DATA_DIR, "responses.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(responses_list, f, ensure_ascii=False, indent=2)
print(f"\n✓ Final responses saved to {output_path}")

# Cleanup
del base_model
del lora_model
del embedding_model
gc.collect()
print("✓ Memory cleaned up")

print("\n" + "=" * 80)
print("SCRIPT 04 COMPLETE")
print("=" * 80)
print(f"Summary:")
print(f"  - QA pairs processed: {len(responses_list)}")
print(f"  - Responses generated: {len(responses_list) * 3} (3 arms)")
print(f"  - Output file: {output_path}")
print(f"  - Checkpoint file: {checkpoint_path}")
print("=" * 80)
