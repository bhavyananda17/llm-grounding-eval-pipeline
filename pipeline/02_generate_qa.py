import os
import json
import torch
import gc
import random
from transformers import AutoTokenizer, AutoModelForCausalLM

print("=" * 80)
print("SCRIPT 02: GENERATE QA PAIRS (OPTIMIZED)")
print("=" * 80)

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
BASE_MODEL_DIR = os.path.join(MODELS_DIR, "base_model")

print("\n[STEP 1/3] Loading Qwen2.5-1.5B from models/base_model/...")
try:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    print("✓ Model loaded on CPU with float32")
except Exception as e:
    print(f"✗ ERROR loading model: {e}")
    raise

print("\n[STEP 2/3] Loading chunks from faiss_chunks.json...")
chunks_path = os.path.join(DATA_DIR, "faiss_chunks.json")
try:
    with open(chunks_path, "r", encoding="utf-8") as f:
        all_chunks = json.load(f)
    print(f"✓ Loaded {len(all_chunks)} chunks")
except Exception as e:
    print(f"✗ ERROR loading chunks: {e}")
    raise

print("\n[STEP 3/3] Generating QA pairs...")
print("Randomly sampling chunks (seed=42)...")

# 158 chunks available from 22 source documents
# Oversample to 175 to account for parse failures, targeting 150+ clean pairs
TARGET_PAIRS = 150
OVERSAMPLE = 175

random.seed(42)
sampled_indices = random.sample(range(len(all_chunks)), min(OVERSAMPLE, len(all_chunks)))
sampled_chunks = [all_chunks[i] for i in sampled_indices]

print(f"✓ Sampled {len(sampled_chunks)} chunks")

qa_pairs = []
failed_count = 0

for idx, chunk in enumerate(sampled_chunks, 1):
    print(f"\nProcessing chunk {idx}/{len(sampled_chunks)}...")
    print(f"Chunk preview: {chunk[:60]}...")
    
    # Structure prompt using HuggingFace chat template approach
    messages = [
        {
            "role": "system",
            "content": "You are a precise data engineering assistant. Generate one specific question and one direct answer about the text provided. Follow the format exactly:\nQUESTION: <text>\nANSWER: <text>\nDo not include any other text."
        },
        {
            "role": "user",
            "content": f"TEXT TO ANALYZE:\n{chunk}"
        }
    ]
    
    # Apply chat template for Qwen2.5
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception as e:
        # Fallback if apply_chat_template fails
        prompt = f"System: You are a precise data engineering assistant. Generate one specific question and one direct answer about the text provided. Follow the format exactly:\nQUESTION: <text>\nANSWER: <text>\nDo not include any other text.\n\nUser: TEXT TO ANALYZE:\n{chunk}\n\nAssistant:"
    
    try:
        # Tokenize with explicit CPU placement
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to("cpu") for k, v in inputs.items()}
        
        # Track input length for token slicing
        input_length = inputs["input_ids"].shape[-1]
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # CRITICAL: Extract only generated tokens (not input echo)
        generation_tokens = outputs[0][input_length:]
        response = tokenizer.decode(generation_tokens, skip_special_tokens=True).strip()
        
        # Strict parsing — handle both "QUESTION:" and "Question:" variants
        response_upper = response.upper()
        if "QUESTION:" not in response_upper or "ANSWER:" not in response_upper:
            failed_count += 1
            print(f"✗ Missing structural keys for chunk {idx}")
            print(f"   Response: {response[:80]}...")
            continue
        
        # Extract fields — find positions in uppercase version, extract from original
        q_pos = response_upper.find("QUESTION:") + len("QUESTION:")
        a_pos = response_upper.find("ANSWER:")
        
        question = response[q_pos:a_pos].strip()
        answer = response[a_pos + len("ANSWER:"):].strip()
        
        # Post-processing validation
        if len(question) < 5 or len(answer) < 5:
            failed_count += 1
            print(f"✗ Too-short Q/A for chunk {idx}")
            continue
        
        # Detect placeholder/instruction copying
        if "[" in question or "[" in answer or "your" in question.lower() or "your" in answer.lower():
            failed_count += 1
            print(f"✗ Placeholder text detected for chunk {idx}")
            print(f"   Q: {question[:50]}")
            continue
        
        # Store only valid pairs
        qa_pairs.append({
            "qa_pair_id": len(qa_pairs) + 1,
            "question": question,
            "answer": answer,
            "source_chunk_index": sampled_indices[idx - 1]
        })
        print(f"✓ QA Pair #{len(qa_pairs)}: Q={question[:40]}...")
    
    except Exception as e:
        failed_count += 1
        print(f"✗ Error generating QA for chunk {idx}: {e}")

# Unload model
del model
gc.collect()
print("\n✓ Model unloaded")

# Check if we have enough pairs
if len(qa_pairs) < 130:
    print(f"\n✗ ERROR: Only {len(qa_pairs)} pairs generated. Need at least 130.")
    raise ValueError(f"Insufficient QA pairs: {len(qa_pairs)} < 130")

print(f"\n✓ Successfully generated {len(qa_pairs)} clean QA pairs")
print(f"  Failed/skipped: {failed_count}/{len(sampled_chunks)}")

# Save to file
output_path = os.path.join(DATA_DIR, "qa_dataset.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
print(f"✓ Saved to {output_path}")

print("\n" + "=" * 80)
print("SCRIPT 02 COMPLETE")
print("=" * 80)
print(f"Summary:")
print(f"  - Clean QA pairs generated: {len(qa_pairs)}")
print(f"  - Failed/skipped: {failed_count}")
print(f"  - Output file: {output_path}")
print("=" * 80)
