"""
Step 1: Build FAISS Index from Source Documents
This script loads the source text files, creates embeddings, and builds a searchable FAISS index.
"""

import os
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from tqdm import tqdm

# Paths
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SOURCE_DOCS_DIR = DATA_DIR / "source_docs"
OUTPUT_DIR = DATA_DIR / "outputs"

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("STEP 1: Building FAISS Index from Source Documents")
print("=" * 60)

# 1. Load all text files
print(f"\n1. Loading documents from: {SOURCE_DOCS_DIR}")
documents = []
doc_metadata = []

# Hardcoded to preserve the original filesystem-dependent iteration order
file_order = [
    'named_entity_recognition.txt', 'attention_mechanism.txt', 'tokenization.txt',
    'chain_of_thought_reasoning.txt', 'information_retrieval.txt', 'instruction_tuning_alignment.txt',
    'few_shot_in_context_learning.txt', 'llm_as_judge_evaluation.txt', 'lora_finetuning.txt',
    'transformer_architecture.txt', 'text_summarization.txt', 'model_scaling_laws.txt',
    'prompt_vs_finetuning.txt', 'hallucination_detection_methods.txt', 'beam_search_decoding.txt',
    'llm_hallucinations.txt', 'vector_embeddings.txt', 'rag_systems.txt',
    'knowledge_distillation.txt', 'quantization.txt', 'transfer_learning_nlp.txt',
    'large_language_models.txt'
]
txt_files = [SOURCE_DOCS_DIR / fname for fname in file_order]
print(f"   Found {len(txt_files)} documents")

for txt_file in txt_files:
    with open(txt_file, 'r', encoding='utf-8') as f:
        content = f.read()
    documents.append(content)
    doc_metadata.append({"filename": txt_file.name, "size": len(content)})
    print(f"   - {txt_file.name}: {len(content)} chars")

total_chars = sum(m["size"] for m in doc_metadata)
print(f"   Total: {total_chars} characters across {len(documents)} docs")

# 2. Split documents into chunks (for better retrieval)
print(f"\n2. Splitting documents into chunks...")
chunk_size = 500  # Characters per chunk
overlap = 100      # Overlap between chunks for context

chunks = []
chunk_metadata = []

for doc_idx, doc in enumerate(documents):
    filename = doc_metadata[doc_idx]["filename"]
    # Split into chunks with overlap
    idx = 0
    while idx < len(doc):
        chunk = doc[idx:idx+chunk_size]
        chunks.append(chunk)
        chunk_metadata.append({
            "doc_index": doc_idx,
            "filename": filename,
            "chunk_index": len(chunk_metadata),
            "start_char": idx,
            "end_char": min(idx+chunk_size, len(doc)),
            "size": len(chunk)
        })
        idx += (chunk_size - overlap)

print(f"   Created {len(chunks)} chunks")
print(f"   Avg chunk size: {np.mean([m['size'] for m in chunk_metadata]):.0f} chars")

# 3. Generate embeddings using sentence-transformers
print(f"\n3. Generating embeddings using sentence-transformers...")
print(f"   (This may take 1-2 minutes for {len(chunks)} chunks)")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(chunks, show_progress_bar=True, batch_size=32)
embeddings = np.array(embeddings, dtype=np.float32)

print(f"   Embeddings shape: {embeddings.shape}")
print(f"   Embedding dimension: {embeddings.shape[1]}")

# 4. Build FAISS index
print(f"\n4. Building FAISS index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print(f"   Index built with {index.ntotal} vectors")
print(f"   Index type: L2 (Euclidean distance)")

assert index.ntotal == len(chunks), f"Mismatch before writing! Index: {index.ntotal}, Chunks: {len(chunks)}"

# 5. Save index and metadata
print(f"\n5. Saving index and metadata...")
index_path = DATA_DIR / "faiss_index.bin"
metadata_path = DATA_DIR / "faiss_metadata.json"
chunks_path = DATA_DIR / "faiss_chunks.json"

faiss.write_index(index, str(index_path))
print(f"   ✓ Index saved to: {index_path}")

with open(metadata_path, 'w') as f:
    json.dump({
        "num_chunks": len(chunks),
        "embedding_model": "all-MiniLM-L6-v2",
        "embedding_dimension": int(dimension),
        "chunk_size": chunk_size,
        "chunk_overlap": overlap,
        "documents": doc_metadata
    }, f, indent=2)
print(f"   ✓ Metadata saved to: {metadata_path}")

with open(chunks_path, 'w', encoding='utf-8') as f:
    json.dump(chunks, f, indent=2, ensure_ascii=False)
print(f"   ✓ Chunks saved to: {chunks_path}")

print("\n" + "=" * 60)
print("✅ FAISS Index built successfully!")
print("=" * 60)
print(f"   - {len(chunks)} chunks indexed")
print(f"   - {dimension}-dimensional embeddings")
print(f"   - Ready for RAG retrieval")
print("=" * 60)
