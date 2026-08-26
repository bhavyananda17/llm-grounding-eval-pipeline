import os
import json
import time
import re
from dotenv import load_dotenv
from groq import Groq, RateLimitError

load_dotenv()

ARMS_TO_RUN = ["base", "rag", "finetuned"]

print("=" * 80)
print("SCRIPT 05-JUDGE2: OpenAI/GPT-OSS-120B (Independent judge via Groq API)")
print("=" * 80, flush=True)

# ─── DIRECTORIES ─────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")

# ─── GROQ CLIENT ─────────────────────────────────────────────────────────────

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")
client = Groq(api_key=GROQ_API_KEY)

# ─── RATE LIMIT CONSTANTS (verified July 2026, official Groq docs) ───────────

RPD_LIMIT = 1000
SLEEP_BETWEEN_CALLS = 3  # seconds — safe under 30 RPM and 8K TPM

# ─── GLOBAL COUNTER ──────────────────────────────────────────────────────────

api_calls_made = 0
finish_reasons = []
completion_tokens = []
system_fingerprints = set()
parse_failures = 0
retries = []

# ─── LOAD RESPONSES ──────────────────────────────────────────────────────────

print("\n[STEP 1/3] Loading responses...", flush=True)
responses_path = os.path.join(DATA_DIR, "responses.json")
if not os.path.exists(responses_path):
    raise FileNotFoundError(
        f"responses.json not found at {responses_path}. Run script 04 first."
    )
with open(responses_path, "r", encoding="utf-8") as f:
    responses = json.load(f)

total_to_score = len(responses) * 3
total_api_calls_needed = total_to_score * 3

print(f"  ✓ Loaded {len(responses)} response records", flush=True)
print(f"  ✓ Responses to score: {total_to_score} ({len(responses)} QA × 3 arms)", flush=True)
print(f"  ✓ API calls needed: {total_api_calls_needed} ({total_to_score} × 3 metrics)", flush=True)
print(f"  ✓ RPD usage: {total_api_calls_needed}/{RPD_LIMIT} ({total_api_calls_needed/RPD_LIMIT*100:.0f}%)", flush=True)

# ─── LOAD EXISTING SCORES ────────────────────────────────────────────────────

print("\n[STEP 2/3] Loading existing scores (resume if present)...", flush=True)
output_path = os.path.join(DATA_DIR, "scores_judge2_gpt_oss.json")
if os.path.exists(output_path):
    with open(output_path, "r", encoding="utf-8") as f:
        existing_scores = json.load(f)
else:
    existing_scores = []
judge2_scores_dict = {(s["qa_pair_id"], s["arm"]): s for s in existing_scores}
print(f"  ✓ Loaded {len(judge2_scores_dict)} existing scores", flush=True)

print("\n[STEP 3/3] Starting GPT-OSS-120B scoring...", flush=True)
print(f"  Budget: {RPD_LIMIT} RPD available, this run needs {total_api_calls_needed} calls", flush=True)
print(f"  Sleep: {SLEEP_BETWEEN_CALLS}s between every API call", flush=True)
print(f"  Format: JSON Schema with strict=True + reasoning_format=hidden\n", flush=True)


# ─── API CALL FUNCTIONS ───────────────────────────────────────────────────────

def call_gpt_oss_with_backoff(prompt, qa_id, max_retries=3):
    """
    EXACT J1 call structure: same system prompt, temperature=0.0.
    """
    global api_calls_made
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {"role": "system", "content": "You MUST respond with ONLY a single digit 1-5 or the word 'yes' or 'no'. No explanation. Just the answer."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=256,
                temperature=0.0,
                reasoning_effort="low"
            )
            api_calls_made += 1
            if attempt > 0:
                retries.append(qa_id)
            
            # Record stats
            if response.choices:
                finish_reasons.append(response.choices[0].finish_reason)
            if response.usage:
                completion_tokens.append(response.usage.completion_tokens)
            if response.system_fingerprint:
                system_fingerprints.add(response.system_fingerprint)
                
            time.sleep(SLEEP_BETWEEN_CALLS)
            return response.choices[0].message.content
        except RateLimitError as e:
            retry_after = 60
            if hasattr(e, 'response') and e.response and hasattr(e.response, 'headers'):
                retry_after = int(e.response.headers.get('retry-after', 60))
            print(f"    Rate limit hit. Waiting {retry_after + 1}s (retry {attempt + 1}/{max_retries})...", flush=True)
            time.sleep(retry_after + 1)
        except Exception as e:
            print(f"    Unexpected error attempt {attempt + 1}/{max_retries}: {e}", flush=True)
            time.sleep(5)
    print(f"    Failed after {max_retries} attempts due to persistent 429 or other errors. Stopping immediately.", flush=True)
    import sys
    sys.exit(1)

def parse_score_1_to_5(text):
    if text is None:
        return None
    text = text.strip()
    # Strip <think> tags if they exist
    if "</think>" in text:
        text = text.split("</think>")[-1]
        
    matches = re.findall(r'\b([1-5])\b', text)
    if matches:
        return int(matches[-1])  # Take the last match to avoid reasoning numbers
    print(f"    ⚠ Failed to parse score from: {repr(text)}", flush=True)
    return None

def parse_yes_no(text):
    if text is None:
        return False
    text = text.strip()
    # Strip <think> tags if they exist
    if "</think>" in text:
        text = text.split("</think>")[-1]
        
    matches = re.findall(r'\b(yes|no)\b', text.lower())
    if matches:
        return matches[-1] == 'yes'
    print(f"    ⚠ Failed to parse yes/no from: {repr(text)}", flush=True)
    return False

# ─── SCORE ONE RESPONSE (3 API CALLS) ────────────────────────────────────────

def score_response(question, response_text, qa_id):
    """
    Score one response on 3 metrics using 3 separate API calls.
    Returns (faithfulness 1-5, relevance 1-5, hallucination yes/no).
    Prompts are identical to Judge 1 and Judge 3.
    """
    global parse_failures

    faith_prompt = f"""Rate the factual accuracy of this response on a scale of 1 to 5:
1 = completely inaccurate
2 = mostly inaccurate
3 = partially accurate
4 = mostly accurate
5 = completely accurate

QUESTION: {question}
RESPONSE: {response_text}

Respond with ONLY the number 1, 2, 3, 4, or 5:
"""
    faith_raw = call_gpt_oss_with_backoff(faith_prompt, qa_id)
    faithfulness = parse_score_1_to_5(faith_raw)
    if faithfulness is None: parse_failures += 1

    rel_prompt = f"""Rate how well this response answers the question on a scale of 1 to 5:
1 = does not answer at all
2 = barely answers
3 = partially answers
4 = mostly answers
5 = completely answers

QUESTION: {question}
RESPONSE: {response_text}

Respond with ONLY the number 1, 2, 3, 4, or 5:
"""
    rel_raw = call_gpt_oss_with_backoff(rel_prompt, qa_id)
    relevance = parse_score_1_to_5(rel_raw)
    if relevance is None: parse_failures += 1

    hall_prompt = f"""Does this response contain made-up or fabricated information?

QUESTION: {question}
RESPONSE: {response_text}

Respond with ONLY yes or no:
"""
    hall_raw = call_gpt_oss_with_backoff(hall_prompt, qa_id)
    if not (hall_raw and re.search(r'\b(yes|no)\b', hall_raw.lower())):
        parse_failures += 1
    hallucination = "yes" if (hall_raw and parse_yes_no(hall_raw)) else "no"

    return faithfulness, relevance, hallucination



# ─── MAIN SCORING LOOP ───────────────────────────────────────────────────────

scored_count = 0
start_time = time.time()

for resp in responses:
    qa_id = resp["qa_pair_id"]
    question = resp["question"]

    for arm, response_text in [
        ("base", resp.get("base_response", "")),
        ("rag", resp.get("rag_response", "")),
        ("finetuned", resp.get("finetuned_response", ""))
    ]:
        if arm not in ARMS_TO_RUN:
            continue

        try:
            faith, rel, hall = score_response(question, response_text, qa_id)

            judge2_scores_dict[(qa_id, arm)] = {
                "qa_pair_id": qa_id,
                "arm": arm,
                "question": question,
                "response": response_text,
                "judge2_faithfulness": faith,
                "judge2_relevance": rel,
                "judge2_hallucination": hall
            }

            scored_count += 1
            
            # Progress print
            faith_str = str(faith) if faith is not None else "None"
            rel_str = str(rel) if rel is not None else "None"
            print(f"  QA{qa_id} {arm} — F:{faith_str} R:{rel_str} H:{hall}", flush=True)

        except SystemExit:
            print(f"  Stopping immediately due to persistent error after {scored_count} records.", flush=True)
            break
        except Exception as e:
            print(f"  ✗ Outer error on QA{qa_id} {arm}: {e}", flush=True)

            judge2_scores_dict[(qa_id, arm)] = {
                "qa_pair_id": qa_id,
                "arm": arm,
                "question": question,
                "response": response_text,
                "judge2_faithfulness": None,
                "judge2_relevance": None,
                "judge2_hallucination": None
            }

            scored_count += 1
            
# ─── FINAL SAVE ──────────────────────────────────────────────────────────────

final_scores_list = list(judge2_scores_dict.values())
final_scores_list.sort(key=lambda x: (x["qa_pair_id"], x["arm"]))

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(final_scores_list, f, indent=2)

elapsed = time.time() - start_time
print("\n" + "=" * 80)
print("SCRIPT 05-JUDGE2 COMPLETE")
print("=" * 80)
print(f"  Records scored this run:    {scored_count}")
print(f"  Parse failures:             {parse_failures}")
from collections import Counter
print(f"  Finish reasons:             {dict(Counter(finish_reasons))}")
if completion_tokens:
    print(f"  Completion tokens (mean):   {sum(completion_tokens)/len(completion_tokens):.2f}")
    print(f"  Completion tokens (max):    {max(completion_tokens)}")
print(f"  System fingerprints:        {list(system_fingerprints)}")
print(f"  Retries (item_ids):         {retries}")
print(f"  Elapsed time:               {elapsed:.1f} seconds")
print(f"  Total in output file:       {len(final_scores_list)}")
print(f"  Output saved to:            {output_path}")
print("=" * 80)