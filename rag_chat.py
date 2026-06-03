from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re

import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer

LMSTUDIO_BASE = "http://localhost:1234/v1"
LM_MODEL_NAME = "qwen2.5-7b-instruct-1m"
#LM_MODEL_NAME = "qwen-3-14b-instruct"
#LM_MODEL_NAME = "labonne_qwen3-14b-abliterated"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BASE_DIR = Path(__file__).resolve().parent
INDEX_DIR = BASE_DIR / "rag_index"
EXCERPT_CHARS = 1000


@dataclass(frozen=True)
class RagResources:
    embedder: SentenceTransformer
    index: faiss.Index
    docs: list[dict]


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_docs() -> list[dict]:
    docs_path = INDEX_DIR / "docs.jsonl"
    print(f"{timestamp()} loading sources...")
    with docs_path.open("r", encoding="utf-8") as f:
        docs = [json.loads(line) for line in f]
    print(f"{timestamp()} finished loading sources")
    return docs


def load_resources() -> RagResources:
    index_path = INDEX_DIR / "faiss.index"
    docs_path = INDEX_DIR / "docs.jsonl"
    if not index_path.exists() or not docs_path.exists():
        raise SystemExit("No index found. Run `python3 build_index.py` first.")

    embedder = SentenceTransformer(EMBED_MODEL)
    index = faiss.read_index(str(index_path))
    docs = load_docs()

    n_index = index.ntotal
    n_docs = len(docs)
    if n_index != n_docs:
        raise RuntimeError(
            f"Index mismatch: faiss.index has {n_index} vectors but docs.jsonl has {n_docs} rows. "
            "Run a full rebuild (build_index.py) or re-sync docs.jsonl to match the index."
        )

    return RagResources(embedder=embedder, index=index, docs=docs)


def strip_think(text: str) -> str:
    # Remove <think>...</think> blocks if present
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def infer_jurisdiction_hint(q: str) -> str | None:
    ql = q.lower()
    if any(x in ql for x in ["scotland", "glasgow", "edinburgh", "aberdeen", "scottish"]):
        return "scotland"
    if any(x in ql for x in ["northern ireland", "belfast", "derry", "ni "]):
        return "northern_ireland"
    # Default for many policing/sentencing questions in your corpus:
    if any(x in ql for x in ["england", "wales", "london", "manchester", "cardiff"]):
        return "england_wales"
    return None

def retrieve(
    query: str,
    resources: RagResources,
    k: int = 4,
    jurisdiction_hint: str | None = None,
):
    q_emb = resources.embedder.encode([query], normalize_embeddings=True)
    q_emb = np.array(q_emb, dtype=np.float32)
    scores, ids = resources.index.search(q_emb, k * 3)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        d = resources.docs[int(idx)]
        j = d.get("jurisdiction", "uk_wide")
        if jurisdiction_hint and j not in (jurisdiction_hint, "uk_wide"):
            continue
        results.append((float(score), d))
        if len(results) >= k:
            break
    return results

SYSTEM = """You are a public-facing UK criminal triage and legal information assistant.
Not a solicitor; do not provide legal advice. Provide general information and signposting only.

You must not output <think> or internal reasoning. Output ONLY valid JSON in English.

Grounding rule: Only state claims supported by the retrieved sources. If unclear or jurisdiction-dependent, ask clarifying questions.

Return JSON only with:
jurisdiction, clarifying_questions, answer, citations, next_steps.
citations is an array of URLs from the retrieved sources.
"""


def call_lm(messages, max_tokens=1000, temperature=0.2):
    print(f"{timestamp()} calling lm")
    payload = {
        "model": LM_MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stop": ["\n;\n", ";\n;\n;\n"],
        "presence_penalty": 0.1,
        "frequency_penalty": 0.3,
    }
    r = requests.post(f"{LMSTUDIO_BASE}/chat/completions", json=payload, timeout=300)
    print(f"{timestamp()} got response from lm")
    if not r.ok:
        print("LM Studio error status:", r.status_code)
        try:
            print("LM Studio error body:", r.json())
        except Exception:
            print("LM Studio error text:", r.text[:1000])
        r.raise_for_status()
    
    content = r.json()["choices"][0]["message"]["content"]
    return strip_think(content)

def looks_non_english(s: str) -> bool:
    return any('\u4e00' <= ch <= '\u9fff' for ch in s)

def infer_topic_hint(q: str) -> str | None:
    q = q.lower()
    if any(w in q for w in ["stop and search", "stop & search", "stopped", "searched", "search me"]):
        return "pace_code_a"
    if any(w in q for w in ["custody", "interview", "detained", "arrested", "police station"]):
        return "pace_code_c"
    if any(w in q for w in ["sentence", "sentencing", "guideline", "guilty plea"]):
        return "sentencing"
    if any(w in q for w in ["disclosure", "unused material", "cpi", "cps disclosure"]):
        return "disclosure"
    return None

def answer(question: str, resources: RagResources):
    print(f"{timestamp()} inferring jurisdiction")
    j_hint = infer_jurisdiction_hint(question)
    print(f"{timestamp()} inferred jurisdiction")
    print(f"{timestamp()} searching sources")
    hits = retrieve(question, resources, k=4, jurisdiction_hint=j_hint)
    t_hint = infer_topic_hint(question)
    if t_hint:
        hits.sort(key=lambda x: (t_hint in (x[1].get("topic","")), x[0]), reverse=True)

    print("\n--- RETRIEVAL DEBUG ---")
    for s, d in hits:
        print(f"{s:.3f} | {d.get('jurisdiction')} | {d.get('topic')} | {d.get('title')} | {d.get('url')}")
        print(d.get("text","")[:250].replace("\n"," ") + "…\n")
    print("--- END DEBUG ---\n")

    source_pack = []
    
    for score, d in hits:
        excerpt = (d.get("text","") or "")
        excerpt = excerpt[:EXCERPT_CHARS]
        source_pack.append(
            f"TITLE: {d.get('title','')}\n"
            f"JURISDICTION: {d.get('jurisdiction','')}\n"
            f"URL: {d.get('url','')}\n"
            f"TOPIC: {d.get('topic','')}\n"
            f"EXCERPT:\n{excerpt}\n"
        )

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content":
            "LANGUAGE: English (UK). Output must be in English only.\n\n"
            f"USER QUESTION:\n{question}\n\n"
            f"RETRIEVED SOURCES (use these; cite URLs in citations[]):\n"
            + "\n---\n".join(source_pack)
            + "\n\nReturn JSON only."
        }
    ]
    
    print(f"{timestamp()} compiled sources")

    out = call_lm(messages)
    if looks_non_english(out):
        # ask the model to restate in English using the same sources
        out = call_lm([
            {"role": "system", "content": SYSTEM + "\nYou must respond in English (UK) only."},
            {"role": "user", "content": "Restate your previous answer in English (UK) only. Output JSON only.\n\n" + out}
        ])

    return out

def main():
    resources = load_resources()
    while True:
        q = input("\nQuestion (blank to quit): ").strip()
        if not q:
            break
        print(answer(q, resources))


if __name__ == "__main__":
    main()
