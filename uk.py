'''
Created on 30 Jan 2026

@author: Laure
'''
import os, json, re
import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer

LMSTUDIO_BASE = "http://localhost:1234/v1"
LM_MODEL_NAME = "qwen3-14b-instruct"  # LM Studio will map this; exact name not critical in many setups

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBED_MODEL)

def clean_text(t: str) -> str:
    t = re.sub(r"\s+", " ", t).strip()
    return t

def chunk_text(text: str, chunk_chars=3200, overlap_chars=400):
    text = clean_text(text)
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == len(text):
            break
        start = max(0, end - overlap_chars)
    return chunks

def load_docs(folder="sources"):
    docs = []
    for root, _, files in os.walk(folder):
        for f in files:
            if not f.endswith(".json"):
                continue
            path = os.path.join(root, f)
            with open(path, "r", encoding="utf-8") as fp:
                doc = json.load(fp)
            docs.append(doc)
    return docs

def build_index(docs):
    metadatas = []
    all_chunks = []

    for doc in docs:
        chunks = chunk_text(doc["text"])
        for i, ch in enumerate(chunks):
            all_chunks.append(ch)
            metadatas.append({
                "title": doc.get("title", ""),
                "url": doc.get("url", ""),
                "jurisdiction": doc.get("jurisdiction", "uk_wide"),
                "chunk_id": i
            })

    embeddings = embedder.encode(all_chunks, normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    index = faiss.IndexFlatIP(embeddings.shape[1])  # cosine via normalized + inner product
    index.add(embeddings)

    return index, all_chunks, metadatas

def retrieve(query, index, chunks, metas, k=5, jurisdiction_hint=None):
    q_emb = embedder.encode([query], normalize_embeddings=True)
    q_emb = np.array(q_emb, dtype=np.float32)

    scores, ids = index.search(q_emb, k*3)  # oversample then filter by jurisdiction
    results = []
    for score, idx in zip(scores[0], ids[0]):
        m = metas[idx]
        if jurisdiction_hint and m["jurisdiction"] not in (jurisdiction_hint, "uk_wide"):
            continue
        results.append((float(score), chunks[idx], m))
        if len(results) >= k:
            break
    return results

UK_TRIAGE_SYSTEM = """You are a public-facing UK_types criminal triage and legal information assistant.
You are NOT a solicitor. You do NOT provide legal advice. You provide general information and signposting only.
UK law differs across England & Wales, Scotland, and Northern Ireland. If jurisdiction is unknown, ask.
Any statement about rights/process MUST be supported by the provided sources. If sources don’t support it, say so and signpost.
Output MUST be valid JSON matching the UKCrimeTriageResponse schema."""

UK_SCHEMA_REMINDER = """Return JSON with keys:
jurisdiction, role_boundary, immediate_safety_check{is_emergency,message}, situation_summary,
clarifying_questions[], general_information[{topic,content,citations[]}], what_to_do_next[],
when_to_get_a_solicitor_urgently[], limitations[], sources[{title,url,jurisdiction}]."""

def call_lmstudio(messages, max_tokens=600, temperature=0.2):
    payload = {
        "model": LM_MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    r = requests.post(f"{LMSTUDIO_BASE}/chat/completions", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def answer(user_question, index, chunks, metas):
    # Very lightweight jurisdiction hinting (you can improve this later)
    j_hint = None
    q_lower = user_question.lower()
    if any(x in q_lower for x in ["london","manchester","birmingham","wales","cardiff","england"]):
        j_hint = "england_wales"
    elif any(x in q_lower for x in ["glasgow","edinburgh","aberdeen","scotland","scottish"]):
        j_hint = "scotland"
    elif any(x in q_lower for x in ["belfast","derry","northern ireland","ni "]):
        j_hint = "northern_ireland"

    hits = retrieve(user_question, index, chunks, metas, k=5, jurisdiction_hint=j_hint)

    # Build a compact “source pack”
    source_blocks = []
    sources_list = []
    for score, text, m in hits:
        source_blocks.append(
            f"TITLE: {m['title']}\nJURISDICTION: {m['jurisdiction']}\nURL: {m['url']}\nEXCERPT: {text}\n"
        )
        sources_list.append(m)

    sources_text = "\n---\n".join(source_blocks) if source_blocks else "NO_RETRIEVED_SOURCES"

    messages = [
        {"role": "system", "content": UK_TRIAGE_SYSTEM},
        {"role": "system", "content": UK_SCHEMA_REMINDER},
        {"role": "user", "content": f"USER QUESTION:\n{user_question}\n\nRETRIEVED SOURCES:\n{sources_text}\n\nRules: cite URLs in citations[]. If no sources, ask clarifying questions and signpost only."}
    ]
    return call_lmstudio(messages)

if __name__ == "__main__":
    docs = load_docs("sources")
    index, chunks, metas = build_index(docs)

    while True:
        q = input("\nQuestion: ").strip()
        if not q:
            break
        print(answer(q, index, chunks, metas))
