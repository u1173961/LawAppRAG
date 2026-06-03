import os, json, glob
import numpy as np
import faiss
import hashlib
from sentence_transformers import SentenceTransformer

INDEX_DIR = "rag_index"
SOURCES_DIR = "sources"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def stable_id(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]

def load_existing_docs():
    docs = []
    path = os.path.join(INDEX_DIR, "docs.jsonl")
    if not os.path.exists(path):
        return docs
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    return docs

def load_new_docs(existing_ids: set):
    new_docs = []
    for path in glob.glob(os.path.join(SOURCES_DIR, "**/*.json"), recursive=True):
        if os.path.basename(path) == "manifest.json":
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        items = data if isinstance(data, list) else [data]
        for d in items:
            if not isinstance(d, dict):
                continue
            if d.get("id") in existing_ids:
                continue
            text = (d.get("text") or "").strip()
            if len(text) < 200:
                continue
            new_docs.append(d)
    return new_docs

def main():
    os.makedirs(INDEX_DIR, exist_ok=True)

    print("Loading existing index + docs...")
    index = faiss.read_index(os.path.join(INDEX_DIR, "faiss.index"))
    existing_docs = load_existing_docs()
    existing_ids = {
        d.get("id") or stable_id(f"{d.get('url','')}#{d.get('chunk_index','')}")
        for d in existing_docs
    }


    print(f"Existing docs: {len(existing_docs)}")

    print("Scanning for new documents...")
    new_docs = load_new_docs(existing_ids)
    print(f"New docs found: {len(new_docs)}")

    if not new_docs:
        print("Nothing to update.")
        return

    embedder = SentenceTransformer(EMBED_MODEL)
    texts = [d["text"] for d in new_docs]
    embs = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    embs = np.array(embs, dtype=np.float32)

    index.add(embs)

    faiss.write_index(index, os.path.join(INDEX_DIR, "faiss.index"))

    with open(os.path.join(INDEX_DIR, "docs.jsonl"), "a", encoding="utf-8") as f:
        for d in new_docs:
            f.write(json.dumps({
                "id": d.get("id") or stable_id(f"{d.get('url','')}#{d.get('chunk_index','')}"),
                "title": d.get("title",""),
                "url": d.get("url",""),
                "jurisdiction": d.get("jurisdiction","uk_wide"),
                "source_org": d.get("source_org",""),
                "topic": d.get("topic",""),
                "chunk_index": d.get("chunk_index"),
                "text": d.get("text",""),
            }, ensure_ascii=False) + "\n")


    print("Index updated successfully.")

if __name__ == "__main__":
    main()
