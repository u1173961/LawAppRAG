import os, json, glob
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OUT_DIR = "rag_index"

def load_json_docs(base="sources"):
    docs = []
    for path in glob.glob(os.path.join(base, "**/*.json"), recursive=True):
        if os.path.basename(path) == "manifest.json":
            continue

        print(f"load_json_docs(): {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Case 1: single document (dict)
        if isinstance(data, dict):
            text = (data.get("text") or "").strip()
            if len(text) >= 200:
                docs.append(data)

        # Case 2: list of documents (e.g. PDF sections)
        elif isinstance(data, list):
            for d in data:
                if not isinstance(d, dict):
                    continue
                text = (d.get("text") or "").strip()
                if len(text) >= 200:
                    docs.append(d)

        else:
            print(f"⚠️ Skipping unrecognised JSON structure: {path}")

    return docs


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    docs = load_json_docs()
    print(f"Loaded {len(docs)} docs")

    embedder = SentenceTransformer(EMBED_MODEL)

    texts = [d["text"] for d in docs]
    embs = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    embs = np.array(embs, dtype=np.float32)

    index = faiss.IndexFlatIP(embs.shape[1])  # cosine via normalized + inner product
    index.add(embs)

    faiss.write_index(index, os.path.join(OUT_DIR, "faiss.index"))

    with open(os.path.join(OUT_DIR, "docs.jsonl"), "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps({
                "title": d.get("title",""),
                "url": d.get("url",""),
                "jurisdiction": d.get("jurisdiction","uk_wide"),
                "source_org": d.get("source_org",""),
                "topic": d.get("topic",""),
                "chunk_index": d.get("chunk_index", None),
                "text": d.get("text",""),
            }, ensure_ascii=False) + "\n")

    print(f"Saved index to {OUT_DIR}/")

if __name__ == "__main__":
    main()
