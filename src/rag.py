import os
import chromadb
import ollama

CHROMA_DIR = '../chroma_db'
MODEL = 'llama3.2'
CHUNK_SIZE = 50


_client = chromadb.PersistentClient(path=CHROMA_DIR)
try:
    _client.delete_collection("data")
except Exception:
    pass
_collection = _client.get_or_create_collection("data")


def _iterator_files(root):
    root = os.path.abspath(root)
    if os.path.isfile(root):
        yield root
        return
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            yield os.path.join(dirpath, name)


def index(paths):
    count = 0
    for root in paths:
        for fp in _iterator_files(root):
            try:
                with open(fp, encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
            except OSError:
                continue
            for i in range(0, len(lines), CHUNK_SIZE):
                chunk = "".join(lines[i:i + CHUNK_SIZE]).strip()
                if not chunk:
                    continue
                start, end = i + 1, min(i + CHUNK_SIZE, len(lines))
                _collection.upsert(
                    ids=[f"{fp}:{start}"],
                    documents=[chunk],
                    metadatas=[{"source": fp, "start": start, "end": end}],
                )
                count += 1
    return count


def retrieve(prompt, n_results=5):
    if _collection.count() == 0:
        return None, None

    res = _collection.query(query_texts=[prompt], n_results=n_results)
    docs = res["documents"][0]
    metas = res["metadatas"][0]

    context = "\n\n".join(
        f"[{m['source']}:{m['start']}-{m['end']}]\n{d}"
        for d, m in zip(docs, metas)
    )
    sources = "\n".join(f"- `{m['source']}: {m['start']}-{m['end']}`" for m in metas)
    return context, sources


def answer_stream(prompt, context):
    stream = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content":
                "Answer ONLY based on the context provided. "
                "If there is no clear answer in context, ANSWER - \"Unable to answer.\".\n\n"},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {prompt}"},
        ],
        options={"temperature": 0.1},
        stream=True,
    )
    for chunk in stream:
        yield chunk["message"]["content"]