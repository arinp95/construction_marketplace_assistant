"""Split documents into overlapping segments for embedding and retrieval."""


def chunk_text(
    text: str,
    source_name: str,
    max_chars: int = 500,
    overlap: int = 80,

) -> list[dict]:
    text = text.strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    raw_chunks: list[str] = []
    buf = ""

    def flush_buf():
        nonlocal buf
        if buf:
            raw_chunks.append(buf)
            buf = ""

    for para in paragraphs:
        if len(para) > max_chars:
            flush_buf()
            start = 0
            while start < len(para):
                end = min(start + max_chars, len(para))
                raw_chunks.append(para[start:end])
                if end == len(para):
                    break
                start = max(0, end - overlap)
            continue

        candidate = f"{buf}\n\n{para}".strip() if buf else para
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            flush_buf()
            buf = para

    flush_buf()

    out: list[dict] = []
    for i, chunk in enumerate(raw_chunks):
        out.append({"text": chunk, "source": source_name, "chunk_index": i})
    return out
