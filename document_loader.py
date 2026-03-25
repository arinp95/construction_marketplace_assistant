"""Load assessment documents from disk: UTF-8 text and PDF."""

from pathlib import Path

from pypdf import PdfReader


def load_document(path: Path) -> str:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="replace")

    if suffix == ".pdf":
        # Assessment files may be UTF-8 text saved with a .pdf extension
        head = path.read_bytes()[:8]
        if not head.startswith(b"%PDF"):
            return path.read_text(encoding="utf-8", errors="replace")
        try:
            reader = PdfReader(str(path))
            parts: list[str] = []
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    parts.append(t)
            joined = "\n\n".join(parts).strip()
            if joined:
                return joined
        except Exception:
            pass
        # Fallback: some assessment files are plain text with a .pdf extension
        return path.read_text(encoding="utf-8", errors="replace")

    return path.read_text(encoding="utf-8", errors="replace")


def iter_document_paths(directories: list[Path]) -> list[Path]:
    paths: list[Path] = []
    exts = {".txt", ".md", ".pdf"}
    for d in directories:
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.is_file() and p.suffix.lower() in exts:
                paths.append(p)
    return paths
