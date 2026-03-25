"""Optional: download assessment documents from Google Drive (requires network)."""

from pathlib import Path

import gdown

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "raw"
FILES = [
    ("1oWcyH0XkzpHeWozMBWJSFEUEw70Lrc2-", "doc1.pdf"),
    ("1m1SudlRSlEK7y_-jweDjhPB5VVWzmQ7-", "doc2.pdf"),
    ("1suFO8EBLxRH6hKKcJln4a9PRsOGu2oYj", "doc3.pdf"),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for fid, name in FILES:
        url = f"https://drive.google.com/uc?id={fid}"
        dest = OUT / name
        print(f"Downloading {name} …")
        gdown.download(url, str(dest), quiet=False)
    print("Done. Run: python ingest.py")


if __name__ == "__main__":
    main()
