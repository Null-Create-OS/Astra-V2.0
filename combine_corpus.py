from pathlib import Path


DATA_DIR = Path("data")

OUTPUT = DATA_DIR / "training.txt"

FILES = [
    DATA_DIR / "conversations.txt",
    DATA_DIR / "archive_corpus_clean.txt",
    DATA_DIR / "knowledge.txt",
]


def read_file(path):
    if not path.exists():
        print(
            f"Skipping missing file: {path}"
        )
        return ""

    print(
        f"Reading: {path}"
    )

    return path.read_text(
        encoding="utf-8",
        errors="ignore",
    )


def main():

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 70)
    print("       ASTRA V5 CORPUS COMBINER")
    print("=" * 70)
    print()

    parts = []

    for path in FILES:

        text = read_file(path)

        if not text.strip():
            continue

        parts.append(
            text.strip()
        )

        print(
            f"  {len(text):,} characters"
        )

    if not parts:
        raise RuntimeError(
            "No training files were found."
        )

    combined = "\n\n".join(parts)

    OUTPUT.write_text(
        combined,
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print("COMBINATION COMPLETE")
    print("=" * 70)
    print()

    print(
        f"Sources combined: {len(parts)}"
    )

    print(
        f"Total characters: "
        f"{len(combined):,}"
    )

    print()
    print(
        f"Saved: {OUTPUT}"
    )
    print()


if __name__ == "__main__":
    main()