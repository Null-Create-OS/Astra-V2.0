import re
import sys
from pathlib import Path


MIN_SECTION_LENGTH = 1000


# ============================================================
# BASIC CLEANING
# ============================================================

def clean_text(text):
    if not text:
        return ""

    # Normalize line endings.
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove null characters.
    text = text.replace("\x00", "")

    # Fix extremely broken OCR spacing.
    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    # Remove repeated blank lines.
    text = re.sub(
        r"\n{4,}",
        "\n\n",
        text,
    )

    # Remove obvious web junk.
    junk_patterns = [
        r"click here",
        r"sign up now",
        r"subscribe now",
        r"privacy policy",
        r"terms of service",
        r"cookie policy",
        r"all rights reserved",
        r"follow us on",
        r"share this page",
        r"advertisement",
    ]

    for pattern in junk_patterns:
        text = re.sub(
            pattern,
            "",
            text,
            flags=re.IGNORECASE,
        )

    return text.strip()


# ============================================================
# OCR LINE REPAIR
# ============================================================

def repair_lines(text):
    lines = text.splitlines()

    output = []
    paragraph = []

    for line in lines:

        line = line.strip()

        if not line:
            if paragraph:
                output.append(
                    " ".join(paragraph)
                )
                paragraph = []

            output.append("")
            continue

        # Lines consisting almost entirely of
        # punctuation/numbers are usually OCR noise.
        letters = sum(
            c.isalpha()
            for c in line
        )

        if letters < 2:
            continue

        paragraph.append(line)

    if paragraph:
        output.append(
            " ".join(paragraph)
        )

    return "\n".join(output)


# ============================================================
# REMOVE BAD SOURCE BLOCKS
# ============================================================

def remove_metadata(text):
    patterns = [
        r"^SOURCE:.*$",
        r"^TITLE:.*$",
        r"^CREATOR:.*$",
        r"^YEAR:.*$",
    ]

    for pattern in patterns:
        text = re.sub(
            pattern,
            "",
            text,
            flags=re.MULTILINE,
        )

    # Remove separator lines.
    text = re.sub(
        r"^={5,}$",
        "",
        text,
        flags=re.MULTILINE,
    )

    return text


# ============================================================
# QUALITY FILTER
# ============================================================

def useful_section(text):
    if len(text) < MIN_SECTION_LENGTH:
        return False

    words = re.findall(
        r"\b[A-Za-z]{2,}\b",
        text,
    )

    if len(words) < 150:
        return False

    alphabetic = sum(
        c.isalpha()
        for c in text
    )

    ratio = alphabetic / max(
        len(text),
        1,
    )

    if ratio < 0.40:
        return False

    # Too many URLs = probably junk.
    urls = len(
        re.findall(
            r"https?://",
            text,
            flags=re.IGNORECASE,
        )
    )

    if urls > 10:
        return False

    return True


# ============================================================
# PROCESS
# ============================================================

def clean_file(input_path):
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"File not found: {input_path}"
        )

    print(
        f"Reading: {input_path}"
    )

    text = input_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    print(
        f"Input characters: {len(text):,}"
    )

    text = remove_metadata(text)
    text = clean_text(text)
    text = repair_lines(text)
    text = clean_text(text)

    # Split into reasonably sized sections.
    sections = re.split(
        r"\n\s*\n",
        text,
    )

    good_sections = []

    for section in sections:

        section = section.strip()

        if useful_section(section):
            good_sections.append(
                section
            )

    output = "\n\n".join(
        good_sections
    )

    output_path = (
        input_path.parent
        / "archive_corpus_clean.txt"
    )

    output_path.write_text(
        output,
        encoding="utf-8",
    )

    print()
    print("=" * 60)
    print("CLEANING COMPLETE")
    print("=" * 60)
    print()
    print(
        f"Accepted sections: "
        f"{len(good_sections)}"
    )
    print(
        f"Output characters: "
        f"{len(output):,}"
    )
    print()
    print(
        f"Saved: {output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    if len(sys.argv) != 2:
        print()
        print(
            "Usage:"
        )
        print(
            "  python text_cleaner.py archive_corpus_raw.txt"
        )
        print()
        return

    clean_file(
        sys.argv[1]
    )


if __name__ == "__main__":
    main()