import os
import re
import json
import time
import hashlib
from pathlib import Path
from urllib.parse import quote

import requests


# ============================================================
# ASTRA INTERNET ARCHIVE CORPUS BUILDER
# ============================================================

CONFIG_FILE = "archive_config.json"

DEFAULT_CONFIG = {
    "max_items": 50,
    "min_text_chars": 5000,

    "sleep_seconds": 1.0,

    "search_queries": [
        "mediatype:texts AND year:[1800 TO 1929]"
    ],

    "allowed_rights_keywords": [
        "public domain",
        "public-domain",
        "creative commons",
        "cc0",
        "cc by",
        "cc-by",
        "cc by-sa",
        "cc-by-sa",
        "open access"
    ]
}


# ============================================================
# PATHS
# ============================================================

DATA_DIR = Path("data")
BOOK_DIR = DATA_DIR / "archive_books"

TRAINING_FILE = DATA_DIR / "training.txt"
SOURCES_FILE = DATA_DIR / "sources.json"


# ============================================================
# HTTP SESSION
# ============================================================

SESSION = requests.Session()

SESSION.headers.update({
    "User-Agent": (
        "AstraCorpusBuilder/1.0 "
        "(educational corpus-building project)"
    )
})


# ============================================================
# CONFIG
# ============================================================

def load_config():
    if not os.path.exists(CONFIG_FILE):
        with open(
            CONFIG_FILE,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                DEFAULT_CONFIG,
                f,
                indent=4
            )

        print(
            f"Created {CONFIG_FILE}. "
            "Edit it and run this program again."
        )

        return DEFAULT_CONFIG

    with open(
        CONFIG_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        config = json.load(f)

    return config


# ============================================================
# INTERNET ARCHIVE SEARCH
# ============================================================

def search_archive(query, rows=50):
    print()
    print("Searching Internet Archive:")
    print(query)
    print()

    url = (
        "https://archive.org/advancedsearch.php"
        "?q="
        + quote(query)
        + "&fl[]=identifier"
        + "&fl[]=title"
        + "&fl[]=creator"
        + "&fl[]=year"
        + "&fl[]=description"
        + "&fl[]=rights"
        + "&rows="
        + str(rows)
        + "&page=1"
        + "&output=json"
    )

    response = SESSION.get(
        url,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    docs = data.get(
        "response",
        {}
    ).get(
        "docs",
        []
    )

    return docs


# ============================================================
# METADATA
# ============================================================

def get_metadata(identifier):
    url = (
        "https://archive.org/metadata/"
        + quote(identifier)
    )

    response = SESSION.get(
        url,
        timeout=30
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# RIGHTS CHECK
# ============================================================

def collect_rights_text(metadata):
    values = []

    for key in [
        "rights",
        "licenseurl",
        "license",
        "description",
        "notes"
    ]:
        value = metadata.get(key)

        if isinstance(value, list):
            values.extend(
                str(x)
                for x in value
            )

        elif value:
            values.append(str(value))

    return " ".join(values).lower()


def is_allowed(metadata, config):
    rights_text = collect_rights_text(
        metadata
    )

    allowed_keywords = [
        x.lower()
        for x in config.get(
            "allowed_rights_keywords",
            []
        )
    ]

    for keyword in allowed_keywords:
        if keyword in rights_text:
            return True

    return False


# ============================================================
# FIND TEXT FILE
# ============================================================

def find_text_file(metadata):
    files = metadata.get(
        "files",
        []
    )

    candidates = []

    for file_info in files:

        name = file_info.get(
            "name",
            ""
        )

        lower = name.lower()

        # Prefer actual text files.
        if lower.endswith(".txt"):
            candidates.append(
                (
                    0,
                    file_info
                )
            )

        # IA sometimes exposes OCR in *_djvu.txt.
        elif lower.endswith(
            "_djvu.txt"
        ):
            candidates.append(
                (
                    -1,
                    file_info
                )
            )

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: x[0]
    )

    return candidates[0][1]


# ============================================================
# DOWNLOAD
# ============================================================

def download_text(
    identifier,
    file_info
):
    filename = file_info.get(
        "name"
    )

    if not filename:
        return None

    encoded_identifier = quote(
        identifier,
        safe=""
    )

    encoded_filename = quote(
        filename,
        safe=""
    )

    url = (
        "https://archive.org/download/"
        f"{encoded_identifier}/"
        f"{encoded_filename}"
    )

    print(
        "Downloading:",
        filename
    )

    response = SESSION.get(
        url,
        timeout=120
    )

    response.raise_for_status()

    return response.text


# ============================================================
# TEXT CLEANING
# ============================================================

def normalize_text(text):
    if not text:
        return ""

    # Normalize newlines.
    text = text.replace(
        "\r\n",
        "\n"
    )

    text = text.replace(
        "\r",
        "\n"
    )

    # Remove null bytes.
    text = text.replace(
        "\x00",
        ""
    )

    # Normalize common OCR artifacts.
    text = text.replace(
        "\u00ad",
        ""
    )

    # Join words broken across lines:
    #
    # exam-
    # ple
    #
    # becomes:
    #
    # example
    text = re.sub(
        r"(\w)-\n(\w)",
        r"\1\2",
        text
    )

    # Collapse spaces around newlines.
    text = re.sub(
        r"[ \t]+\n",
        "\n",
        text
    )

    text = re.sub(
        r"\n[ \t]+",
        "\n",
        text
    )

    # Remove extremely long runs of blank lines.
    text = re.sub(
        r"\n{4,}",
        "\n\n\n",
        text
    )

    # Remove repeated spaces.
    text = re.sub(
        r"[ \t]{2,}",
        " ",
        text
    )

    # Strip each line.
    lines = []

    for line in text.splitlines():
        line = line.strip()

        if line:
            lines.append(line)

    text = "\n".join(lines)

    return text.strip()


# ============================================================
# OCR JUNK FILTER
# ============================================================

def looks_like_junk(text):
    if len(text) < 500:
        return True

    # Ratio of printable characters.
    printable = sum(
        1
        for c in text
        if c.isprintable()
        or c in "\n\t"
    )

    ratio = printable / max(
        len(text),
        1
    )

    if ratio < 0.90:
        return True

    # A giant amount of replacement characters
    # usually means broken encoding.
    replacement_count = text.count(
        "\ufffd"
    )

    if replacement_count > 20:
        return True

    return False


# ============================================================
# SAFE FILENAME
# ============================================================

def safe_filename(identifier):
    filename = re.sub(
        r"[^a-zA-Z0-9._-]+",
        "_",
        identifier
    )

    return filename[:150]


# ============================================================
# HASH / DEDUPLICATION
# ============================================================

def text_hash(text):
    return hashlib.sha256(
        text.encode(
            "utf-8",
            errors="ignore"
        )
    ).hexdigest()


# ============================================================
# LOAD EXISTING SOURCES
# ============================================================

def load_sources():
    if not SOURCES_FILE.exists():
        return []

    try:
        with open(
            SOURCES_FILE,
            "r",
            encoding="utf-8"
        ) as f:
            data = json.load(f)

        if isinstance(data, list):
            return data

    except Exception:
        pass

    return []


# ============================================================
# SAVE SOURCES
# ============================================================

def save_sources(sources):
    with open(
        SOURCES_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            sources,
            f,
            indent=4,
            ensure_ascii=False
        )


# ============================================================
# BUILD TRAINING CORPUS
# ============================================================

def rebuild_training_file():
    print()
    print("Building training.txt...")
    print()

    text_files = sorted(
        BOOK_DIR.glob("*.txt")
    )

    seen_hashes = set()

    pieces = []

    for path in text_files:

        try:
            text = path.read_text(
                encoding="utf-8",
                errors="ignore"
            )
        except Exception as exc:
            print(
                "Could not read:",
                path,
                exc
            )
            continue

        text = normalize_text(
            text
        )

        if len(text) < 500:
            continue

        digest = text_hash(
            text
        )

        if digest in seen_hashes:
            continue

        seen_hashes.add(
            digest
        )

        pieces.append(
            text
        )

    if not pieces:
        print(
            "No usable text files found."
        )
        return

    with open(
        TRAINING_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        for index, text in enumerate(
            pieces
        ):

            f.write(
                "\n\n"
            )

            f.write(
                f"=== CORPUS DOCUMENT "
                f"{index + 1} ===\n\n"
            )

            f.write(
                text
            )

            f.write(
                "\n\n"
            )

    total_chars = sum(
        len(x)
        for x in pieces
    )

    print(
        f"Documents: {len(pieces)}"
    )

    print(
        f"Characters: {total_chars:,}"
    )

    print(
        f"Saved: {TRAINING_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("========================================")
    print("   ASTRA ARCHIVE.ORG CORPUS BUILDER")
    print("========================================")
    print()

    config = load_config()

    DATA_DIR.mkdir(
        exist_ok=True
    )

    BOOK_DIR.mkdir(
        exist_ok=True
    )

    sources = load_sources()

    known_identifiers = {
        item.get("identifier")
        for item in sources
    }

    all_documents = []

    for query in config.get(
        "search_queries",
        []
    ):

        try:
            documents = search_archive(
                query,
                rows=int(
                    config.get(
                        "max_items",
                        50
                    )
                )
            )

            all_documents.extend(
                documents
            )

        except Exception as exc:

            print(
                "Search failed:",
                exc
            )

    # Deduplicate search results.
    unique = {}

    for document in all_documents:

        identifier = document.get(
            "identifier"
        )

        if identifier:
            unique[
                identifier
            ] = document

    documents = list(
        unique.values()
    )

    print()
    print(
        f"Found {len(documents)} candidate items."
    )
    print()

    downloaded = 0

    for document in documents:

        if downloaded >= int(
            config.get(
                "max_items",
                50
            )
        ):
            break

        identifier = document.get(
            "identifier"
        )

        if not identifier:
            continue

        if identifier in known_identifiers:
            print(
                "Already processed:",
                identifier
            )
            continue

        title = document.get(
            "title",
            identifier
        )

        print()
        print("----------------------------------------")
        print("Item:", title)
        print("ID:", identifier)

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        try:
            metadata = get_metadata(
                identifier
            )
        except Exception as exc:

            print(
                "Metadata failed:",
                exc
            )

            continue

        # ----------------------------------------------------
        # Rights
        # ----------------------------------------------------

        if not is_allowed(
            metadata,
            config
        ):

            print(
                "Skipped: no allowed "
                "public/open license found."
            )

            continue

        print(
            "Rights check: PASSED"
        )

        # ----------------------------------------------------
        # Text file
        # ----------------------------------------------------

        file_info = find_text_file(
            metadata
        )

        if file_info is None:

            print(
                "Skipped: no TXT/OCR text file."
            )

            continue

        try:
            text = download_text(
                identifier,
                file_info
            )

        except Exception as exc:

            print(
                "Download failed:",
                exc
            )

            continue

        # ----------------------------------------------------
        # Clean
        # ----------------------------------------------------

        text = normalize_text(
            text
        )

        minimum_chars = int(
            config.get(
                "min_text_chars",
                5000
            )
        )

        if len(text) < minimum_chars:

            print(
                f"Skipped: only "
                f"{len(text)} characters."
            )

            continue

        if looks_like_junk(
            text
        ):

            print(
                "Skipped: text appears "
                "corrupted or unusable."
            )

            continue

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        local_name = (
            safe_filename(
                identifier
            )
            + ".txt"
        )

        local_path = (
            BOOK_DIR
            / local_name
        )

        local_path.write_text(
            text,
            encoding="utf-8"
        )

        # ----------------------------------------------------
        # Source record
        # ----------------------------------------------------

        rights = (
            metadata.get(
                "rights"
            )
            or document.get(
                "rights"
            )
            or "Not specified"
        )

        source_record = {
            "identifier": identifier,
            "title": title,
            "creator": document.get(
                "creator",
                ""
            ),
            "year": document.get(
                "year",
                ""
            ),
            "rights": rights,
            "archive_url": (
                "https://archive.org/details/"
                + identifier
            ),
            "downloaded_file": str(
                local_path
            ),
            "characters": len(text),
            "sha256": text_hash(text)
        }

        sources.append(
            source_record
        )

        known_identifiers.add(
            identifier
        )

        save_sources(
            sources
        )

        downloaded += 1

        print(
            f"Saved: {local_path}"
        )

        print(
            f"Characters: {len(text):,}"
        )

        print(
            f"Progress: "
            f"{downloaded}/"
            f"{config.get('max_items', 50)}"
        )

        time.sleep(
            float(
                config.get(
                    "sleep_seconds",
                    1.0
                )
            )
        )

    # ========================================================
    # BUILD TRAINING FILE
    # ========================================================

    rebuild_training_file()

    print()
    print("========================================")
    print("             COMPLETE")
    print("========================================")
    print()

    print(
        f"Books/texts downloaded: {downloaded}"
    )

    print(
        f"Individual files: {BOOK_DIR}"
    )

    print(
        f"Training corpus: {TRAINING_FILE}"
    )

    print(
        f"Source records: {SOURCES_FILE}"
    )

    print()


if __name__ == "__main__":
    main()