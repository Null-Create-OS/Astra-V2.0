import os
import json
import time
import re
import requests

from pathlib import Path
from urllib.parse import quote


# ============================================================
# CONFIG
# ============================================================

CONFIG_FILE = "archive_config.json"

DEFAULT_CONFIG = {
    "output_dir": "archive_downloads",
    "min_text_chars": 5000,
    "max_items": 100,
    "timeout": 20,
    "delay": 1.0,

    "preferred_formats": [
        "_djvu.txt",
        "_djvu.xml",
        "_hocr.html",
        ".txt"
    ],

    "blocked_extensions": [
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".webp",
        ".bmp",
        ".tif",
        ".tiff",
        ".jp2",
        ".pdf",
        ".epub",
        ".mobi",
        ".azw",
        ".mp3",
        ".wav",
        ".ogg",
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".zip",
        ".tar",
        ".gz",
        ".7z"
    ]
}


# ============================================================
# CONFIG LOADING
# ============================================================

def load_config():
    if not os.path.exists(CONFIG_FILE):
        print(f"Config not found: {CONFIG_FILE}")
        print("Using default configuration.")
        return DEFAULT_CONFIG.copy()

    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            user_config = json.load(f)

        config = DEFAULT_CONFIG.copy()
        config.update(user_config)

        return config

    except Exception as e:
        print(f"Could not read {CONFIG_FILE}: {e}")
        print("Using default configuration.")
        return DEFAULT_CONFIG.copy()


# ============================================================
# SESSION
# ============================================================

def create_session():
    session = requests.Session()

    session.headers.update({
        "User-Agent": (
            "AstraCorpusBuilder/1.0 "
            "(educational research corpus downloader)"
        ),
        "Accept": "*/*",
        "Connection": "keep-alive"
    })

    return session


# ============================================================
# SAFE REQUEST
# ============================================================

def safe_get(session, url, timeout):
    """
    Download a URL safely.

    Returns:
        response
        None if unavailable
    """

    try:
        response = session.get(
            url,
            timeout=timeout,
            allow_redirects=True
        )

        status = response.status_code

        # ----------------------------------------------------
        # AUTH / ACCESS
        # ----------------------------------------------------

        if status in (401, 403):
            print(
                f"      Skipping unavailable file "
                f"(HTTP {status})"
            )
            return None

        # ----------------------------------------------------
        # NOT FOUND
        # ----------------------------------------------------

        if status == 404:
            print("      Skipping missing file (HTTP 404)")
            return None

        # ----------------------------------------------------
        # SERVER ERRORS
        # ----------------------------------------------------

        if 500 <= status <= 599:
            print(
                f"      Skipping server-error file "
                f"(HTTP {status})"
            )
            return None

        # ----------------------------------------------------
        # OTHER HTTP ERRORS
        # ----------------------------------------------------

        if status != 200:
            print(
                f"      Skipping file "
                f"(HTTP {status})"
            )
            return None

        return response

    except requests.exceptions.Timeout:
        print("      Skipping file (timeout)")
        return None

    except requests.exceptions.ConnectionError:
        print("      Skipping file (connection error)")
        return None

    except requests.exceptions.RequestException as e:
        print(
            f"      Skipping file "
            f"(request error: {e})"
        )
        return None


# ============================================================
# TEXT EXTRACTION
# ============================================================

def clean_text(text):
    if not text:
        return ""

    # Remove null bytes
    text = text.replace("\x00", " ")

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove excessive whitespace
    text = re.sub(r"[ \t]+", " ", text)

    # Remove huge runs of blank lines
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

    return text.strip()


def extract_djvu_xml(xml_text):
    """
    Extract readable text from Internet Archive DJVU XML.
    """

    if not xml_text:
        return ""

    # Remove XML tags
    text = re.sub(
        r"<[^>]+>",
        " ",
        xml_text
    )

    # Decode common HTML/XML entities
    replacements = {
        "&amp;": "&",
        "&lt;": "<",
        "&gt;": ">",
        "&quot;": '"',
        "&apos;": "'",
        "&#39;": "'",
        "&#160;": " "
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return clean_text(text)


def extract_hocr(html):
    """
    Basic OCR extraction from Archive.org hOCR HTML.
    """

    if not html:
        return ""

    # Remove scripts/styles
    html = re.sub(
        r"<script.*?</script>",
        " ",
        html,
        flags=re.I | re.S
    )

    html = re.sub(
        r"<style.*?</style>",
        " ",
        html,
        flags=re.I | re.S
    )

    # Remove tags
    text = re.sub(
        r"<[^>]+>",
        " ",
        html
    )

    return clean_text(text)


# ============================================================
# FILE NAME FILTER
# ============================================================

def is_blocked_filename(filename, config):
    lower = filename.lower()

    for ext in config.get(
        "blocked_extensions",
        []
    ):
        if lower.endswith(ext.lower()):
            return True

    return False


def looks_like_text_format(filename, config):
    lower = filename.lower()

    preferred = config.get(
        "preferred_formats",
        []
    )

    for suffix in preferred:
        if lower.endswith(suffix.lower()):
            return True

    return False


# ============================================================
# DOWNLOAD TEXT FORMAT
# ============================================================

def download_format(
    session,
    item_id,
    filename,
    config
):
    encoded_item = quote(item_id, safe="")

    url = (
        f"https://archive.org/download/"
        f"{encoded_item}/"
        f"{quote(filename, safe='')}"
    )

    print(f"   Testing: {filename}")

    response = safe_get(
        session,
        url,
        config["timeout"]
    )

    if response is None:
        return None

    raw = response.content

    # --------------------------------------------------------
    # Decode
    # --------------------------------------------------------

    try:
        text = raw.decode(
            "utf-8",
            errors="ignore"
        )
    except Exception:
        text = ""

    # --------------------------------------------------------
    # Format-specific extraction
    # --------------------------------------------------------

    lower = filename.lower()

    if lower.endswith(".xml"):
        text = extract_djvu_xml(text)

    elif lower.endswith(".html"):
        text = extract_hocr(text)

    else:
        text = clean_text(text)

    if not text:
        print("      Empty file.")
        return None

    minimum = int(
        config.get(
            "min_text_chars",
            5000
        )
    )

    if len(text) < minimum:
        print(
            f"      Too short "
            f"({len(text)} chars)"
        )
        return None

    return text


# ============================================================
# SAVE TEXT
# ============================================================

def save_text(
    output_dir,
    item_id,
    text
):
    os.makedirs(
        output_dir,
        exist_ok=True
    )

    safe_name = re.sub(
        r"[^a-zA-Z0-9_.-]+",
        "_",
        item_id
    )

    path = os.path.join(
        output_dir,
        safe_name + ".txt"
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:
        f.write(text)

    return path


# ============================================================
# PROCESS ITEM
# ============================================================

def process_item(
    session,
    item_id,
    title,
    config
):
    print()
    print("=" * 80)
    print(f"Processing: {item_id}")
    print(f"Title: {title}")

    # --------------------------------------------------------
    # Get metadata
    # --------------------------------------------------------

    metadata_url = (
        f"https://archive.org/metadata/"
        f"{quote(item_id, safe='')}"
    )

    response = safe_get(
        session,
        metadata_url,
        config["timeout"]
    )

    if response is None:
        print("   Could not access metadata. Skipping item.")
        return False

    try:
        metadata = response.json()
    except Exception:
        print("   Invalid metadata response. Skipping item.")
        return False

    files = metadata.get("files", [])

    if not files:
        print("   No files listed.")
        return False

    # --------------------------------------------------------
    # Find useful text formats
    # --------------------------------------------------------

    candidates = []

    preferred = config.get(
        "preferred_formats",
        []
    )

    for file_info in files:

        filename = file_info.get("name", "")

        if not filename:
            continue

        if is_blocked_filename(
            filename,
            config
        ):
            continue

        if not looks_like_text_format(
            filename,
            config
        ):
            continue

        lower = filename.lower()

        # Score format
        score = 0

        for index, suffix in enumerate(preferred):
            if lower.endswith(
                suffix.lower()
            ):
                score = len(preferred) - index
                break

        # Prefer files with actual OCR/text
        if "_djvu.txt" in lower:
            score += 100

        if "_djvu.xml" in lower:
            score += 90

        if "_hocr.html" in lower:
            score += 70

        if lower.endswith(".txt"):
            score += 60

        candidates.append(
            (score, filename)
        )

    candidates.sort(
        reverse=True
    )

    if not candidates:
        print("   No useful text formats found.")
        return False

    # --------------------------------------------------------
    # Try formats one by one
    # --------------------------------------------------------

    for score, filename in candidates:

        print(
            f"   Downloading: {filename}"
        )

        text = download_format(
            session,
            item_id,
            filename,
            config
        )

        if text is None:
            continue

        output_path = save_text(
            config["output_dir"],
            item_id,
            text
        )

        print(
            f"   SUCCESS: "
            f"{len(text):,} characters"
        )

        print(
            f"   Saved: {output_path}"
        )

        return True

    print(
        "   No accessible useful text "
        "formats were available."
    )

    return False


# ============================================================
# SEARCH ARCHIVE.ORG
# ============================================================

def search_archive(
    session,
    query,
    rows,
    timeout
):
    url = (
        "https://archive.org/advancedsearch.php"
        "?q="
        + quote(query)
        + "&fl[]=identifier"
        + "&fl[]=title"
        + "&fl[]=description"
        + "&fl[]=subject"
        + "&rows="
        + str(rows)
        + "&page=1"
        + "&output=json"
    )

    response = safe_get(
        session,
        url,
        timeout
    )

    if response is None:
        return []

    try:
        data = response.json()
    except Exception:
        print("Could not decode Archive.org search response.")
        return []

    docs = data.get(
        "response",
        {}
    ).get(
        "docs",
        []
    )

    return docs


# ============================================================
# MAIN
# ============================================================

def main():

    config = load_config()

    session = create_session()

    output_dir = config.get(
        "output_dir",
        "archive_downloads"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    queries = config.get(
        "queries",
        [
            "mediatype:texts"
        ]
    )

    max_items = int(
        config.get(
            "max_items",
            100
        )
    )

    delay = float(
        config.get(
            "delay",
            1.0
        )
    )

    timeout = int(
        config.get(
            "timeout",
            20
        )
    )

    print()
    print("=" * 80)
    print("ASTRA ARCHIVE.ORG CORPUS BUILDER")
    print("=" * 80)
    print()

    print(
        f"Output directory: {output_dir}"
    )

    print(
        f"Minimum text length: "
        f"{config.get('min_text_chars', 5000):,}"
    )

    print(
        f"Maximum items: {max_items}"
    )

    print()

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    seen = set()
    processed = 0
    successful = 0

    for query in queries:

        if processed >= max_items:
            break

        print()
        print(
            f"Searching Archive.org: {query}"
        )

        docs = search_archive(
            session,
            query,
            max_items,
            timeout
        )

        print(
            f"Found {len(docs)} results."
        )

        for doc in docs:

            if processed >= max_items:
                break

            item_id = doc.get(
                "identifier"
            )

            if not item_id:
                continue

            if item_id in seen:
                continue

            seen.add(item_id)

            title = doc.get(
                "title",
                item_id
            )

            success = process_item(
                session,
                item_id,
                title,
                config
            )

            processed += 1

            if success:
                successful += 1

            time.sleep(delay)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("SCRAPING COMPLETE")
    print("=" * 80)
    print()

    print(
        f"Items processed: {processed}"
    )

    print(
        f"Useful texts saved: {successful}"
    )

    print(
        f"Skipped/unavailable: "
        f"{processed - successful}"
    )

    print(
        f"Output: {output_dir}"
    )

    print()


if __name__ == "__main__":
    main()
