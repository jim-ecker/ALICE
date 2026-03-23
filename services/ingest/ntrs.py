from dataclasses import dataclass
from pathlib import Path

import requests

BASE_URL = "https://ntrs.nasa.gov/api"

CENTER_CODES = {
    "langley": "LaRC",
    "goddard": "GSFC",
    "ames": "ARC",
    "johnson": "JSC",
    "kennedy": "KSC",
    "marshall": "MSFC",
    "glenn": "GRC",
    "armstrong": "AFRC",
}


@dataclass
class NTRSRecord:
    id: int
    title: str
    filename: str
    download_path: str
    citation_url: str
    subject_categories: list[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.subject_categories is None:
            self.subject_categories = []


def search(
    query: str,
    center: str | None = None,
    max_docs: int = 20,
    author: str | None = None,
) -> list[NTRSRecord]:
    records = []
    page_from = 0
    page_size = 100

    while len(records) < max_docs:
        params = {
            "q": query,
            "page.size": page_size,
            "page.from": page_from,
            "disseminated": "DOCUMENT_AND_METADATA",
        }
        if center:
            params["center"] = center
        if author:
            params["author"] = author

        response = requests.get(f"{BASE_URL}/citations/search", params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        results = data.get("results", [])
        if not results:
            break

        for result in results:
            if not result.get("downloadsAvailable"):
                continue
            downloads = result.get("downloads", [])
            pdf = next((d for d in downloads if d.get("mimetype") == "application/pdf"), None)
            if not pdf:
                continue
            records.append(NTRSRecord(
                id=result["id"],
                title=result.get("title", ""),
                filename=pdf["name"],
                download_path=pdf["links"]["pdf"],
                citation_url=f"https://ntrs.nasa.gov/citations/{result['id']}",
                subject_categories=result.get("subjectCategories", []),
            ))
            if len(records) >= max_docs:
                break

        page_from += len(results)
        if len(results) < page_size:
            break

    return records[:max_docs]


def ntrs_id_from_url(citation_url: str) -> str | None:
    """Extract the NTRS numeric ID from a citation URL."""
    parts = citation_url.rstrip("/").rsplit("/", 1)
    return parts[-1] if len(parts) == 2 else None


def get_title(citation_url: str) -> str | None:
    """Fetch the title of a document from NTRS given its citation URL."""
    ntrs_id = ntrs_id_from_url(citation_url)
    if not ntrs_id:
        return None
    response = requests.get(f"{BASE_URL}/citations/{ntrs_id}", timeout=30)
    response.raise_for_status()
    return response.json().get("title")


def download_pdf(record: NTRSRecord, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / record.filename

    if dest.exists():
        return dest

    url = f"https://ntrs.nasa.gov{record.download_path}"

    # Try with session and retry logic for problematic downloads
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (compatible; ALICE-bot/1.0; +https://github.com/yourorg/alice)'
    })

    for attempt in range(3):
        try:
            response = session.get(url, timeout=(30, 180), stream=True)
            response.raise_for_status()

            with dest.open("wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            return dest

        except requests.exceptions.Timeout as e:
            if attempt == 2:  # Last attempt
                raise e
            # Wait before retry
            import time
            time.sleep(1)

    return dest
