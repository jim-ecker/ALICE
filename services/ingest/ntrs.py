from dataclasses import dataclass
import logging
from pathlib import Path
import time

import requests

BASE_URL = "https://ntrs.nasa.gov/api"
LOGGER = logging.getLogger(__name__)

SEARCH_TIMEOUT_SECONDS = 30
PDF_CONNECT_TIMEOUT_SECONDS = 15
PDF_READ_TIMEOUT_SECONDS = 30
PDF_TOTAL_TIMEOUT_SECONDS = 45
PDF_DOWNLOAD_ATTEMPTS = 2

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
    resume_offset: int = 0

    def __post_init__(self):
        if self.subject_categories is None:
            self.subject_categories = []


def search(
    query: str,
    center: str | None = None,
    max_docs: int = 20,
    author: str | None = None,
    offset: int = 0,
) -> tuple[list[NTRSRecord], int]:
    """Search NTRS for records with downloadable PDFs.

    Returns (records, next_page_from) where next_page_from is the raw NTRS
    result offset after the last raw result examined — pass it as `offset`
    to resume.
    """
    records = []
    page_from = offset
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

        LOGGER.info(
            "NTRS search request: query=%r author=%r center=%r offset=%d size=%d",
            query,
            author,
            center,
            page_from,
            page_size,
        )
        response = requests.get(
            f"{BASE_URL}/citations/search",
            params=params,
            timeout=SEARCH_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()

        results = data.get("results", [])
        LOGGER.info("NTRS search response: %d raw results", len(results))
        if not results:
            break

        consumed_results = 0
        for result in results:
            consumed_results += 1
            if not result.get("downloadsAvailable"):
                continue
            downloads = result.get("downloads", [])
            pdf = next((d for d in downloads if d.get("mimetype") == "application/pdf"), None)
            if not pdf:
                pdf = next((d for d in downloads if "pdf" in d.get("links", {})), None)
            if not pdf:
                continue
            raw_name = pdf["name"]
            if not raw_name.lower().endswith(".pdf"):
                raw_name = Path(raw_name).stem + ".pdf"
            records.append(NTRSRecord(
                id=result["id"],
                title=result.get("title", ""),
                filename=raw_name,
                download_path=pdf["links"]["pdf"],
                citation_url=f"https://ntrs.nasa.gov/citations/{result['id']}",
                subject_categories=result.get("subjectCategories", []),
                resume_offset=page_from + consumed_results,
            ))
            if len(records) >= max_docs:
                break

        page_from += consumed_results
        if len(results) < page_size:
            break

    return records[:max_docs], page_from


def ntrs_id_from_url(citation_url: str) -> str | None:
    """Extract the NTRS numeric ID from a citation URL."""
    parts = citation_url.rstrip("/").rsplit("/", 1)
    return parts[-1] if len(parts) == 2 else None


def get_title(citation_url: str) -> str | None:
    """Fetch the title of a document from NTRS given its citation URL."""
    ntrs_id = ntrs_id_from_url(citation_url)
    if not ntrs_id:
        return None
    response = requests.get(f"{BASE_URL}/citations/{ntrs_id}", timeout=SEARCH_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json().get("title")


def download_pdf(record: NTRSRecord, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / record.filename
    temp_dest = dest.with_suffix(dest.suffix + ".part")

    if dest.exists():
        LOGGER.info("Skipping existing PDF: %s", dest)
        return dest

    url = f"https://ntrs.nasa.gov{record.download_path}"

    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (compatible; ALICE-bot/1.0; +https://github.com/yourorg/alice)'
    })

    LOGGER.info("Starting PDF download: title=%r url=%s dest=%s", record.title, url, dest)

    for attempt in range(PDF_DOWNLOAD_ATTEMPTS):
        try:
            LOGGER.info(
                "PDF download attempt %d/%d: %s",
                attempt + 1,
                PDF_DOWNLOAD_ATTEMPTS,
                url,
            )
            response = session.get(
                url,
                timeout=(PDF_CONNECT_TIMEOUT_SECONDS, PDF_READ_TIMEOUT_SECONDS),
                stream=True,
            )
            response.raise_for_status()
            started_at = time.monotonic()

            with temp_dest.open("wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if time.monotonic() - started_at > PDF_TOTAL_TIMEOUT_SECONDS:
                        raise requests.exceptions.Timeout(
                            f"timed out after {PDF_TOTAL_TIMEOUT_SECONDS}s with partial or no PDF data"
                        )
                    if chunk:
                        f.write(chunk)

            temp_dest.replace(dest)
            LOGGER.info("Completed PDF download: %s", dest)
            return dest

        except requests.exceptions.RequestException as e:
            temp_dest.unlink(missing_ok=True)
            LOGGER.warning(
                "PDF download failed on attempt %d/%d for %s: %s",
                attempt + 1,
                PDF_DOWNLOAD_ATTEMPTS,
                url,
                e,
            )
            if attempt == PDF_DOWNLOAD_ATTEMPTS - 1:
                raise e
            time.sleep(1)
        finally:
            try:
                response.close()  # type: ignore[name-defined]
            except Exception:
                pass

    return dest
