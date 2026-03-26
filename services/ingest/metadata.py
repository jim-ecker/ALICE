"""PDF metadata extraction and LLM-based inference for local folder ingestion."""
from __future__ import annotations

import json
import re
from pathlib import Path


def extract_pdf_metadata(pdf_path: Path) -> dict:
    """Extract metadata from a PDF's /Info header dictionary and first-page text.

    Returns a dict with keys: title, authors, year, abstract.
    Values are empty strings where not found.
    """
    import pypdf

    meta: dict[str, str] = {"title": "", "authors": "", "year": "", "abstract": ""}
    first_page_text = ""

    try:
        reader = pypdf.PdfReader(str(pdf_path))
        info = reader.metadata or {}

        raw_title = info.get("/Title", "") or ""
        raw_author = info.get("/Author", "") or ""
        raw_subject = info.get("/Subject", "") or ""
        raw_date = info.get("/CreationDate", "") or ""

        meta["title"] = raw_title.strip()
        meta["authors"] = raw_author.strip()
        meta["abstract"] = raw_subject.strip()

        # Extract year from CreationDate string (e.g. "D:20210315..." or "2021-03-15")
        if raw_date:
            year_match = re.search(r"(\d{4})", raw_date)
            if year_match:
                meta["year"] = year_match.group(1)

        # Extract first-page text for LLM inference
        if reader.pages:
            first_page_text = reader.pages[0].extract_text() or ""
    except Exception:
        pass

    meta["_first_page_text"] = first_page_text[:3000]
    return meta


def extract_excel_metadata(path: Path) -> dict:
    """Extract metadata from an Excel workbook's built-in properties."""
    import openpyxl
    meta: dict[str, str] = {"title": "", "authors": "", "year": "", "abstract": ""}
    try:
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        props = wb.properties
        meta["title"] = props.title or ""
        meta["authors"] = props.creator or ""
        if props.created:
            meta["year"] = str(props.created.year)
        wb.close()
    except Exception:
        pass
    if not meta["title"]:
        meta["title"] = path.stem
    return meta


def infer_missing_metadata(partial_meta: dict, llm) -> dict:
    """Use the LLM to fill in missing metadata fields from first-page text.

    Only called when title or authors is absent. Returns updated dict.
    Fields inferred by LLM are marked with a '_inferred' set in the result.
    """
    first_page_text = partial_meta.get("_first_page_text", "")
    if not first_page_text:
        return partial_meta

    missing = [k for k in ("title", "authors", "year", "abstract") if not partial_meta.get(k)]
    if not missing:
        return partial_meta

    prompt = (
        f"You are a research librarian. Based only on the text below (the first page of a PDF), "
        f"extract the following fields: {', '.join(missing)}.\n\n"
        f"Return ONLY a JSON object with those keys. Use empty string if you cannot determine a value.\n\n"
        f"TEXT:\n{first_page_text}"
    )
    messages = [
        {"role": "system", "content": "You extract bibliographic metadata from academic document text. Return only valid JSON."},
        {"role": "user", "content": prompt},
    ]

    inferred_fields: set[str] = set()
    try:
        response = llm.chat(messages, 400)
        # Extract JSON from response (may be wrapped in markdown code block)
        json_match = re.search(r"\{[^{}]+\}", response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            for key in missing:
                value = (data.get(key) or "").strip()
                if value:
                    partial_meta[key] = value
                    inferred_fields.add(key)
    except Exception:
        pass

    partial_meta["_inferred_fields"] = inferred_fields
    return partial_meta
