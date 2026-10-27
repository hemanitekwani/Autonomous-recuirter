"""Upload validation and text extraction (tests 3-5)."""
from __future__ import annotations

import pytest

from src.resume_extractor import (
    ResumeError,
    extract_text,
    resolve_stored_path,
    save_resume,
    validate_upload,
)
from tests.conftest import make_docx, make_pdf, sample_text


def test_pdf_extraction(tmp_path):
    data = make_pdf(sample_text("priya_sharma"), tmp_path / "r.pdf").read_bytes()
    ext = validate_upload("Resume.PDF", data, 5, "application/pdf")
    text = extract_text(data, ext)
    assert "Priya Sharma" in text and "Senior ML Engineer" in text


def test_docx_extraction(tmp_path):
    data = make_docx(sample_text("rahul_verma"), tmp_path / "r.docx").read_bytes()
    ext = validate_upload("cv.docx", data, 5)
    assert ext == ".docx"
    assert "Spring Boot" in extract_text(data, ext)


@pytest.mark.parametrize("name", ["resume.exe", "resume.txt", "resume", "resume.pdf.exe"])
def test_rejects_bad_extension(name):
    with pytest.raises(ResumeError):
        validate_upload(name, b"%PDF-1.7 x", 5)


def test_rejects_oversized_upload():
    with pytest.raises(ResumeError, match="limit"):
        validate_upload("r.pdf", b"%PDF-" + b"0" * (1024 * 1024 + 10), 1)


def test_rejects_content_that_does_not_match_extension():
    with pytest.raises(ResumeError, match="not a valid PDF"):
        validate_upload("r.pdf", b"MZ\x90\x00 executable", 5)
    with pytest.raises(ResumeError):
        validate_upload("r.docx", b"PK\x03\x04garbage", 5)
    with pytest.raises(ResumeError, match="empty"):
        validate_upload("r.pdf", b"", 5)


def test_rejects_mismatched_mime():
    with pytest.raises(ResumeError):
        validate_upload("r.pdf", b"%PDF-1.7", 5, "text/html")


def test_rejects_pdf_without_text(tmp_path):
    data = make_pdf(" ", tmp_path / "blank.pdf").read_bytes()
    with pytest.raises(ResumeError, match="No readable text"):
        extract_text(data, ".pdf")


def test_find_email_from_resume_text():
    from src.resume_extractor import find_email

    assert find_email(sample_text("priya_sharma")) == "priya.sharma@example.com"
    assert find_email("Contact: Jane.Doe+cv@Mail.Example.co.in | +91 99999") == "jane.doe+cv@mail.example.co.in"
    assert find_email("No contact details here") is None


def test_storage_uses_server_generated_names_and_blocks_traversal(tmp_path):
    path = save_resume(b"%PDF-1.7", ".pdf", tmp_path)
    assert path.parent == tmp_path.resolve() and len(path.stem) == 32
    assert resolve_stored_path(path.name, tmp_path) == path
    with pytest.raises(ResumeError):
        resolve_stored_path("../../etc/passwd", tmp_path)
