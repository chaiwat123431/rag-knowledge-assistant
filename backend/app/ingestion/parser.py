"""Text extraction from source documents.

Supports the formats listed in PLANNING.md's Phase 1 scope: PDF, plain text,
and markdown. No OCR — a PDF with no extractable text layer just yields an
empty (or near-empty) string, per the "no scanned/image-only PDFs" non-goal.
"""

from pathlib import Path

from pypdf import PdfReader

SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md"}
SUPPORTED_PDF_EXTENSIONS = {".pdf"}


class UnsupportedFileTypeError(ValueError):
    """Raised when the file extension isn't one we know how to parse."""


def extract_text(file_path: str) -> str:
    """Extract raw text from a document.

    Args:
        file_path: path to a .pdf, .txt, or .md file.

    Returns:
        The extracted text as a single string. For PDFs, pages are joined
        with newlines.

    Raises:
        FileNotFoundError: if `file_path` doesn't exist.
        UnsupportedFileTypeError: if the file extension isn't supported.
    """
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    extension = path.suffix.lower()

    if extension in SUPPORTED_TEXT_EXTENSIONS:
        return path.read_text(encoding="utf-8")

    if extension in SUPPORTED_PDF_EXTENSIONS:
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)

    supported = sorted(SUPPORTED_TEXT_EXTENSIONS | SUPPORTED_PDF_EXTENSIONS)
    raise UnsupportedFileTypeError(
        f"Unsupported file type '{extension}' for {file_path}. "
        f"Supported extensions: {supported}"
    )
