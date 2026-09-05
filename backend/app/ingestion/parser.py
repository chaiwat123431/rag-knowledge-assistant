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


class TextDecodingError(ValueError):
    """Raised when a .txt/.md file can't be decoded as text."""


class PdfExtractionError(ValueError):
    """Raised when a .pdf file can't be parsed (corrupt, malformed, encrypted...)."""


def extract_text(file_path: str) -> str:
    """Extract raw text from a document.

    Args:
        file_path: path to a .pdf, .txt, or .md file.

    Returns:
        The extracted text as a single string. For PDFs, pages are joined
        with newlines.

    Raises:
        FileNotFoundError: if `file_path` doesn't exist.
        IsADirectoryError: if `file_path` is a directory.
        OSError: for other access failures (permission denied, a device
            file, etc). Catch this (its parent class) to handle any
            access problem generically rather than catching
            FileNotFoundError alone.
        UnsupportedFileTypeError: if the file extension isn't supported.
        TextDecodingError: if a .txt/.md file isn't valid UTF-8 text.
        PdfExtractionError: if a .pdf file can't be parsed.
    """
    path = Path(file_path)
    extension = path.suffix.lower()

    # No manual exists()/is_file() pre-check: read_text() and PdfReader()
    # already open the path themselves and raise the correctly-typed
    # FileNotFoundError/IsADirectoryError/PermissionError natively, so a
    # separate check here would just add a stat() call and a TOCTOU race
    # against the actual read without adding any real safety.

    if extension in SUPPORTED_TEXT_EXTENSIONS:
        try:
            # utf-8-sig transparently strips a leading UTF-8 BOM (common
            # from Windows editors) while still reading plain UTF-8 fine.
            return path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError as exc:
            raise TextDecodingError(
                f"Could not decode {file_path} as UTF-8 text: {exc}"
            ) from exc

    if extension in SUPPORTED_PDF_EXTENSIONS:
        try:
            reader = PdfReader(str(path))
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(pages)
        except OSError:
            # Missing file / directory / permission-denied — not a
            # corrupt PDF, so let the native exception through as-is.
            raise
        except Exception as exc:
            raise PdfExtractionError(
                f"Could not extract text from PDF {file_path}: {exc}"
            ) from exc

    # Only checked here: for a supported extension we let read_text()/
    # PdfReader() report a missing path natively (see above); for an
    # unsupported one there's no read attempt to raise it for us.
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    supported = sorted(SUPPORTED_TEXT_EXTENSIONS | SUPPORTED_PDF_EXTENSIONS)
    raise UnsupportedFileTypeError(
        f"Unsupported file type '{extension}' for {file_path}. "
        f"Supported extensions: {supported}"
    )
