import pytest
from pypdf import PdfWriter

from app.ingestion.parser import (
    PdfExtractionError,
    TextDecodingError,
    UnsupportedFileTypeError,
    extract_text,
)


def _make_pdf(path, text="Hello from a test PDF."):
    """Build a minimal one-page PDF with real extractable text.

    pypdf's writer has no drawing API, so we hand-assemble a page: a
    /Font resource (Helvetica) plus a content stream with a single Tj
    text-show operator. Without a font resource, pypdf's extraction can't
    map character codes and silently returns ''. Good enough to exercise
    extract_text's PDF branch end to end.
    """
    from pypdf.generic import ContentStream, DictionaryObject, NameObject

    writer = PdfWriter()
    page = writer.add_blank_page(width=200, height=200)

    font = DictionaryObject()
    font[NameObject("/Type")] = NameObject("/Font")
    font[NameObject("/Subtype")] = NameObject("/Type1")
    font[NameObject("/BaseFont")] = NameObject("/Helvetica")
    font_ref = writer._add_object(font)

    font_dict = DictionaryObject()
    font_dict[NameObject("/F1")] = font_ref
    resources = DictionaryObject()
    resources[NameObject("/Font")] = font_dict
    page[NameObject("/Resources")] = resources

    content = f"BT /F1 12 Tf 10 100 Td ({text}) Tj ET".encode("latin-1")
    stream = ContentStream(None, writer)
    stream.set_data(content)
    page[NameObject("/Contents")] = writer._add_object(stream)

    with open(path, "wb") as f:
        writer.write(f)


def test_extract_text_from_txt(tmp_path):
    file_path = tmp_path / "note.txt"
    file_path.write_text("Some plain text content.", encoding="utf-8")

    assert extract_text(str(file_path)) == "Some plain text content."


def test_extract_text_from_markdown(tmp_path):
    file_path = tmp_path / "note.md"
    file_path.write_text("# Title\n\nSome **markdown** body.", encoding="utf-8")

    assert extract_text(str(file_path)) == "# Title\n\nSome **markdown** body."


def test_extract_text_from_pdf(tmp_path):
    file_path = tmp_path / "doc.pdf"
    _make_pdf(file_path, text="Hello from a test PDF.")

    result = extract_text(str(file_path))

    assert "Hello from a test PDF." in result


def test_extract_text_missing_file_raises(tmp_path):
    missing = tmp_path / "does_not_exist.txt"

    with pytest.raises(FileNotFoundError):
        extract_text(str(missing))


def test_extract_text_unsupported_extension_raises(tmp_path):
    file_path = tmp_path / "data.docx"
    file_path.write_text("irrelevant", encoding="utf-8")

    with pytest.raises(UnsupportedFileTypeError):
        extract_text(str(file_path))


def test_extract_text_strips_utf8_bom(tmp_path):
    file_path = tmp_path / "note.txt"
    file_path.write_bytes("Hello world".encode("utf-8-sig"))

    assert extract_text(str(file_path)) == "Hello world"


def test_extract_text_non_utf8_file_raises_clear_error(tmp_path):
    file_path = tmp_path / "note.txt"
    file_path.write_bytes("café résumé".encode("latin-1"))

    with pytest.raises(TextDecodingError):
        extract_text(str(file_path))


def test_extract_text_corrupt_pdf_raises_clear_error(tmp_path):
    file_path = tmp_path / "broken.pdf"
    file_path.write_bytes(b"not a real pdf file content")

    with pytest.raises(PdfExtractionError):
        extract_text(str(file_path))


def test_extract_text_directory_raises_is_a_directory_error(tmp_path):
    directory = tmp_path / "some_folder.txt"
    directory.mkdir()

    with pytest.raises(IsADirectoryError):
        extract_text(str(directory))


def test_extract_text_missing_pdf_raises_file_not_found_not_pdf_error(tmp_path):
    # A missing .pdf is an access problem, not a corrupt PDF — it must
    # come through as FileNotFoundError, not get relabeled
    # PdfExtractionError by the generic `except Exception` branch.
    missing = tmp_path / "does_not_exist.pdf"

    with pytest.raises(FileNotFoundError):
        extract_text(str(missing))


def test_extract_text_missing_file_unsupported_extension_raises_file_not_found(
    tmp_path,
):
    # Existence is checked before reporting "unsupported type" for a
    # format we'd never have opened anyway.
    missing = tmp_path / "does_not_exist.docx"

    with pytest.raises(FileNotFoundError):
        extract_text(str(missing))
