from pathlib import Path
from io import BytesIO

from PIL import Image

from rag_app.pdf_loader import extract_pdf_pages
from rag_app.pdf_loader import _to_data_url


class FakePage:
    def __init__(self, text):
        self._text = text

    def extract_text(self):
        return self._text


class FakeReader:
    def __init__(self, _file):
        self.pages = [FakePage("第一页内容"), FakePage("第二页内容")]


def test_extract_pdf_pages_keeps_one_based_page_numbers(monkeypatch, tmp_path):
    pdf_path = tmp_path / "manual.pdf"
    pdf_path.write_bytes(b"%PDF fake")
    monkeypatch.setattr("rag_app.pdf_loader.PdfReader", FakeReader)

    pages = extract_pdf_pages(pdf_path, doc_id="manual", doc_title="说明书")

    assert [page.page for page in pages] == [1, 2]
    assert pages[0].text == "第一页内容"
    assert pages[1].doc_title == "说明书"


def test_to_data_url_converts_pdf_images_to_png_data_url():
    buffer = BytesIO()
    Image.new("RGB", (1, 1), "red").save(buffer, format="PNG")

    data_url = _to_data_url(buffer.getvalue(), "icon.jp2")

    assert data_url.startswith("data:image/png;base64,")
