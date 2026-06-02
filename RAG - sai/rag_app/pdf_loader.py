from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

from PIL import Image
from pypdf import PdfReader

from rag_app.models import PageImage, PageText


def extract_pdf_pages(pdf_path: Path, doc_id: str, doc_title: str) -> list[PageText]:
    with pdf_path.open("rb") as handle:
        reader = PdfReader(handle)
        pages: list[PageText] = []
        for index, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            clean_text = " ".join(text.split())
            if clean_text:
                pages.append(
                    PageText(
                        doc_id=doc_id,
                        doc_title=doc_title,
                        page=index,
                        text=clean_text,
                    )
                )
        return pages


def extract_pdf_images(pdf_path: Path, doc_id: str, doc_title: str) -> list[PageImage]:
    with pdf_path.open("rb") as handle:
        reader = PdfReader(handle)
        images: list[PageImage] = []
        for page_number, page in enumerate(reader.pages, start=1):
            for image_index, image in enumerate(page.images):
                data = image.data
                if not data:
                    continue
                images.append(
                    PageImage(
                        doc_id=doc_id,
                        doc_title=doc_title,
                        page=page_number,
                        image_index=image_index,
                        data_url=_to_data_url(data, image.name),
                    )
                )
        return images


def _to_data_url(data: bytes, image_name: str | None) -> str:
    try:
        image = Image.open(BytesIO(data)).convert("RGB")
        image.thumbnail((768, 768), Image.Resampling.LANCZOS)
        output = BytesIO()
        image.save(output, format="PNG")
        encoded = base64.b64encode(output.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    except Exception:
        suffix = Path(image_name or "").suffix.lower().lstrip(".")
        mime_subtype = "jpeg" if suffix in {"jpg", "jpeg"} else suffix or "png"
        encoded = base64.b64encode(data).decode("ascii")
        return f"data:image/{mime_subtype};base64,{encoded}"
