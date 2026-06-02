from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps
from pypdf import PdfReader

from rag_app.models import SourceCitation


FEATURE_SIZE = 64
MATCH_THRESHOLD = 0.04


@dataclass(frozen=True)
class IconCatalogEntry:
    page: int
    image_index: int
    label: str
    snippet: str


PAGE_20_ICON_CATALOG = [
    IconCatalogEntry(
        page=20,
        image_index=1,
        label="主驾安全带未系指示灯",
        snippet="这是主驾安全带未系指示灯。此灯点亮表示驾驶员安全带未系，用于提醒驾驶员及时系好安全带。",
    ),
    IconCatalogEntry(
        page=20,
        image_index=2,
        label="副驾安全带未系指示灯",
        snippet="这是副驾安全带未系指示灯。此灯点亮表示前排乘客安全带未系，用于提醒前排乘客及时系好安全带。",
    ),
    IconCatalogEntry(
        page=20,
        image_index=3,
        label="后排安全带未系指示灯",
        snippet="这是后排安全带未系指示灯。此灯点亮表示后排有乘客安全带未系，L 表示后排左，M 表示后排中，R 表示后排右。",
    ),
    IconCatalogEntry(
        page=20,
        image_index=4,
        label="安全气囊指示灯",
        snippet="这是安全气囊指示灯。此灯点亮表示安全气囊存在故障，请将车辆停放至安全位置并联系 AITO 用户中心。",
    ),
    IconCatalogEntry(
        page=20,
        image_index=5,
        label="电子驻车指示灯",
        snippet="这是电子驻车指示灯。此灯闪烁表示电子驻车功能正在工作中；此灯点亮表示电子驻车完成工作。",
    ),
    IconCatalogEntry(
        page=20,
        image_index=6,
        label="驱动电机故障指示灯",
        snippet="这是驱动电机故障指示灯。此灯点亮表示车辆驱动电机出现故障，存在车辆不能继续行驶的风险，请将车辆停放至安全位置并联系 AITO 用户中心。",
    ),
    IconCatalogEntry(
        page=20,
        image_index=7,
        label="低压供电系统故障指示灯",
        snippet="这是低压供电系统故障指示灯。此灯点亮表示低压蓄电池充电系统存在故障，请先尝试启动车辆给低压蓄电池充电；若仍亮起，请将车辆停放至安全位置并联系 AITO 用户中心。",
    ),
    IconCatalogEntry(
        page=20,
        image_index=8,
        label="动力电池故障指示灯",
        snippet="这是动力电池故障指示灯。此灯点亮表示车辆动力电池出现故障，请将车辆停放至安全位置并联系 AITO 用户中心。",
    ),
    IconCatalogEntry(
        page=20,
        image_index=9,
        label="动力系统故障指示灯",
        snippet="这是动力系统故障指示灯。此灯点亮表示车辆动力系统出现故障，请将车辆停放至安全位置并联系 AITO 用户中心。",
    ),
    IconCatalogEntry(
        page=20,
        image_index=10,
        label="充电枪连接指示灯",
        snippet="这是充电枪连接指示灯。此灯点亮表示充电枪已连接。",
    ),
    IconCatalogEntry(
        page=20,
        image_index=11,
        label="制动系统故障指示灯",
        snippet="这是制动系统故障指示灯。此灯点亮表示制动系统存在故障或制动液位低，存在制动失灵风险；请将车辆停放至安全位置并联系 AITO 用户中心。",
    ),
    IconCatalogEntry(
        page=20,
        image_index=12,
        label="ADS 功能故障指示灯",
        snippet="这是 ADS 功能故障指示灯。此灯闪烁表示 ADS 功能故障，请谨慎驾驶并尽快联系 AITO 用户中心。",
    ),
    IconCatalogEntry(
        page=20,
        image_index=13,
        label="介入请求指示灯",
        snippet="这是介入请求指示灯。此灯点亮表示遇到急弯、碰撞风险等系统可能无法处理的复杂场景，驾驶员应立即控制车辆速度和方向，确保安全驾驶。",
    ),
    IconCatalogEntry(
        page=20,
        image_index=14,
        label="转向辅助系统故障指示灯",
        snippet="这是转向辅助系统故障指示灯。此灯点亮表示转向辅助系统出现故障，请将车辆停放至安全位置并联系 AITO 用户中心。",
    ),
    IconCatalogEntry(
        page=20,
        image_index=15,
        label="空气悬架维修模式/超载指示灯",
        snippet="这是空气悬架维修模式/超载指示灯。此灯点亮表示车辆超载或空气悬架处于维修模式，已影响空气悬架调节功能。",
    ),
]


def match_manual_icon(images: list[str], docs_dir: Path) -> SourceCitation | None:
    if not images:
        return None

    query_feature = _feature_from_data_url(images[0])
    if query_feature is None:
        return None

    pdfs = sorted(docs_dir.glob("*.pdf"))
    if not pdfs:
        return None

    pdf_path = pdfs[0]
    reader = PdfReader(str(pdf_path))
    best: tuple[float, IconCatalogEntry] | None = None

    for entry in PAGE_20_ICON_CATALOG:
        try:
            image = reader.pages[entry.page - 1].images[entry.image_index]
            feature = _feature_from_bytes(image.data)
        except Exception:
            continue
        if feature is None:
            continue
        distance = _mean_squared_error(query_feature, feature)
        if best is None or distance < best[0]:
            best = (distance, entry)

    if best is None or best[0] > MATCH_THRESHOLD:
        return None

    distance, entry = best
    return SourceCitation(
        doc_title=pdf_path.stem,
        page=entry.page,
        snippet=entry.snippet,
        score=round(max(0.0, 1.0 - distance / MATCH_THRESHOLD), 4),
    )


def _feature_from_data_url(data_url: str) -> list[int] | None:
    if "," not in data_url:
        return None
    _, encoded = data_url.split(",", 1)
    try:
        return _feature_from_bytes(base64.b64decode(encoded))
    except Exception:
        return None


def _feature_from_bytes(data: bytes) -> list[int] | None:
    image = Image.open(BytesIO(data)).convert("RGB")
    box = _red_foreground_box(image)
    if box is None:
        return None

    icon = image.crop(box).convert("L")
    icon = ImageOps.invert(icon)
    icon.thumbnail((FEATURE_SIZE - 8, FEATURE_SIZE - 8), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (FEATURE_SIZE, FEATURE_SIZE), 0)
    canvas.paste(
        icon,
        ((FEATURE_SIZE - icon.width) // 2, (FEATURE_SIZE - icon.height) // 2),
    )
    return list(canvas.getdata())


def _red_foreground_box(image: Image.Image) -> tuple[int, int, int, int] | None:
    pixels = image.load()
    xs: list[int] = []
    ys: list[int] = []
    for y in range(image.height):
        for x in range(image.width):
            red, green, blue = pixels[x, y]
            if red > 80 and red > green * 1.1 and red > blue * 1.1:
                xs.append(x)
                ys.append(y)
    if not xs:
        return None

    padding = 4
    return (
        max(min(xs) - padding, 0),
        max(min(ys) - padding, 0),
        min(max(xs) + padding + 1, image.width),
        min(max(ys) + padding + 1, image.height),
    )


def _mean_squared_error(left: list[int], right: list[int]) -> float:
    return sum((a - b) ** 2 for a, b in zip(left, right)) / len(left) / 255 / 255
