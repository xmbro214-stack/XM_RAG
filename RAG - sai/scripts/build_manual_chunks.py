from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image
from pypdf import PdfReader


DOC_NAME = "问界M6纯电版使用说明书"
MANUAL_VERSION = "UM2026V1.2"
PUBLISH_DATE = "2026-05"
DOC_ABBR = "m6ev"
SAFETY_LEVELS = {
    "危险": "danger",
    "警告": "warning",
    "注意": "caution",
    "提示": "tip",
}
PROCEDURE_TERMS = (
    "方法",
    "开启",
    "关闭",
    "调节",
    "使用",
    "连接",
    "激活",
    "查看",
    "设置",
    "退出",
    "删除",
    "初始化",
)
LIMITATION_TERMS = (
    "局限性",
    "前提条件",
    "使用条件",
    "退出条件",
    "不可用",
    "自动退出",
    "满足以下条件",
    "满足以下任一条件",
    "无法开启",
    "功能可能受影响",
)
TROUBLE_TERMS = (
    "故障",
    "异常",
    "指示灯点亮",
    "无法使用",
    "联系 AITO 用户中心",
    "停放至安全位置",
    "检修",
    "应急",
)
SPEC_TERMS = (
    "参数",
    "胎压",
    "加注量",
    "功率",
    "容量",
    "速度",
    "温度",
    "续航",
    "重量",
    "扭矩",
    "尺寸",
    "阈值",
)


@dataclass(frozen=True)
class TocEntry:
    chapter: str
    section: str
    page_start: int
    page_end: int

    @property
    def chapter_path(self) -> str:
        return self.chapter if self.chapter == self.section else f"{self.chapter} > {self.section}"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    pdf_path = next((root / "data" / "docs").glob("*.pdf"))
    index_dir = root / "data" / "index"
    image_dir = index_dir / "images"
    index_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    reader = PdfReader(str(pdf_path))
    page_texts = {
        page_number: reader.pages[page_number - 1].extract_text() or ""
        for page_number in range(1, len(reader.pages) + 1)
    }
    toc_entries = parse_toc(page_texts[2] + "\n" + page_texts[3])
    image_refs = export_page_images(reader, image_dir)

    chunks: list[dict] = []
    used_ids: set[str] = set()

    for entry in toc_entries:
        chunks.append(
            make_chunk(
                used_ids=used_ids,
                page_start=entry.page_start,
                page_end=entry.page_end,
                chapter_path=entry.chapter_path,
                content_type="toc",
                title=entry.section,
                raw_text=f"{entry.chapter_path}，起始页 {entry.page_start}，结束页 {entry.page_end}。",
                normalized_text=f"{entry.chapter_path} {entry.section} 页码 {entry.page_start}-{entry.page_end}",
                keywords=[entry.chapter, entry.section, "目录", "页码"],
                retrieval_hints=[f"{entry.section} 在第几页", f"{entry.section} 目录"],
                answer_template=f"{entry.section} 位于第 {entry.page_start} 页至第 {entry.page_end} 页。",
            )
        )

    parent_by_section: dict[str, str] = {}
    for entry in toc_entries:
        parent = make_chunk(
            used_ids=used_ids,
            page_start=entry.page_start,
            page_end=entry.page_end,
            chapter_path=entry.chapter_path,
            content_type="section_text",
            title=f"{entry.section}章节上下文",
            raw_text=f"{entry.chapter_path}章节，页码范围 {entry.page_start}-{entry.page_end}。",
            normalized_text=f"{entry.chapter_path} 章节上下文 页码范围 {entry.page_start}-{entry.page_end}",
            keywords=[entry.chapter, entry.section],
            retrieval_hints=[entry.chapter_path, entry.section],
            answer_template=f"相关章节：{entry.chapter_path}\n来源：第 {entry.page_start}-{entry.page_end} 页。",
        )
        chunks.append(parent)
        parent_by_section[entry.chapter_path] = parent["chunk_id"]

    for page_number, text in page_texts.items():
        if page_number in {2, 3}:
            continue
        entry = find_toc_entry(toc_entries, page_number)
        chapter_path = entry.chapter_path if entry else (first_heading(text) or "前言")
        parent_id = parent_by_section.get(chapter_path)
        page_images = image_refs.get(page_number, [])

        is_indicator_page = (
            "故障指示灯" in chapter_path
            or "仪表指示灯" in text
            or (
                "图标" in text
                and "说明" in text
                and any(term in text for term in ("指示灯", "此灯点亮", "此灯闪烁"))
            )
        )

        if is_indicator_page:
            chunks.extend(
                extract_indicator_chunks(
                    text,
                    page_number,
                    chapter_path,
                    parent_id,
                    page_images,
                    used_ids,
                )
            )

        chunks.extend(
            extract_acronym_definition_chunks(
                text=text,
                page_number=page_number,
                chapter_path=chapter_path,
                parent_id=parent_id,
                used_ids=used_ids,
            )
        )

        chunks.extend(
            extract_safety_chunks(
                text,
                page_number,
                chapter_path,
                parent_id,
                page_images,
                used_ids,
            )
        )

        chunks.extend(
            extract_table_like_chunks(
                text,
                page_number,
                chapter_path,
                parent_id,
                page_images,
                used_ids,
            )
        )

        if not is_indicator_page:
            chunks.extend(
                extract_general_chunks(
                    text,
                    page_number,
                    chapter_path,
                    parent_id,
                    page_images,
                    used_ids,
                )
            )

    output_path = index_dir / "chunks.jsonl"
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    stale_vectors = index_dir / "vectors.npy"
    if stale_vectors.exists():
        stale_vectors.unlink()


def parse_toc(toc_text: str) -> list[TocEntry]:
    chapters: set[str] = {
        "重要提示",
        "用车建议",
        "车辆概览",
        "驾驶安全",
        "车辆控制",
        "驾驶车辆",
        "出行与娱乐",
        "智慧用车",
        "辅助驾驶",
        "辅助泊车",
        "主动安全辅助",
        "充电和供电",
        "保养维护",
        "故障救援",
        "车辆规格",
        "隐私保护",
        "免责声明",
    }
    entries: list[tuple[str, str, int]] = []
    current_chapter = ""
    for raw_line in toc_text.splitlines():
        line = raw_line.strip()
        if not line or line == "目录" or re.fullmatch(r"\d+\s*目录", line):
            continue
        if line in chapters:
            current_chapter = line
            continue
        match = re.match(r"^(.+?)\.{2,}\s*(\d+)$", line)
        if not match or not current_chapter:
            continue
        title = clean_text(match.group(1))
        page = int(match.group(2))
        entries.append((current_chapter, title, page))

    result: list[TocEntry] = []
    for index, (chapter, section, page_start) in enumerate(entries):
        next_start = entries[index + 1][2] if index + 1 < len(entries) else 388 + 1
        result.append(
            TocEntry(
                chapter=chapter,
                section=section,
                page_start=page_start,
                page_end=max(page_start, next_start - 1),
            )
        )
    return result


def export_page_images(reader: PdfReader, image_dir: Path) -> dict[int, list[str]]:
    refs: dict[int, list[str]] = {}
    for page_number, page in enumerate(reader.pages, start=1):
        page_refs: list[str] = []
        for image_index, image in enumerate(page.images):
            data = image.data
            if not data:
                continue
            output = image_dir / f"p{page_number}_i{image_index + 1}.png"
            if output.exists():
                page_refs.append(str(output.relative_to(image_dir.parents[2])).replace("\\", "/"))
                continue
            try:
                with Image.open_frombytes(data):  # type: ignore[attr-defined]
                    pass
            except Exception:
                pass
            try:
                from io import BytesIO

                opened = Image.open(BytesIO(data)).convert("RGBA")
                opened.save(output)
                page_refs.append(str(output.relative_to(image_dir.parents[2])).replace("\\", "/"))
            except Exception:
                suffix = Path(image.name or "").suffix or ".bin"
                raw_output = output.with_suffix(suffix)
                raw_output.write_bytes(data)
                page_refs.append(str(raw_output.relative_to(image_dir.parents[2])).replace("\\", "/"))
        if page_refs:
            refs[page_number] = page_refs
    return refs


def find_toc_entry(entries: list[TocEntry], page_number: int) -> TocEntry | None:
    candidates = [
        entry for entry in entries if entry.page_start <= page_number <= entry.page_end
    ]
    return candidates[-1] if candidates else None


def extract_indicator_chunks(
    text: str,
    page_number: int,
    chapter_path: str,
    parent_id: str | None,
    page_images: list[str],
    used_ids: set[str],
) -> list[dict]:
    normalized = normalize_text(text)
    body = re.sub(r"^.*?图标\s+说明", "", normalized)
    body = remove_page_footer(body, page_number)
    matches = list(re.finditer(r"([^：。；]{2,40}(?:指示灯|警告灯))：", body))
    chunks: list[dict] = []
    image_cursor = 0

    if matches:
        for index, match in enumerate(matches):
            start = match.start()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
            raw = body[start:end].strip(" 。")
            title = clean_text(match.group(1))
            parts = split_indicator_parts(raw)
            for part_index, part in enumerate(parts):
                part_title = title if part_index == 0 else infer_indicator_title(part)
                chunks.append(
                    make_indicator_chunk(
                        used_ids,
                        part,
                        part_title,
                        page_number,
                        chapter_path,
                        parent_id,
                        page_images[image_cursor : image_cursor + 1],
                    )
                )
                image_cursor += 1

    if not chunks:
        sentences = [
            sentence.strip(" 。")
            for sentence in re.split(r"。", body)
            if sentence.strip().startswith("此灯")
        ]
        for index, sentence in enumerate(sentences):
            title = infer_indicator_title(sentence)
            chunks.append(
                make_indicator_chunk(
                    used_ids,
                    sentence,
                    title,
                    page_number,
                    chapter_path,
                    parent_id,
                    page_images[index : index + 1],
                )
            )

    return chunks


def split_indicator_parts(raw: str) -> list[str]:
    parts = [part.strip(" 。") for part in re.split(r"(?<=。)\s+(?=此灯)", raw) if part.strip(" 。")]
    return parts or [raw]


def make_indicator_chunk(
    used_ids: set[str],
    raw: str,
    title: str,
    page_number: int,
    chapter_path: str,
    parent_id: str | None,
    images: list[str],
) -> dict:
    status = extract_first(raw, r"(此灯[^，。；]*[，。；])")
    meaning = extract_first(raw, r"表示([^。；]+)")
    action = extract_action(raw)
    linked_sections = extract_linked_sections(raw)
    aliases = aliases_for(title)
    keywords = unique_words([title, "仪表指示灯", "故障灯", *aliases, *linked_sections])
    related_system = infer_related_system(title)
    is_fault = any(term in raw for term in ("故障", "风险", "联系 AITO", "不能继续行驶"))
    answer_template = (
        f"识别结果：{title}\n"
        f"含义：{meaning or raw}\n"
        f"处理建议：{action or '请按手册说明处理。'}\n"
        f"注意事项：{extract_warning_or_tip(raw)}\n"
        f"来源：第 {page_number} 页。"
    )
    return make_chunk(
        used_ids=used_ids,
        page_start=page_number,
        page_end=page_number,
        chapter_path=chapter_path,
        content_type="indicator_icon",
        title=title,
        raw_text=raw,
        normalized_text=f"{chapter_path} {title} {raw} {' '.join(aliases)}",
        keywords=keywords,
        aliases=aliases,
        entities=[title, related_system] if related_system else [title],
        related_function=title,
        related_system=related_system,
        image_refs=images,
        visual_description=(
            f"第 {page_number} 页图标表中与“{title}”对应的图标，具体形状、颜色和符号见 image_refs。"
            if images
            else ""
        ),
        linked_sections=linked_sections,
        retrieval_hints=[
            f"{title}是什么意思",
            f"{title}亮了怎么办",
            "灯亮了怎么办",
            "一直亮是什么意思",
            "能不能继续开" if is_fault else "这个图标表示什么",
            "是否需要联系 AITO" if is_fault else "仪表图标含义",
        ],
        answer_template=answer_template,
        parent_id=parent_id,
        safety_level="warning" if is_fault else "none",
    )


def extract_safety_chunks(
    text: str,
    page_number: int,
    chapter_path: str,
    parent_id: str | None,
    page_images: list[str],
    used_ids: set[str],
) -> list[dict]:
    lines = [line.strip() for line in text.splitlines()]
    chunks: list[dict] = []
    label_positions = [
        (index, line)
        for index, line in enumerate(lines)
        if line in SAFETY_LEVELS
    ]
    for pos, (line_index, label) in enumerate(label_positions):
        next_index = (
            label_positions[pos + 1][0]
            if pos + 1 < len(label_positions)
            else min(len(lines), line_index + 12)
        )
        block_lines = [
            line
            for line in lines[line_index + 1 : next_index]
            if line and not looks_like_footer(line, page_number)
        ]
        raw = clean_text(" ".join(block_lines))
        if not raw:
            continue
        related = infer_related_function(chapter_path, text[: text.find(label)])
        title = f"{related or last_path_part(chapter_path)}相关{label}"
        chunks.append(
            make_chunk(
                used_ids=used_ids,
                page_start=page_number,
                page_end=page_number,
                chapter_path=chapter_path,
                content_type="safety_notice",
                title=title,
                raw_text=raw,
                normalized_text=f"{chapter_path} {label} {related} {raw}",
                keywords=unique_words([label, related, "安全提示", "注意事项"]),
                aliases=[f"{related}{label}", f"{related}注意事项"] if related else [label],
                entities=[related] if related else [],
                safety_level=SAFETY_LEVELS[label],
                related_function=related,
                related_system=infer_related_system(related),
                image_refs=page_images[:1] if "图示" in raw else [],
                linked_sections=[chapter_path],
                retrieval_hints=[
                    f"{related}有什么注意事项" if related else "有什么安全提示",
                    "危险警告注意提示",
                    "能不能这样操作",
                ],
                answer_template=(
                    f"安全等级：{SAFETY_LEVELS[label]}\n"
                    f"相关功能：{related}\n"
                    f"提示内容：{raw}\n"
                    "建议：请严格按照手册要求操作。"
                ),
                parent_id=parent_id,
            )
        )
    return chunks


def extract_acronym_definition_chunks(
    *,
    text: str,
    page_number: int,
    chapter_path: str,
    parent_id: str | None,
    used_ids: set[str],
) -> list[dict]:
    normalized = normalize_text(text)
    matches = list(
        re.finditer(
            r"([\u4e00-\u9fffA-Za-z0-9（）()、·\s]{2,40})"
            r"（([A-Z]{2,8})）",
            normalized,
        )
    )
    chunks: list[dict] = []
    for match in matches:
        chinese_name = clean_text(match.group(1))
        abbr = match.group(2)
        if len(chinese_name) > 24 or chinese_name in {"功能介绍", "局限性"}:
            continue

        window_start = match.start()
        next_match_start = matches[matches.index(match) + 1].start() if matches.index(match) + 1 < len(matches) else len(normalized)
        window = normalized[window_start : min(len(normalized), max(next_match_start, window_start + 1200))]
        intro_match = re.search(
            rf"{re.escape(chinese_name)}（([A-Za-z][A-Za-z\s]+)，简称为\s*{abbr}）系统",
            window,
        )
        if not intro_match:
            continue

        english_name = clean_text(intro_match.group(1))
        raw = clean_text(window[:1200])
        title = f"{chinese_name}（{abbr}）"
        aliases = [
            abbr,
            chinese_name,
            english_name,
            f"{abbr} 是什么",
            f"{chinese_name}是什么",
        ]
        chunks.append(
            make_chunk(
                used_ids=used_ids,
                page_start=page_number,
                page_end=page_number,
                chapter_path=chapter_path,
                content_type="section_text",
                title=title,
                raw_text=raw,
                normalized_text=(
                    f"{chapter_path} {title} {chinese_name} {english_name} "
                    f"简称 {abbr} {raw} {' '.join(aliases)}"
                ),
                keywords=unique_words(
                    [
                        chinese_name,
                        abbr,
                        english_name,
                        "功能介绍",
                        "主动安全",
                        "碰撞风险",
                    ]
                ),
                aliases=aliases,
                entities=[chinese_name, abbr, english_name],
                related_function=chinese_name,
                related_system=infer_related_system(raw) or "主动安全辅助系统",
                linked_sections=extract_linked_sections(raw),
                retrieval_hints=[
                    f"{abbr} 是什么",
                    f"{abbr}是什么意思",
                    f"{chinese_name}是什么",
                    f"{abbr}全称",
                    f"{abbr}有什么用",
                ],
                answer_template=(
                    f"{abbr} 是{chinese_name}，英文全称为 {english_name}。\n"
                    f"功能说明：{raw}\n"
                    f"来源：第 {page_number} 页。"
                ),
                parent_id=parent_id,
            )
        )
    return chunks


def extract_table_like_chunks(
    text: str,
    page_number: int,
    chapter_path: str,
    parent_id: str | None,
    page_images: list[str],
    used_ids: set[str],
) -> list[dict]:
    normalized = normalize_text(text)
    chunks: list[dict] = []
    if "名称 标识 含义" in normalized:
        table_name = first_heading(text) or last_path_part(chapter_path)
        rows = re.findall(r"([^ ]+标识)\s+([^，。]+?)(?=(?:[^ ]+标识)|$)", normalized)
        for item, meaning in rows:
            raw = f"名称：{item}；含义：{clean_text(meaning)}"
            chunks.append(
                make_chunk(
                    used_ids=used_ids,
                    page_start=page_number,
                    page_end=page_number,
                    chapter_path=chapter_path,
                    content_type="table_row",
                    title=item,
                    raw_text=raw,
                    normalized_text=f"{chapter_path} {table_name} {raw}",
                    keywords=unique_words([table_name, item, "标识", "含义"]),
                    entities=[item],
                    safety_level="warning" if "危险" in raw or "触电" in raw else "none",
                    related_function=table_name,
                    related_system="高压系统" if "高压" in raw else "",
                    image_refs=page_images[:1],
                    visual_description=(
                        f"第 {page_number} 页“{table_name}”表格中与“{item}”对应的标识，具体图形见 image_refs。"
                    ),
                    retrieval_hints=[f"{item}是什么意思", "高压安全标识", "标识含义"],
                    answer_template=(
                        f"参数项：{item}\n数值/阈值：\n适用条件：\n说明：{clean_text(meaning)}\n来源：第 {page_number} 页。"
                    ),
                    parent_id=parent_id,
                )
            )
    return chunks


def extract_general_chunks(
    text: str,
    page_number: int,
    chapter_path: str,
    parent_id: str | None,
    page_images: list[str],
    used_ids: set[str],
) -> list[dict]:
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not looks_like_footer(line.strip(), page_number)
    ]
    if not lines:
        return []

    blocks: list[tuple[str, list[str]]] = []
    current_title = first_heading(text) or last_path_part(chapter_path) or f"第{page_number}页内容"
    current_lines: list[str] = []
    for line in lines:
        if line in SAFETY_LEVELS:
            continue
        if looks_like_heading(line):
            if current_lines:
                blocks.append((current_title, current_lines))
            current_title = line
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        blocks.append((current_title, current_lines))

    chunks: list[dict] = []
    for title, block_lines in blocks:
        raw = clean_text(" ".join(block_lines))
        if len(raw) < 18:
            continue
        content_type = classify_block(title, raw, chapter_path)
        for part_index, piece in enumerate(split_semantic_text(raw, content_type)):
            chunk_title = title if part_index == 0 else f"{title}（续 {part_index + 1}）"
            aliases = aliases_for(chunk_title, content_type)
            keywords = extract_keywords(chapter_path, chunk_title, piece, content_type)
            chunks.append(
                make_chunk(
                    used_ids=used_ids,
                    page_start=page_number,
                    page_end=page_number,
                    chapter_path=chapter_path,
                    content_type=content_type,
                    title=chunk_title,
                    raw_text=piece,
                    normalized_text=f"{chapter_path} {chunk_title} {piece} {' '.join(aliases)}",
                    keywords=keywords,
                    aliases=aliases,
                    entities=extract_entities(chunk_title, piece),
                    safety_level="none",
                    related_function=infer_related_function(chapter_path, chunk_title),
                    related_system=infer_related_system(chunk_title + piece),
                    image_refs=page_images[:1] if content_type == "diagram_callout" else [],
                    linked_sections=extract_linked_sections(piece),
                    retrieval_hints=retrieval_hints_for(content_type, chunk_title, piece),
                    answer_template=answer_template_for(content_type, chunk_title, piece, page_number),
                    parent_id=parent_id,
                )
            )
    return chunks


def classify_block(title: str, raw: str, chapter_path: str) -> str:
    joined = f"{title} {raw}"
    if "缩略语" in chapter_path or "单位术语" in chapter_path:
        return "glossary"
    if any(term in joined for term in LIMITATION_TERMS):
        return "limitation"
    if any(term in joined for term in TROUBLE_TERMS) and "故障救援" in chapter_path:
        return "troubleshooting"
    if any(term in title for term in PROCEDURE_TERMS):
        return "procedure"
    if any(term in joined for term in SPEC_TERMS) and re.search(r"\d", joined):
        return "specification"
    if re.search(r"(图\d+|[（(]?[1-9]\d?[）)]\s*[^。；]+)", joined) and "简介" in chapter_path:
        return "diagram_callout"
    return "section_text"


def split_semantic_text(raw: str, content_type: str) -> list[str]:
    if content_type != "section_text":
        return [raw]
    if len(raw) <= 900:
        return [raw]
    sentences = re.split(r"(?<=。)", raw)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) <= 850:
            current += sentence
            continue
        if current:
            chunks.append(current.strip())
        current = sentence
    if current:
        chunks.append(current.strip())
    return chunks


def make_chunk(
    *,
    used_ids: set[str],
    page_start: int,
    page_end: int,
    chapter_path: str,
    content_type: str,
    title: str,
    raw_text: str,
    normalized_text: str,
    keywords: list[str],
    aliases: list[str] | None = None,
    entities: list[str] | None = None,
    safety_level: str = "none",
    related_function: str = "",
    related_system: str = "",
    source_refs: list[dict] | None = None,
    image_refs: list[str] | None = None,
    visual_description: str = "",
    linked_sections: list[str] | None = None,
    retrieval_hints: list[str] | None = None,
    answer_template: str = "",
    parent_id: str | None = None,
) -> dict:
    chunk_id = stable_chunk_id(page_start, content_type, title, used_ids)
    page_start = int(page_start)
    page_end = int(page_end)
    raw_text = clean_text(raw_text)
    normalized_text = clean_text(normalized_text)
    return {
        "chunk_id": chunk_id,
        "parent_id": parent_id,
        "doc_name": DOC_NAME,
        "manual_version": MANUAL_VERSION,
        "publish_date": PUBLISH_DATE,
        "page_start": page_start,
        "page_end": page_end,
        "chapter_path": chapter_path,
        "content_type": content_type,
        "title": title or last_path_part(chapter_path) or f"第{page_start}页内容",
        "raw_text": raw_text,
        "normalized_text": normalized_text or raw_text,
        "keywords": unique_words(keywords),
        "aliases": unique_words(aliases or []),
        "entities": unique_words(entities or []),
        "safety_level": safety_level,
        "related_function": related_function,
        "related_system": related_system,
        "source_refs": source_refs
        or [{"pdf": f"data/docs/{DOC_NAME}.pdf", "page": page_start}],
        "image_refs": image_refs or [],
        "visual_description": visual_description,
        "linked_sections": unique_words(linked_sections or []),
        "retrieval_hints": unique_words(retrieval_hints or []),
        "answer_template": answer_template,
    }


def stable_chunk_id(
    page_start: int, content_type: str, title: str, used_ids: set[str]
) -> str:
    digest = hashlib.sha1(f"{page_start}:{content_type}:{title}".encode("utf-8")).hexdigest()[:8]
    base = f"{DOC_ABBR}_p{page_start}_{content_type}_{digest}"
    candidate = base
    counter = 2
    while candidate in used_ids:
        candidate = f"{base}_{counter}"
        counter += 1
    used_ids.add(candidate)
    return candidate


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def normalize_text(value: str) -> str:
    return clean_text(value.replace("\n", " "))


def remove_page_footer(value: str, page_number: int) -> str:
    value = re.sub(rf"\b{page_number}\s+\S+\s*$", "", value)
    value = re.sub(rf"\S+\s+{page_number}\s*$", "", value)
    return value.strip()


def looks_like_footer(line: str, page_number: int) -> bool:
    return bool(
        re.fullmatch(rf"{page_number}\s+\S+", line)
        or re.fullmatch(rf"\S+\s+{page_number}", line)
    )


def looks_like_heading(line: str) -> bool:
    if len(line) > 28:
        return False
    if line.startswith(("●", "•", "-", "（", "(")):
        return False
    if re.search(r"[。；：:，,]", line):
        return False
    return bool(re.search(r"[\u4e00-\u9fffA-Za-z]", line))


def first_heading(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line and looks_like_heading(line) and not re.fullmatch(r"\d+.*", line):
            return line
    return ""


def last_path_part(chapter_path: str) -> str:
    return chapter_path.split(">")[-1].strip() if chapter_path else ""


def extract_first(value: str, pattern: str) -> str:
    match = re.search(pattern, value)
    return clean_text(match.group(1)) if match else ""


def extract_action(raw: str) -> str:
    sentences = re.split(r"(?<=。)|(?<=；)", raw)
    action_sentences = [
        clean_text(sentence)
        for sentence in sentences
        if any(term in sentence for term in ("请", "勿", "联系 AITO", "停车", "充电", "谨慎驾驶"))
    ]
    return " ".join(action_sentences)


def extract_warning_or_tip(raw: str) -> str:
    if any(term in raw for term in ("风险", "请勿", "为防止", "不能继续行驶")):
        return raw
    return ""


def extract_linked_sections(raw: str) -> list[str]:
    refs: list[str] = []
    for match in re.finditer(r"请参阅([^（(。]+)[（(]\s*(\d+)\s*页[）)]", raw):
        refs.append(f"{clean_text(match.group(1))}（第 {match.group(2)} 页）")
    return refs


def infer_indicator_title(sentence: str) -> str:
    match = re.search(r"表示([^。；，]+)", sentence)
    if match:
        name = clean_text(match.group(1))
        name = re.sub(r"^(驾驶模式为|路面辅助模式为)", "", name)
        name = re.sub(r"(已开启|正在工作|工作中)$", "", name)
        return f"{name}指示灯"
    return "未命名指示灯"


def infer_related_function(chapter_path: str, context: str) -> str:
    candidates = [
        line.strip()
        for line in context.splitlines()
        if line.strip() and looks_like_heading(line.strip())
    ]
    return candidates[-1] if candidates else last_path_part(chapter_path)


def infer_related_system(text: str) -> str:
    mapping = {
        "安全气囊": "安全气囊系统",
        "胎压": "胎压监测系统",
        "制动": "制动系统",
        "防抱死": "防抱死制动系统",
        "动力电池": "动力电池系统",
        "驱动电机": "驱动系统",
        "低压": "低压供电系统",
        "电子驻车": "电子驻车系统",
        "空气悬架": "空气悬架系统",
        "车身稳定": "车身稳定性系统",
        "转向": "转向辅助系统",
        "ADS": "ADS 辅助驾驶系统",
        "ACC": "自适应巡航辅助系统",
        "LCC": "车道巡航辅助系统",
        "NCA": "领航辅助系统",
        "车外灯": "车外灯光系统",
        "高压": "高压系统",
    }
    for key, value in mapping.items():
        if key in text:
            return value
    return ""


def aliases_for(title: str, content_type: str = "") -> list[str]:
    aliases: list[str] = []
    if "安全气囊" in title:
        aliases += ["安全气囊灯", "气囊灯", "气囊故障灯", "SRS 灯", "airbag warning", "红色人形气囊图标"]
    if "胎压" in title:
        aliases += ["胎压灯", "胎压报警", "轮胎报警灯", "胎压故障灯", "轮胎压力异常", "TPMS"]
    if "电子驻车" in title:
        aliases += ["电子手刹灯", "驻车灯", "P 灯", "EPB 指示灯"]
    if "动力电池" in title:
        aliases += ["电池故障灯", "电池报警", "动力电池报警"]
    if "驱动电机" in title:
        aliases += ["电机故障灯", "驱动故障", "动力故障灯"]
    if content_type == "procedure" or any(term in title for term in PROCEDURE_TERMS):
        aliases += ["怎么开启", "怎么关闭", "怎么设置", "在哪里设置", "如何使用", "入口在哪里"]
    if content_type == "troubleshooting" or "故障" in title:
        aliases += ["灯亮了怎么办", "一直亮是什么意思", "闪烁是什么意思", "能不能继续开", "要不要停车", "是否需要联系 AITO"]
    return aliases


def unique_words(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        value = clean_text(str(value))
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def extract_keywords(
    chapter_path: str, title: str, text: str, content_type: str
) -> list[str]:
    candidates = [*chapter_path.replace(">", " ").split(), title, content_type]
    candidates += re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,12}(?:系统|功能|模式|指示灯|按键|开关|报警|故障)?", text)
    return unique_words(candidates[:20])


def extract_entities(title: str, text: str) -> list[str]:
    candidates = [title]
    candidates += re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,16}(?:系统|功能|模式|指示灯|按键|踏板|安全带|气囊|电池|电机)", text)
    return unique_words(candidates[:10])


def retrieval_hints_for(content_type: str, title: str, text: str) -> list[str]:
    if content_type == "procedure":
        return [f"{title}怎么操作", f"{title}在哪里设置", "操作步骤", "开启关闭方法"]
    if content_type == "limitation":
        return [f"{title}不能用", f"{title}退出条件", "使用条件", "局限性"]
    if content_type == "troubleshooting":
        return [f"{title}怎么办", "故障处理", "是否需要联系 AITO", "要不要停车"]
    if content_type == "specification":
        return [f"{title}参数", f"{title}是多少", "规格参数", "阈值"]
    return [title, "是什么", "怎么用"]


def answer_template_for(content_type: str, title: str, text: str, page: int) -> str:
    if content_type == "procedure":
        return f"功能：{title}\n操作步骤：\n{text}\n前提条件：见原文。\n注意事项：见原文。\n来源：第 {page} 页。"
    if content_type == "specification":
        return f"参数项：{title}\n数值/阈值：见原文。\n适用条件：见原文。\n说明：{text}\n来源：第 {page} 页。"
    if content_type == "troubleshooting":
        return f"故障现象：{title}\n含义：{text}\n风险：见原文。\n处理建议：{extract_action(text) or '请按手册说明处理。'}\n来源：第 {page} 页。"
    return f"{title}\n{text}\n来源：第 {page} 页。"


if __name__ == "__main__":
    main()
