from pathlib import Path
import re


def test_assistant_cards_do_not_shrink_inside_scrollable_chat():
    css = Path("web/styles.css").read_text(encoding="utf-8")
    match = re.search(r"\.assistant-card\s*\{(?P<body>[^}]*)\}", css, re.S)

    assert match is not None
    assert re.search(r"flex-shrink\s*:\s*0\s*;", match.group("body"))


def test_assistant_cards_do_not_clip_answer_content():
    css = Path("web/styles.css").read_text(encoding="utf-8")
    match = re.search(r"\.assistant-card\s*\{(?P<body>[^}]*)\}", css, re.S)

    assert match is not None
    body = match.group("body")
    assert not re.search(r"overflow\s*:\s*hidden\s*;", body)
    assert re.search(r"height\s*:\s*auto\s*;", body)
    assert re.search(r"min-height\s*:", body)


def test_static_assets_are_cache_busted_for_layout_updates():
    html = Path("web/index.html").read_text(encoding="utf-8")

    assert 'href="/static/styles.css?v=' in html
    assert 'src="/static/app.js?v=' in html


def test_welcome_vehicle_logo_uses_photo_asset():
    html = Path("web/index.html").read_text(encoding="utf-8")
    script = Path("web/app.js").read_text(encoding="utf-8")
    css = Path("web/styles.css").read_text(encoding="utf-8")
    concept_match = re.search(r"\.m6-concept\s*\{(?P<body>[^}]*)\}", css, re.S)

    assert Path("web/assets/wenjie-m6-photo.svg").exists()
    assert 'class="m6-photo-card"' in html
    assert 'class="m6-photo"' in html
    assert "/static/assets/wenjie-m6-photo.svg" in html
    assert "m6-photo-card" in script
    assert "m6-photo" in script
    assert "/static/assets/wenjie-m6-photo.svg" in script
    assert concept_match is not None
    assert re.search(r"width\s*:\s*min\(306px,\s*40vw\)\s*;", concept_match.group("body"))
    assert re.search(r"height\s*:\s*162px\s*;", concept_match.group("body"))
    assert ".m6-photo-card" in css
    assert ".m6-photo" in css
    assert re.search(r"object-fit\s*:\s*cover\s*;", css)
    assert re.search(r"border-radius\s*:\s*18px\s*;", css)
