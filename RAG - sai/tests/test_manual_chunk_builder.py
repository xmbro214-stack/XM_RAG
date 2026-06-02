import importlib.util
import sys
from pathlib import Path


def load_builder_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "build_manual_chunks.py"
    spec = importlib.util.spec_from_file_location("build_manual_chunks", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_extract_acronym_definition_keeps_locp_semantics_together():
    builder = load_builder_module()
    used_ids = set()
    text = """
侧向障碍物防碰撞（LOCP）
在车辆与侧向成排水马、栅栏、小幅占用自车
车道的车辆等障碍物存在碰撞风险，或存在掉
入与路面有高度差区域（如排水渠、田坎等）
的风险时，紧急辅助驾驶员控制车辆，以规避
和减轻碰撞风险。
功能介绍
侧向障碍物防碰撞（Lateral Obstacle
Collision Prevention，简称为 LOCP）系统
利用摄像头等传感器识别周边行驶环境，当车
辆在非急弯路段上以约 30 km/h ~ 130 km/h
的车速行驶，LOCP 会在存在以下侧向碰撞风
险时紧急辅助驾驶员短暂地转动方向盘。
"""

    chunks = builder.extract_acronym_definition_chunks(
        text=text,
        page_number=302,
        chapter_path="主动安全辅助 > 侧向安全",
        parent_id="parent",
        used_ids=used_ids,
    )

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk["title"] == "侧向障碍物防碰撞（LOCP）"
    assert chunk["content_type"] == "section_text"
    assert "Lateral Obstacle Collision Prevention" in chunk["raw_text"]
    assert "侧向成排水马、栅栏" in chunk["raw_text"]
    assert "30 km/h ~ 130 km/h" in chunk["raw_text"]
    assert "低速碰撞预防" not in chunk["raw_text"]
    assert "LOCP 是什么" in chunk["retrieval_hints"]
