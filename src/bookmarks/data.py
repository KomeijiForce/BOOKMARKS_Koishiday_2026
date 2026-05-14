from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_action_item(item: dict[str, Any]) -> dict[str, Any]:
    item = dict(item)
    if "characters" not in item and "character" in item:
        item["characters"] = [item["character"]]
    if "characters" not in item:
        item["characters"] = []
    if isinstance(item["characters"], str):
        item["characters"] = [item["characters"]]
    return item


def flatten_title_actions(title2action_series: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    action_seq: list[dict[str, Any]] = []
    for title, action_series in title2action_series.items():
        for raw_item in action_series:
            item = normalize_action_item(raw_item)
            item.setdefault("title", title)
            action_seq.append(item)
    return action_seq


def load_artifact(data_dir: str, artifact: str) -> dict[str, Any]:
    root = Path(data_dir)
    all_characters = load_json(root / "all_characters.json")
    band2members_path = root / "band2members.json"
    band2members = load_json(band2members_path) if band2members_path.exists() else {}
    title2action_series = load_json(root / f"title2action_series.{artifact}.json")
    main_characters = all_characters[artifact]["major"]
    action_seq = flatten_title_actions(title2action_series)
    return {
        "artifact": artifact,
        "action_seq": action_seq,
        "original_action_seq": deepcopy(action_seq),
        "main_characters": main_characters,
        "utterance": artifact in band2members,
    }


def build_test_ids(action_seq: list[dict[str, Any]], main_characters: list[str], test_split: float) -> list[int]:
    test_ids: list[int] = []
    for character in main_characters:
        character_ids = [idx for idx, item in enumerate(action_seq) if character in item.get("characters", [])]
        start = int(len(character_ids) * test_split)
        test_ids.extend(character_ids[start:])
    return sorted(set(test_ids))


def scene_before(action_seq: list[dict[str, Any]], idx: int, window: int) -> str:
    return "\n".join(item["action"] for item in action_seq[max(0, idx - window):idx])
