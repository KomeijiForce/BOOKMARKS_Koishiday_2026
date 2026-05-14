from __future__ import annotations

import ast
import json
import re
from typing import Any


def extract_python_block(text: str) -> str:
    blocks = re.findall(r"```python\n(.*?)\n```", text, flags=re.DOTALL)
    if blocks:
        return blocks[0]
    blocks = re.findall(r"```(?:json)?\n(.*?)\n```", text, flags=re.DOTALL)
    if blocks:
        return blocks[0]
    return text


def parse_queries(response: str) -> list[dict[str, str]]:
    code = extract_python_block(response)
    local_vars: dict[str, Any] = {}
    try:
        exec(code, {}, local_vars)
        queries = local_vars.get("queries")
        if isinstance(queries, list):
            return [q for q in queries if isinstance(q, dict) and "query" in q and "tag" in q]
    except Exception:
        pass

    try:
        parsed = json.loads(code)
        if isinstance(parsed, dict):
            parsed = parsed.get("queries", [])
        if isinstance(parsed, list):
            return [q for q in parsed if isinstance(q, dict) and "query" in q and "tag" in q]
    except Exception:
        pass

    try:
        match = re.search(r"queries\s*=\s*(\[.*\])", code, flags=re.DOTALL)
        if match:
            parsed = ast.literal_eval(match.group(1))
            if isinstance(parsed, list):
                return [q for q in parsed if isinstance(q, dict) and "query" in q and "tag" in q]
    except Exception:
        pass

    raise ValueError("Could not parse bookmark queries from the LLM response.")


def parse_action(response: str) -> str:
    match = re.search(r'"action"\s*:\s*"(.*?)"\s*,?\s*\n?\}', response, flags=re.DOTALL)
    if match:
        return match.group(1).strip()

    code = extract_python_block(response)
    try:
        data = json.loads(code)
        if isinstance(data, dict) and "action" in data:
            return str(data["action"]).strip()
    except Exception:
        pass

    try:
        data = ast.literal_eval(code)
        if isinstance(data, dict) and "action" in data:
            return str(data["action"]).strip()
    except Exception:
        pass

    return response.strip()


def parse_score(response: str) -> str:
    match = re.search(r'"score"\s*:\s*"([ABC])"', response, flags=re.DOTALL)
    if match:
        return match.group(1)
    match = re.search(r"\b([ABC])\b", response)
    if match:
        return match.group(1)
    raise ValueError(f"Could not parse score from response: {response[:200]}")
