from __future__ import annotations

from .llm import LLMClient
from .parsing import parse_score


def score_action_nli(character: str, scene: str, prediction: str, action: str, llm: LLMClient) -> int:
    prompt = f'''# Scene
{scene}

Your Response: {prediction}
Ground Truth: {action}

Score the action of {character} in the response based on the ground truth.
A: The ground truth entails the action of {character} in the response. (Following the same character logic.)
B: The ground truth is neutral to the action of {character} in the response. (Reflecting a different facet.)
C: The ground truth contradicts the action of {character} in the response. (Following a contradicted character logic.)

Output in json: 
```json
{{
"reasoning": "...",
"score": "A/B/C"
}}
```'''
    score = parse_score(llm.generate(prompt, llm.eval_model))
    return {"A": 100, "B": 50, "C": 0}[score]


def score_action_em(character: str, scene: str, prediction: str, action: str, llm: LLMClient) -> int:
    prompt = f'''# Scene
{scene}

Your Response: {prediction}
Ground Truth: {action}

Decide whether the main action taken by {character} in "Your Response" matches the main action in "Ground Truth".
Focus on the primary verb-level action (what {character} does), and ignore minor differences in wording, intensity, tone, or small extra details (e.g., sub-actions) that do not change the core action.
If multiple actions appear, judge using the most central action in each text.

A: Exact match — the main action in the response is the same as the main action in the ground truth (allowing paraphrase).
B: Partial/adjacent match — the response is close but not the same main action, and the differences are not minor.
C: Mismatch — the main action is different or incompatible.

Output in json:
```json
{{
  "reasoning": "...",
  "score": "A/B/C"
}}
```'''
    score = parse_score(llm.generate(prompt, llm.eval_model))
    return {"A": 100, "B": 0, "C": 0}[score]


def benchmark_precision(character: str, scene: str, prediction: str, action: str, llm: LLMClient, metrics: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    if "nli" in metrics:
        result["nli"] = score_action_nli(character, scene, prediction, action, llm)
    if "em" in metrics:
        result["em"] = score_action_em(character, scene, prediction, action, llm)
    return result
