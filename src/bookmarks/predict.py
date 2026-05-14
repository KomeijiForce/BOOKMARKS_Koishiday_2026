from __future__ import annotations

from .llm import LLMClient
from .parsing import parse_action


def next_action_prediction(character: str, scene: str, grounding: str, llm: LLMClient, utterance: bool = True) -> str:
    if utterance:
        output_hint = f"A concise sentence, {character}: ..."
    else:
        output_hint = "A concise sentence"

    prompt = f'''# Background Summary
{grounding}
---
# Current Scene
{scene}
---
Infer the next action ({output_hint}) of {character} in the following scene:

Output in the following format:
```python
{{
    "reasoning": "...",
    "action": "...",
}}
```'''
    response = llm.generate(prompt, llm.model)
    return parse_action(response)
