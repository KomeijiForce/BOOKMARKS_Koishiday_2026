from __future__ import annotations

from .llm import LLMClient


def profile_extract(character: str, pairs: list[dict], llm: LLMClient) -> str:
    block = "\n---\n".join(pair["action"] for pair in pairs)
    prompt = f'''# Task

Please provide a 200-word, narrative-style character profile for {character}.
The profile should read like a cohesive introduction, weaving together the character’s background, physical description, personality traits and core motivations, notable attributes, relationships, key experiences, major plot involvement and key decisions or actions, character arc or development throughout the story, and other important details.
The profile should be written in a concise yet informative style, similar to what one might find in a comprehensive character guide, in language. Focus on the most crucial information that gives readers a clear understanding of the character’s significance in the work.
The profile should be based on either your existing knowledge of the character or the provided information, without fabricating or inferring any inaccurate or uncertain details.

# Scene-Action Pairs

{block}

Now, based on the given scene-action pairs, please generate the character profile, starting with ===Profile===.'''
    return llm.generate(prompt)


def profile_aggregate(main_profile: str, summarized_profile: str, llm: LLMClient) -> str:
    prompt = f'''# Main Profile
{main_profile}

# New Summarized Profile (From New Episodes)
{summarized_profile}

Directly update the main profile based on the new summarized profile, keep its length in around 200 words.'''
    return llm.generate(prompt)
