from __future__ import annotations

import re
from typing import Optional

from .classifier import BehaviorDiscriminator
from .llm import LLMClient
from .parsing import parse_queries


STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "being", "been",
    "do", "does", "did", "done", "to", "of", "in", "on", "at", "by", "for",
    "with", "about", "from", "into", "through", "during", "before", "after",
    "above", "below", "up", "down", "out", "off", "over", "under",
    "and", "or", "but", "if", "then", "else", "when", "while", "as",
    "what", "which", "who", "whom", "whose", "where", "why", "how",
    "this", "that", "these", "those",
    "it", "its", "he", "she", "they", "them", "their", "there",
    "now", "current", "currently", "right",
    "usually", "generally", "tend", "tends",
}


def init_bookmarks(character: str, scene: str, profile: str, history: str, llm: LLMClient, utterance: bool = True, k: int = 3) -> list[dict]:
    predict_target = "utterance" if utterance else "action"
    prompt = f'''# Scene
{scene}

# Profile of {character}
{profile}

Now, it's {character}'s turn to make the next {predict_target}.
To precisely ground {character}'s {predict_target},
we need more information about the state of the current story.
The system supports three kinds of search queries:

1. concept search:
   - Used to look up the meaning or details of a concrete concept
     mentioned earlier in the story.
   - Must query a named concept or entity. (e.g., What is "..." ...?)
   - Example: What is "Star Beat" as mentioned in this conversation?

2. state search:
   - Used to iteratively update and recover the *current* state of the world
     up to this point in the story.
   - Never use chronical words like **before/after the scene** or **recent** in the query 
     because the search is done recursively, always ask about **now**.
   - Example: Where are the band members right now?

3. behavioral search:
   - Used to recall or infer how a {character} tends to behave
     in a particular situation or under certain conditions, based on the story so far.
   - Behavioral queries should be phrased in a **general** way so that many past
     scenes can potentially match the condition, rather than referring to one
     extremely specific single moment. (simply speaking, make it fewer than 15 words.)
   - Example: How does Arisa usually react when someone makes an impulsive decision?

Search history:
{history}

Your task:
Propose {k} most crucial search queries as questions to check
before deciding {character}'s next {predict_target}.
You may freely use any of the three query types above, as long as
the query is useful for grounding {character}'s next {predict_target}.
**However, you should make sure each query is meaningful, targeting at
information not given in the scene.**

Important (for each query):
- Keep the question simple and concise, "unknown" in the history
  means a failed trial, don't try that direction again and
  **avoid even relevant queries** unless no other important information
  to gather, the storyline might just does not mention it.
- Don't focus only on the content, after the content to express
  is clear, search for how the character generally express it,
  especially given that the personality of the character will
  affect the content they will explicitly express.
- For the question, make sure there is no coreference involved:
  the question must explicitly mention all subjects and objects
  instead of using pronouns like "she", "they", "there", or "that".
- The question should be about the story world, characters, and events,
  not about the model or the querying system itself.
- When query tagged as "behavioral", keep the condition general enough that
  it could match multiple past scenes, not a single unique instance.
- For the query, you MUST assign a tag field with one of:
  "concept", "state", or "behavioral",
  indicating which search type this query is intended for.
  only tag as "concept" when searching a named concept or entity.
- Keep each query concise and high potential for information gathering.

Output as Python code in the following form:

```python
queries = [
    {{"query": "...", "tag": "..."}},
    {{"query": "...", "tag": "..."}},
    # ...
]
```'''
    queries = parse_queries(llm.generate(prompt))[:k]
    bookmarks = []
    for query in queries:
        bookmark = {**query, "character": character, "answer": "unknown", "index": 0}
        if bookmark["tag"] == "behavioral":
            bookmark["action list"] = []
        bookmarks.append(bookmark)
    return bookmarks


def llm_check(chunk: str, query: str, answer: str, llm: LLMClient) -> str:
    prompt = f'''Before the following scene,
Current answer to "{query}": "{answer}"

After the following scene:
{chunk}

Considering this scene, how should we treat the current answer to "{query}"?

Please answer with exactly one of:
- "reset"  -> the current answer is now outdated or contradicted and should be set to "unknown"
- "update" -> the scene triggers state transition or provides more accurate information so the answer should be updated
- "none"   -> the scene does not contain relevant update or information to affect the current answer
'''
    response = llm.generate(prompt, llm.disc_model).strip().lower()
    if "reset" in response:
        return "reset"
    if "update" in response:
        return "update"
    return "none"


def llm_update(chunk: str, query: str, answer: str, llm: LLMClient) -> str:
    prompt = f'''Current maintained answer to "{query}":
"{answer}"

After the following scene:
{chunk}

The answer to "{query}" is detected to be updated.

Please directly output the new maintained answer concisely.
Important:
- Preserve the trace of important transitions instead of only giving the latest state.
- Record how the answer changed when the transition matters for future understanding.
- Keep it compact; do not summarize unrelated events.
- If the previous answer is still relevant, incorporate it as transition history.
- Do not explain your reasoning.

Output only the updated answer.'''
    return llm.generate(prompt, llm.disc_model)


def iterative_search(story_chunks: list[str], query: str, answer: str, llm: LLMClient) -> str:
    for chunk in story_chunks:
        action = llm_check(chunk, query, answer, llm)
        if action == "reset":
            answer = "unknown"
        elif action == "update":
            answer = llm_update(chunk, query, answer, llm)
    return answer


def update_state_bookmark(bookmark: dict, action_seq: list[dict], llm: LLMClient, step: int = 64) -> dict:
    assert bookmark["tag"] == "state"
    start_index = bookmark["index"]
    query, answer = bookmark["query"], bookmark["answer"]
    story_chunks = [
        "\n".join(item["action"] for item in action_seq[idx:idx + step])
        for idx in range(start_index, len(action_seq), step)
    ]
    bookmark["index"] = len(action_seq)
    bookmark["answer"] = iterative_search(story_chunks, query, answer, llm)
    return bookmark


def build_merge_rank_spans(indices: list[int], radius: int) -> list[list[int]]:
    if radius < 0:
        raise ValueError("radius must be >= 0")
    uniq = sorted(set(indices))
    if not uniq:
        return []

    originals = [(max(0, idx - radius), idx + radius + 1, idx) for idx in uniq]
    cur_s, cur_e, cur_count, cur_rep = originals[0][0], originals[0][1], 1, originals[0][2]
    merged = []
    for s, e, idx in originals[1:]:
        if s < cur_e:
            cur_e = max(cur_e, e)
            cur_count += 1
            cur_rep = min(cur_rep, idx)
        else:
            merged.append((cur_s, cur_e, cur_count, cur_rep))
            cur_s, cur_e, cur_count, cur_rep = s, e, 1, idx
    merged.append((cur_s, cur_e, cur_count, cur_rep))
    merged.sort(key=lambda x: (-x[2], -x[3]))
    return [[s, e] for s, e, _, _ in merged]


def update_concept_bookmark(bookmark: dict, action_seq: list[dict], llm: LLMClient, span_radius: int = 8, topk: int = 8) -> dict:
    assert bookmark["tag"] == "concept"
    start_index = bookmark["index"]
    query, answer = bookmark["query"], bookmark["answer"]

    if "keyword" not in bookmark:
        prompt = f"# Query\n{query}\nExtract the keyword of the searched concept from the query for string matching search, directly output it."
        bookmark["keyword"] = llm.generate(prompt).strip()

    keyword = bookmark["keyword"]
    indices = [
        idx for idx, item in enumerate(action_seq[start_index:], start=start_index)
        if keyword.lower() in item["action"].lower()
    ]

    appearance_chunks = []
    for start, end in build_merge_rank_spans(indices, span_radius)[:topk]:
        appearance_chunks.append("\n".join(item["action"] for item in action_seq[start:end]))

    if appearance_chunks:
        keyword_appearance = "\n---\n".join(appearance_chunks)
        prompt = f'''{keyword_appearance}

Based on the appearance of "{keyword}" above, update the answer to "{query}" below and directly output the new one.
Current Answer: {answer}'''
        answer = llm.generate(prompt)

    bookmark["index"] = len(action_seq)
    bookmark["answer"] = answer
    return bookmark


def behavior_yes_no(prompt: str, llm: LLMClient, discriminator: Optional[BehaviorDiscriminator]) -> bool:
    if discriminator is not None and discriminator.available:
        labels, _ = discriminator.predict([prompt], ["yes", "no"])
        return labels[0] == "yes"
    return llm.generate(prompt, llm.disc_model).strip().lower().startswith("yes")


def update_behavioral_bookmark(
    bookmark: dict,
    action_seq: list[dict],
    llm: LLMClient,
    discriminator: Optional[BehaviorDiscriminator] = None,
    scene_window: int = 10,
) -> dict:
    assert bookmark["tag"] == "behavioral"
    start_index = bookmark["index"]
    query, answer = bookmark["query"], bookmark["answer"]
    action_list = bookmark.get("action list", [])
    character = bookmark["character"]

    for idx in range(start_index, len(action_seq)):
        item = action_seq[idx]
        if character not in item.get("characters", []):
            continue
        scene = "\n".join(_item["action"] for _item in action_seq[max(0, idx - scene_window):idx])
        action = item["action"]
        prompt = f'''# Scene
{scene}
# Action by {character}
{action}

We are using the following behavioral query as a filter over past scenes:

# Query
{query}

Question:
Does THIS specific action by {character} in THIS scene provide direct evidence
for the behavior that the query is asking about?

Answer "yes" ONLY if BOTH of the following are true:
- The situation described in the query (characters, conditions, emotional state,
  context, etc.) actually holds in this scene.
- {character}'s action shown above is a concrete example of that behavior in
  this situation (not just vaguely related or partially matching).

If any required condition in the query is missing, unclear, or only weakly
related, answer "no".

Directly answer with a single word: yes or no.
'''
        if behavior_yes_no(prompt, llm, discriminator):
            action_list.append(action)

    if action_list:
        prompt = f'''You are given a list of filtered character actions for {character} that all match a particular behavioral query.

# Character action samples (chronologically ordered):
{action_list}

# Behavioral query:
{query}

Important:
- The action list is sorted chronologically from earliest to latest.
- If earlier and later actions appear to conflict, prioritize the pattern implied by the later actions.
- Focus ONLY on behavioral patterns grounded in the provided samples.

Your task:
Summarize how the character, {character}, behaves in a concise, general sentence or two.
Do NOT add new traits, motivations, or interpretations that are not supported by the actions.

Format:
Return ONLY the final summarized answer, no explanation.'''
        answer = llm.generate(prompt)
    else:
        answer = "unknown"

    bookmark["index"] = len(action_seq)
    bookmark["answer"] = answer
    bookmark["action list"] = action_list
    return bookmark


def normalize_query(query: str) -> list[str]:
    tokens = re.findall(r"\b\w+\b", query.lower())
    return [token for token in tokens if token not in STOPWORDS]


def overlap_rate(query_a: str, query_b: str) -> float:
    tokens_a = set(normalize_query(query_a))
    tokens_b = set(normalize_query(query_b))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / max(1, min(len(tokens_a), len(tokens_b)))


def bookmark_relation(new_bookmark: dict, old_bookmark: dict, llm: LLMClient) -> str:
    prompt = f'''You are checking whether an existing bookmark can support a new bookmark.

Tag: {new_bookmark["tag"]}

Existing bookmark query:
{old_bookmark["query"]}

Existing bookmark answer:
{old_bookmark["answer"]}

New bookmark query:
{new_bookmark["query"]}

Decide the relation from the existing bookmark to the new bookmark.

Return exactly one word:

reuse
- The two queries should share the same maintained bookmark slot.
- They ask for essentially the same information over time.

derive
- The existing bookmark should not be reused as the same slot.
- But its current synchronized answer can be used to initialize the new bookmark.

none
- The existing bookmark is not sufficiently relevant.

Be conservative:
- Use "reuse" only for near-equivalent maintained memory targets.
- Use "derive" for closely related but non-identical queries.
- Use "none" otherwise.

Directly output one word: reuse, derive, or none.
'''
    response = llm.generate(prompt).strip().lower()
    if response.startswith("reuse"):
        return "reuse"
    if response.startswith("derive"):
        return "derive"
    return "none"


def derive_bookmark_answer(new_bookmark: dict, old_bookmark: dict, llm: LLMClient, scene: Optional[str] = None) -> str:
    scene_block = f"# Current Scene\n{scene}\n\n" if scene is not None else ""
    prompt = f'''{scene_block}You are initializing a new bookmark from an existing bookmark.

# Existing bookmark query
{old_bookmark["query"]}

# Existing bookmark answer
{old_bookmark["answer"]}

# New bookmark query
{new_bookmark["query"]}

Write an initial answer for the new bookmark using the existing bookmark answer as a basis.

Requirements:
- Be concise.
- Be conservative.
- Do not invent unsupported details.
- If the existing bookmark is not enough, output exactly: unknown

Return only the answer.
'''
    answer = llm.generate(prompt).strip()
    return answer if answer else "unknown"


def find_supporting_bookmark(new_bookmark: dict, bookmarks: list[dict], llm: LLMClient, topk: int = 5) -> tuple[str, Optional[int]]:
    scored_candidates = []
    for idx, old_bookmark in enumerate(bookmarks):
        if old_bookmark["tag"] != new_bookmark["tag"] or old_bookmark["index"] == 0:
            continue
        if new_bookmark["tag"] == "behavioral" and new_bookmark["character"] != old_bookmark["character"]:
            continue
        scored_candidates.append((overlap_rate(new_bookmark["query"], old_bookmark["query"]), idx))

    scored_candidates.sort(reverse=True)
    for _, idx in scored_candidates[:topk]:
        relation = bookmark_relation(new_bookmark, bookmarks[idx], llm)
        if relation in {"reuse", "derive"}:
            return relation, idx
    return "new", None


def init_runtime_fields(bookmark: dict) -> dict:
    if bookmark["tag"] == "behavioral" and "action list" not in bookmark:
        bookmark["action list"] = []
    return bookmark


def attach_or_reuse_bookmarks(new_bookmarks: list[dict], bookmarks: list[dict], llm: LLMClient, scene: Optional[str] = None, topk: int = 5) -> list[int]:
    active_bookmark_indices = []
    for new_bookmark in new_bookmarks:
        relation, parent_idx = find_supporting_bookmark(new_bookmark, bookmarks, llm, topk=topk)
        if relation == "reuse" and parent_idx is not None:
            active_bookmark_indices.append(parent_idx)
            continue

        current_bookmark = dict(new_bookmark)
        if relation == "derive" and parent_idx is not None:
            parent_bookmark = bookmarks[parent_idx]
            current_bookmark["answer"] = derive_bookmark_answer(current_bookmark, parent_bookmark, llm, scene=scene)
            current_bookmark["index"] = parent_bookmark["index"]
            current_bookmark["character"] = current_bookmark.get("character", parent_bookmark.get("character"))
            current_bookmark["derived_from"] = parent_idx
            if current_bookmark["tag"] == "behavioral":
                current_bookmark["action list"] = []
            current_bookmark.pop("keyword", None)
        else:
            current_bookmark["answer"] = current_bookmark.get("answer", "unknown")
            current_bookmark["index"] = current_bookmark.get("index", 0)

        current_bookmark = init_runtime_fields(current_bookmark)
        active_bookmark_indices.append(len(bookmarks))
        bookmarks.append(current_bookmark)

    deduped_indices = []
    seen = set()
    for idx in active_bookmark_indices:
        if idx not in seen:
            deduped_indices.append(idx)
            seen.add(idx)
    return deduped_indices


def update_bookmark(bookmark: dict, history_seq: list[dict], llm: LLMClient, discriminator: Optional[BehaviorDiscriminator], step: int, concept_span_radius: int, concept_topk: int, behavior_scene_window: int) -> dict:
    tag = bookmark["tag"]
    if tag == "concept":
        return update_concept_bookmark(bookmark, history_seq, llm, span_radius=concept_span_radius, topk=concept_topk)
    if tag == "state":
        return update_state_bookmark(bookmark, history_seq, llm, step=step)
    if tag == "behavioral":
        return update_behavioral_bookmark(bookmark, history_seq, llm, discriminator=discriminator, scene_window=behavior_scene_window)
    raise ValueError(f"Unknown bookmark tag: {tag}")
