from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from .bookmarks import attach_or_reuse_bookmarks, init_bookmarks, update_bookmark
from .classifier import BehaviorDiscriminator
from .data import build_test_ids, load_artifact, scene_before
from .evaluate import benchmark_precision
from .llm import LLMClient
from .predict import next_action_prediction
from .profile import profile_aggregate, profile_extract


def build_report(bookmarks: list[dict], idx: int, character: str, active_recency: int) -> str:
    lines = []
    for bookmark in bookmarks:
        if idx - bookmark.get("index", 0) > active_recency:
            continue
        if bookmark["tag"] == "behavioral" and bookmark.get("character") != character:
            continue
        lines.append(f'# Query: {bookmark["query"]}\n# Answer: {bookmark["answer"]}\n---')
    return "\n".join(lines)


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def maybe_update_profile(character: str, idx: int, action_seq: list[dict], profiles: dict, profile_indices: dict, llm: LLMClient, step: int, update_every: int) -> None:
    if update_every <= 0:
        return
    profile_idx = profile_indices[character]
    if idx - profile_idx < update_every:
        return
    profile = profiles[character]
    for jdx in range(profile_idx, idx, step):
        pairs = action_seq[jdx:min(idx, jdx + step)]
        block_profile = profile_extract(character, pairs, llm)
        profile = profile_aggregate(profile, block_profile, llm)
    profiles[character] = profile
    profile_indices[character] = idx


def run(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir)
    cache_file = args.cache_file or str(output_dir / "prompt_cache.jsonl")
    llm = LLMClient(
        model=args.model,
        eval_model=args.eval_model,
        disc_model=args.disc_model,
        cache_file=cache_file,
        max_retries=args.max_retries,
    )
    discriminator = BehaviorDiscriminator(args.classifier_path, args.device)

    data = load_artifact(args.data_dir, args.artifact)
    action_seq = data["action_seq"]
    original_action_seq = data["original_action_seq"]
    main_characters = data["main_characters"]
    utterance = data["utterance"] if args.utterance is None else args.utterance

    test_ids = build_test_ids(action_seq, main_characters, args.test_split)
    if args.max_test_instances is not None:
        test_ids = test_ids[:args.max_test_instances]

    metrics = [metric.strip() for metric in args.metrics.split(",") if metric.strip()]
    records_path = output_dir / f"{args.artifact}.bookmarks.records.jsonl"
    if records_path.exists() and not args.append:
        records_path.unlink()

    bookmarks: list[dict] = []
    profiles = defaultdict(lambda: "[empty]")
    profile_indices = defaultdict(int)
    results = {character: [] for character in main_characters}
    aggregate_scores = defaultdict(list)

    for idx in tqdm(test_ids, desc=f"BOOKMARKS on {args.artifact}"):
        item = action_seq[idx]
        target_characters = [character for character in item.get("characters", []) if character in main_characters]
        for character in target_characters:
            maybe_update_profile(
                character,
                idx,
                action_seq,
                profiles,
                profile_indices,
                llm,
                step=args.step,
                update_every=args.profile_update_every,
            )
            scene = scene_before(action_seq, idx, args.history_window)
            reference = action_seq[idx]["action"]
            history_seq = original_action_seq[:idx]

            history_report = build_report(bookmarks, idx, character, args.active_recency)
            new_bookmarks = init_bookmarks(
                character,
                scene,
                profiles[character],
                history_report,
                llm,
                utterance=utterance,
                k=args.n_query,
            )
            active_indices = attach_or_reuse_bookmarks(
                new_bookmarks,
                bookmarks,
                llm,
                scene=scene,
                topk=args.reuse_topk,
            )

            for bookmark_idx in active_indices:
                bookmarks[bookmark_idx] = update_bookmark(
                    bookmarks[bookmark_idx],
                    history_seq,
                    llm,
                    discriminator,
                    step=args.step,
                    concept_span_radius=args.concept_span_radius,
                    concept_topk=args.concept_topk,
                    behavior_scene_window=args.behavior_scene_window,
                )

            report = build_report(bookmarks, idx, character, args.active_recency)
            grounding = profiles[character] + "\n\n" + report
            prediction = next_action_prediction(character, scene, grounding, llm, utterance=utterance)
            scores = benchmark_precision(character, scene, prediction, reference, llm, metrics)

            for metric, score in scores.items():
                aggregate_scores[metric].append(score)
            if "em" in scores:
                results[character].append(scores["em"])

            record = {
                "artifact": args.artifact,
                "idx": idx,
                "title": item.get("title"),
                "character": character,
                "scene": scene,
                "reference": reference,
                "prediction": prediction,
                "scores": scores,
                "active_bookmark_indices": active_indices,
                "active_bookmarks": [bookmarks[i] for i in active_indices],
                "grounding": grounding,
            }
            append_jsonl(records_path, record)

            if args.verbose:
                print(f"#{idx} {character} {scores}")
                print("Prediction:", prediction)
                print("Reference:", reference)
                print("-" * 80)

    summary = {
        "artifact": args.artifact,
        "n_instances": sum(len(v) for v in results.values()),
        "metrics": {metric: float(np.mean(values)) if values else None for metric, values in aggregate_scores.items()},
        "character_em": {
            character: float(np.mean(values)) if values else None
            for character, values in results.items()
        },
        "records_path": str(records_path),
        "num_bookmarks": len(bookmarks),
    }
    save_json(output_dir / f"{args.artifact}.bookmarks.summary.json", summary)
    save_json(output_dir / f"{args.artifact}.bookmarks.final_memory.json", {"bookmarks": bookmarks})
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the BOOKMARKS framework on one artifact.")
    parser.add_argument("--artifact", required=True, help="Artifact name, e.g. MyGO!!!!! or Poppin'Party.")
    parser.add_argument("--data-dir", default="../CDT", help="Directory containing all_characters.json and title2action_series.<artifact>.json.")
    parser.add_argument("--output-dir", default="outputs", help="Where records, summaries, and cache are written.")
    parser.add_argument("--cache-file", default=None, help="Prompt cache JSONL path. Defaults to <output-dir>/prompt_cache.jsonl.")
    parser.add_argument("--model", default="gpt-5.1", help="Model used for prediction and most generation calls.")
    parser.add_argument("--eval-model", default="gpt-4.1", help="Model used for evaluation metrics.")
    parser.add_argument("--disc-model", default="gpt-5.4-mini", help="Model used for lightweight yes/no/update decisions.")
    parser.add_argument("--classifier-path", default=None, help="Optional local HF classifier path for behavioral filtering. If omitted, disc-model is used.")
    parser.add_argument("--device", default=None, help="Torch device for classifier, e.g. cuda:0 or cpu.")
    parser.add_argument("--n-query", type=int, default=5, help="Number of bookmark queries proposed per prediction.")
    parser.add_argument("--step", type=int, default=64, help="Chunk size for state/profile updating.")
    parser.add_argument("--history-window", type=int, default=10, help="Number of previous actions shown as the current scene.")
    parser.add_argument("--profile-update-every", type=int, default=16, help="Update profile after this many new actions; set <=0 to disable.")
    parser.add_argument("--concept-span-radius", type=int, default=8, help="Radius around concept keyword hits.")
    parser.add_argument("--concept-topk", type=int, default=8, help="Maximum concept evidence spans.")
    parser.add_argument("--behavior-scene-window", type=int, default=10, help="Previous actions shown when filtering behavioral evidence.")
    parser.add_argument("--reuse-topk", type=int, default=5, help="Candidate bookmarks checked for reuse/derive.")
    parser.add_argument("--active-recency", type=int, default=5, help="Include bookmarks updated within this many action indices in grounding.")
    parser.add_argument("--test-split", type=float, default=0.5, help="Use each character's actions after this fraction as test instances.")
    parser.add_argument("--max-test-instances", type=int, default=None, help="Optional cap for quick debugging.")
    parser.add_argument("--metrics", default="em", help="Comma-separated metrics: em,nli.")
    parser.add_argument("--utterance", action=argparse.BooleanOptionalAction, default=None, help="Override whether predictions should be formatted as character utterances.")
    parser.add_argument("--max-retries", type=int, default=3, help="OpenAI retry count per uncached call.")
    parser.add_argument("--append", action="store_true", help="Append to an existing records JSONL instead of replacing it.")
    parser.add_argument("--verbose", action="store_true", help="Print each prediction and reference.")
    return parser.parse_args(argv)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
