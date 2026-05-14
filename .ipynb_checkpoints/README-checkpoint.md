# BOOKMARKS Framework Benchmark Runner

This repo is a cleaned-up Python implementation of the BOOKMARKS framework extracted from the notebook. It intentionally excludes baseline methods and keeps only the bookmark-based memory pipeline.

## What is included

- `src/bookmarks/bookmarks.py` — bookmark proposal, reuse/derive, and concept/state/behavioral bookmark updates.
- `src/bookmarks/benchmark.py` — main argparse entrypoint for benchmarking one artifact.
- `src/bookmarks/data.py` — loaders for the CDT-style action-series files.
- `src/bookmarks/llm.py` — OpenAI client wrapper with JSONL prompt cache.
- `src/bookmarks/profile.py` — profile extraction/aggregation.
- `src/bookmarks/predict.py` — next-action prediction.
- `src/bookmarks/evaluate.py` — EM/NLI LLM-based scoring.
- `scripts/run_bookmarks.sh` — shell entrypoint.

## Expected data layout

By default, `--data-dir ../CDT` should contain:

```text
../CDT/
  all_characters.json
  band2members.json                         # optional; used to decide utterance formatting
  title2action_series.<ARTIFACT>.json
```

Each action item should contain `action` and either `characters` or `character`.

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="your_api_key"
```

Optional: if you want to reproduce the notebook's local DeBERTa behavioral filter, install `torch` and `transformers`, then pass `--classifier-path`.
If `--classifier-path` is omitted, behavioral yes/no filtering uses `--disc-model` instead.

## Run

```bash
bash scripts/run_bookmarks.sh "MyGO!!!!!" \
  --data-dir ../CDT \
  --output-dir outputs/mygo \
  --n-query 5 \
  --step 64 \
  --metrics em \
  --max-test-instances 20
```

For a full run, remove `--max-test-instances`.

You can also run the module directly:

```bash
PYTHONPATH=src python -m bookmarks.benchmark \
  --artifact "Poppin'Party" \
  --data-dir ../CDT \
  --output-dir outputs/poppinparty
```

## Useful parameters

- `--artifact`: benchmark artifact name.
- `--data-dir`: directory containing CDT JSON files.
- `--n-query`: number of bookmark queries proposed per target prediction.
- `--step`: chunk size for state/profile updating.
- `--history-window`: number of previous actions used as current scene.
- `--profile-update-every`: update profile after this many new actions; set `0` to disable.
- `--reuse-topk`: number of existing bookmarks checked for reuse/derive.
- `--concept-span-radius`, `--concept-topk`: concept evidence retrieval settings.
- `--metrics`: `em`, `nli`, or `em,nli`.
- `--model`, `--eval-model`, `--disc-model`: OpenAI models.
- `--classifier-path`: optional local HF sequence-classification model for behavioral filtering.

## Outputs

For each artifact, the runner writes:

```text
<output-dir>/<ARTIFACT>.bookmarks.records.jsonl       # per-instance predictions and active bookmarks
<output-dir>/<ARTIFACT>.bookmarks.summary.json        # aggregate metrics
<output-dir>/<ARTIFACT>.bookmarks.final_memory.json   # final bookmark memory state
<output-dir>/prompt_cache.jsonl                       # prompt cache unless --cache-file is set
```
