# BOOKMARKS

**Efficient Active Storyline Memory System for Role-playing**

<img width="5550" height="1914" alt="fig_main_v1-1" src="https://github.com/user-attachments/assets/388086c4-7a56-49f8-9120-452d08d48822" />


<img height="96" alt="KomeijiForce_Logo" src="https://github.com/user-attachments/assets/3b931cd1-8ce9-4e89-8852-f20d288cad1d" /> - Let there be fantasy

## What is included

- `src/bookmarks/bookmarks.py` — bookmark proposal, reuse/derive, and concept/state/behavioral bookmark updates.
- `src/bookmarks/benchmark.py` — main argparse entrypoint for benchmarking one artifact.
- `src/bookmarks/data.py` — loaders for the action-series files.
- `src/bookmarks/llm.py` — OpenAI client wrapper with JSONL prompt cache.
- `src/bookmarks/profile.py` — profile extraction/aggregation.
- `src/bookmarks/predict.py` — next-action prediction.
- `src/bookmarks/evaluate.py` — EM/NLI LLM-based scoring.
- `scripts/run_bookmarks.sh` — shell entrypoint.

## Expected data layout

By default, `--data-dir data` should contain:

```text
data/
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
bash scripts/run_bookmarks.sh "Poppin'Party" \
  --data-dir data \
  --output-dir outputs/popipa \
  --n-query 5 \
  --step 64 \
  --classifier-path "KomeijiForce/deberta-v3-base-behavior-check-v4-0"\
  --device "cuda:0"\
  --metrics "em" \
  --max-test-instances 100
```

For a full run, remove `--max-test-instances`.

You can also run the module directly:

```bash
PYTHONPATH=src python -m bookmarks.benchmark \
  --artifact "Poppin'Party" \
  --data-dir data \
  --output-dir outputs/popipa
```

## Useful parameters

- `--artifact`: benchmark artifact name.
- `--data-dir`: directory containing JSON files.
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
