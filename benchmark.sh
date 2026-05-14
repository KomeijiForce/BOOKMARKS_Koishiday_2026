bash scripts/run_bookmarks.sh "Poppin'Party" \
  --data-dir data \
  --output-dir outputs/popipa \
  --n-query 5 \
  --step 64 \
  --classifier-path "KomeijiForce/deberta-v3-base-behavior-check-v4-0"\
  --device "cuda:0"\
  --metrics "em" \
  --max-test-instances 100
