#! user/bin/env bash

for alpha in 0.0 0.2 0.4 0.6 0.8 1.0; do
  echo "Running evaluation with alpha=${alpha}..."
  ALPHA="${alpha}" EPOCH=80 LIMIT=10 bash scripts/run_full_evaluation_pipeline.sh
done