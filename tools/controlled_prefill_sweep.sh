#!/usr/bin/env bash

set -euo pipefail

MODEL="/data/models/Qwen3-Omni-30B-A3B-Instruct"
TIMING_PATH="/home/prefill_timing_controlled.jsonl"
RESULT_DIR="/home/res/controlled_prefill"

rm -f "$TIMING_PATH"
mkdir -p "$RESULT_DIR"

for batch_size in 1 2 4 8; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] controlled prefill batch_size=$batch_size"
    vllm bench serve \
        --omni \
        --dataset-name random-mm \
        --port 8004 \
        --model "$MODEL" \
        --endpoint /v1/chat/completions \
        --backend openai-chat-omni \
        --request-rate inf \
        --burstiness 1 \
        --num-prompts "$batch_size" \
        --max-concurrency "$batch_size" \
        --random-input-len 1024 \
        --random-range-ratio 0.0 \
        --random-output-len 1 \
        --output-len 1 \
        --random-mm-base-items-per-request 0 \
        --random-mm-num-mm-items-range-ratio 0 \
        --random-mm-limit-mm-per-prompt '{"image":1,"video":1,"audio":1}' \
        --random-mm-bucket-config '{(256, 256, 1): 0.5, (720, 1280, 16): 0.4, (0, 1, 5): 0.10}' \
        --ignore-eos \
        --temperature 0 \
        --trust-remote-code \
        --request-id-prefix "controlled-bs${batch_size}-" \
        --save-detailed \
        --save-result \
        --result-dir "$RESULT_DIR/bs${batch_size}" \
        --extra_body '{"modalities":["text","audio"]}' \
        > "$RESULT_DIR/bs${batch_size}.log" 2>&1
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] controlled prefill sweep finished"
echo "timing: $TIMING_PATH"
