import argparse
import asyncio
import csv
import os
from typing import Any

from vllm.benchmarks.serve import compute_result_filename, main_async

# Import patch to register daily-omni dataset and omni backends
# This monkey-patches vllm.benchmarks.datasets.get_samples before it's used
# Must be imported before any vllm.benchmarks module usage
import vllm_omni.benchmarks.patch.patch  # noqa: F401


# Ordered columns for the CSV summary row.
# Only flat scalar fields are included; per-request arrays and nested
# structures are skipped via csv.DictWriter(extrasaction="ignore").
_CSV_COLUMNS: list[str] = [
    # Metadata
    "date",
    "label",
    "model_id",
    "backend",
    "tokenizer_id",
    "num_prompts",
    "request_rate",
    "burstiness",
    "max_concurrency",
    # Completion
    "completed",
    "failed",
    "duration",
    # Token / throughput
    "total_input_tokens",
    "total_output_tokens",
    "request_throughput",
    "request_goodput",
    "output_throughput",
    "total_token_throughput",
    "max_output_tokens_per_s",
    "max_concurrent_requests",
    # Audio aggregates
    "total_audio_duration_s",
    "total_audio_frames",
    "audio_throughput",
    "rtfx",
    # --- Text metrics (mean, median, std, p99) ---
    "mean_ttft_ms",
    "median_ttft_ms",
    "std_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "std_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "median_itl_ms",
    "std_itl_ms",
    "p99_itl_ms",
    "mean_e2el_ms",
    "median_e2el_ms",
    "std_e2el_ms",
    "p99_e2el_ms",
    # --- Audio metrics (mean, median, std, p99) ---
    "mean_audio_ttfp_ms",
    "median_audio_ttfp_ms",
    "std_audio_ttfp_ms",
    "p99_audio_ttfp_ms",
    "mean_audio_e2el_ms",
    "median_audio_e2el_ms",
    "std_audio_e2el_ms",
    "p99_audio_e2el_ms",
    "mean_audio_itl_ms",
    "median_audio_itl_ms",
    "std_audio_itl_ms",
    "p99_audio_itl_ms",
    "mean_audio_text_gap_ms",
    "median_audio_text_gap_ms",
    "std_audio_text_gap_ms",
    "p99_audio_text_gap_ms",
    "mean_audio_rtf",
    "median_audio_rtf",
    "std_audio_rtf",
    "p99_audio_rtf",
    "mean_audio_duration_s",
    "median_audio_duration_s",
    "std_audio_duration_s",
    "p99_audio_duration_s",
    # Pipeline
    "mean_pipeline_ratio",
]


def _write_result_csv(
    result_json: dict[str, Any],
    csv_filename: str,
    append: bool = False,
) -> None:
    """Write a single-row CSV summary of benchmark results.

    Extracts flat scalar fields from *result_json*.  Per-request arrays
    and nested structures are silently skipped.
    """
    file_exists = os.path.exists(csv_filename) and os.path.getsize(csv_filename) > 0
    write_header = not (append and file_exists)

    mode = "a" if (append and file_exists) else "w"
    with open(csv_filename, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(result_json)


def main(args: argparse.Namespace) -> dict[str, Any]:
    if getattr(args, "seed_tts_wer_eval", False):
        os.environ["SEED_TTS_WER_EVAL"] = "1"
    if getattr(args, "seed_tts_wer_save_items", False):
        os.environ["SEED_TTS_WER_SAVE_ITEMS"] = "1"
    if getattr(args, "daily_omni_save_eval_items", False):
        os.environ["DAILY_OMNI_SAVE_EVAL_ITEMS"] = "1"

    result = asyncio.run(main_async(args))

    # Write a CSV summary alongside the JSON when results are being saved.
    if args.save_result or args.append_result:
        # Reconstruct the JSON filename the same way main_async did.
        model_id: str = result.get("model_id", args.model)
        label: str = result.get("label", args.label or args.backend)
        current_dt: str = result.get("date", "")
        json_filename = compute_result_filename(args, model_id, label, current_dt)
        if json_filename:
            csv_filename = json_filename.rsplit(".json", 1)[0] + ".csv"
            _write_result_csv(result, csv_filename, append=args.append_result)

    return result
