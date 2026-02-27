#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@dataclass(frozen=True)
class ModelSpec:
    key: str
    card: str
    output_file: str
    use_lang_prompt: bool


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="ctc_300m",
        card="omniASR_CTC_300M_v2",
        output_file="predictions_ctc_300m.csv",
        use_lang_prompt=False,
    ),
    ModelSpec(
        key="llm_300m",
        card="omniASR_LLM_300M_v2",
        output_file="predictions_llm_300m.csv",
        use_lang_prompt=True,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run ASR inference on Russian data with CTC+LLM 300M models "
            "and compute raw WER."
        )
    )
    parser.add_argument("--manifest", required=True, help="Path to CSV/TSV manifest.")
    parser.add_argument(
        "--audio-col",
        default="audio_path",
        help="Manifest column containing audio paths.",
    )
    parser.add_argument(
        "--text-col", default="text", help="Manifest column containing references."
    )
    parser.add_argument(
        "--sep",
        default=None,
        help="CSV separator. Auto-detected from extension if not specified.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of samples processed per batch.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Inference device. auto selects cuda if available.",
    )
    parser.add_argument(
        "--output-dir",
        default="./outputs/infer_wer",
        help="Directory for predictions and summary CSV files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for first N manifest rows.",
    )
    parser.add_argument(
        "--skip-missing-audio",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip rows with missing audio files.",
    )
    parser.add_argument(
        "--fail-on-error",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stop execution when inference fails for any sample.",
    )
    return parser.parse_args()


def resolve_separator(manifest_path: Path, sep: str | None) -> str:
    if sep is not None:
        return sep
    suffix = manifest_path.suffix.lower()
    if suffix in {".tsv", ".tab"}:
        return "\t"
    return ","


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def chunked(items: list[int], chunk_size: int) -> Iterable[list[int]]:
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def safe_error_text(error: Exception) -> str:
    text = str(error).replace("\n", " ").strip()
    return text[:4000]


def word_edit_distance(ref_text: str, hyp_text: str) -> tuple[int, int]:
    ref_tokens = ref_text.strip().split()
    hyp_tokens = hyp_text.strip().split()

    m = len(ref_tokens)
    n = len(hyp_tokens)
    if m == 0:
        return n, 0
    if n == 0:
        return m, m

    prev_row = list(range(n + 1))
    for i in range(1, m + 1):
        cur_row = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if ref_tokens[i - 1] == hyp_tokens[j - 1] else 1
            cur_row[j] = min(
                prev_row[j] + 1,
                cur_row[j - 1] + 1,
                prev_row[j - 1] + cost,
            )
        prev_row = cur_row

    return prev_row[n], m


def run_model_inference(
    spec: ModelSpec,
    base_rows: list[dict[str, Any]],
    batch_size: int,
    device: str,
    fail_on_error: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

    rows = [dict(r) for r in base_rows]
    print(f"[{spec.key}] Loading pipeline ({spec.card}) on device={device}")
    pipeline = ASRInferencePipeline(model_card=spec.card, device=device)

    pending_indices = [i for i, row in enumerate(rows) if row["status"] == "pending"]
    total_pending = len(pending_indices)
    print(f"[{spec.key}] Pending samples for inference: {total_pending}")

    for batch_no, batch_indices in enumerate(chunked(pending_indices, batch_size), 1):
        audio_paths = [rows[i]["audio_path"] for i in batch_indices]
        try:
            if spec.use_lang_prompt:
                hypotheses = pipeline.transcribe(
                    audio_paths,
                    lang=["rus_Cyrl"] * len(audio_paths),
                    batch_size=len(audio_paths),
                )
            else:
                hypotheses = pipeline.transcribe(audio_paths, batch_size=len(audio_paths))

            if len(hypotheses) != len(batch_indices):
                raise RuntimeError(
                    "Unexpected number of hypotheses: "
                    f"expected={len(batch_indices)} got={len(hypotheses)}"
                )

            for row_idx, hyp in zip(batch_indices, hypotheses):
                rows[row_idx]["hypothesis"] = hyp
                rows[row_idx]["status"] = "ok"
                rows[row_idx]["error"] = ""
        except Exception as batch_error:
            print(
                f"[{spec.key}] Batch {batch_no} failed, falling back to per-sample: "
                f"{safe_error_text(batch_error)}"
            )
            for row_idx in batch_indices:
                audio_path = rows[row_idx]["audio_path"]
                try:
                    if spec.use_lang_prompt:
                        hyp = pipeline.transcribe(
                            [audio_path], lang=["rus_Cyrl"], batch_size=1
                        )[0]
                    else:
                        hyp = pipeline.transcribe([audio_path], batch_size=1)[0]
                    rows[row_idx]["hypothesis"] = hyp
                    rows[row_idx]["status"] = "ok"
                    rows[row_idx]["error"] = ""
                except Exception as sample_error:
                    rows[row_idx]["status"] = "infer_error"
                    rows[row_idx]["error"] = safe_error_text(sample_error)
                    if fail_on_error:
                        raise RuntimeError(
                            f"[{spec.key}] Inference failed for {audio_path}"
                        ) from sample_error

    total_edits = 0
    total_ref_words = 0
    n_ok = 0
    for row in rows:
        if row["status"] != "ok":
            row["wer"] = math.nan
            row["ref_word_count"] = 0
            row["edit_distance"] = 0
            continue

        edits, ref_words = word_edit_distance(row["reference"], row["hypothesis"])
        row["edit_distance"] = edits
        row["ref_word_count"] = ref_words
        row["wer"] = edits / max(1, ref_words)
        total_edits += edits
        total_ref_words += ref_words
        n_ok += 1

    n_total = len(rows)
    n_failed = n_total - n_ok
    aggregate_wer = math.nan if n_ok == 0 else total_edits / max(1, total_ref_words)
    summary = {
        "model": spec.key,
        "n_total": n_total,
        "n_ok": n_ok,
        "n_failed": n_failed,
        "wer": aggregate_wer,
    }

    df = pd.DataFrame(rows)
    ordered_cols = [
        "id",
        "audio_path",
        "reference",
        "hypothesis",
        "model",
        "wer",
        "ref_word_count",
        "edit_distance",
        "status",
        "error",
    ]
    df = df[ordered_cols]
    return df, summary


def load_manifest(
    manifest_path: Path,
    sep: str,
    audio_col: str,
    text_col: str,
    limit: int | None,
) -> pd.DataFrame:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    manifest_df = pd.read_csv(manifest_path, sep=sep)
    missing_cols = [col for col in (audio_col, text_col) if col not in manifest_df]
    if missing_cols:
        raise ValueError(
            "Manifest is missing required columns: " + ", ".join(missing_cols)
        )

    manifest_df = manifest_df[[audio_col, text_col]].copy()
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be > 0")
        manifest_df = manifest_df.head(limit)

    manifest_df[audio_col] = manifest_df[audio_col].fillna("").astype(str)
    manifest_df[text_col] = manifest_df[text_col].fillna("").astype(str)
    manifest_df = manifest_df.reset_index(drop=True)
    return manifest_df


def build_base_rows(
    manifest_df: pd.DataFrame,
    audio_col: str,
    text_col: str,
    skip_missing_audio: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    missing_audio: list[str] = []
    for idx, row in manifest_df.iterrows():
        audio_path = row[audio_col].strip()
        ref_text = row[text_col]
        status = "pending"
        error = ""

        path_exists = bool(audio_path) and Path(audio_path).is_file()
        if not path_exists:
            status = "missing_audio"
            error = f"Audio file not found: {audio_path}"
            missing_audio.append(audio_path)

        rows.append(
            {
                "id": int(idx),
                "audio_path": audio_path,
                "reference": ref_text,
                "hypothesis": "",
                "model": "",
                "wer": math.nan,
                "ref_word_count": 0,
                "edit_distance": 0,
                "status": status,
                "error": error,
            }
        )

    if missing_audio and not skip_missing_audio:
        preview = ", ".join(missing_audio[:3])
        raise FileNotFoundError(
            f"Found {len(missing_audio)} missing audio files, for example: {preview}"
        )

    if missing_audio:
        print(f"Missing audio files skipped: {len(missing_audio)}")
    return rows


def main() -> None:
    args = parse_args()

    manifest_path = Path(args.manifest)
    sep = resolve_separator(manifest_path, args.sep)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Loading manifest from {manifest_path} with sep={repr(sep)}; "
        f"device={device}; batch_size={args.batch_size}"
    )
    manifest_df = load_manifest(
        manifest_path=manifest_path,
        sep=sep,
        audio_col=args.audio_col,
        text_col=args.text_col,
        limit=args.limit,
    )
    print(f"Loaded rows: {len(manifest_df)}")

    base_rows = build_base_rows(
        manifest_df=manifest_df,
        audio_col=args.audio_col,
        text_col=args.text_col,
        skip_missing_audio=args.skip_missing_audio,
    )

    summary_rows: list[dict[str, Any]] = []
    for spec in MODEL_SPECS:
        model_rows = [dict(r) for r in base_rows]
        for row in model_rows:
            row["model"] = spec.key
        predictions_df, summary = run_model_inference(
            spec=spec,
            base_rows=model_rows,
            batch_size=args.batch_size,
            device=device,
            fail_on_error=args.fail_on_error,
        )
        predictions_path = output_dir / spec.output_file
        predictions_df.to_csv(predictions_path, index=False)
        summary_rows.append(summary)
        print(
            f"[{spec.key}] saved {predictions_path}; "
            f"n_ok={summary['n_ok']} n_failed={summary['n_failed']} wer={summary['wer']}"
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
