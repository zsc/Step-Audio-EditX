#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torchaudio

from tokenizer import StepAudioTokenizer


def _json_dumps(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _iter_audio_files(input_dir: Path, pattern: str) -> list[Path]:
    # pattern can be like "*.wav" or "**/*.wav"
    return sorted(p for p in input_dir.glob(pattern) if p.is_file())


def _read_list_file(list_file: Path) -> list[Path]:
    paths: list[Path] = []
    for line in list_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        paths.append(Path(line))
    return paths


def _normalize_mono_wav(wav: torch.Tensor) -> torch.Tensor:
    # wav: (C, T)
    if wav.ndim != 2:
        raise ValueError(f"Expected wav shape (C, T), got {tuple(wav.shape)}")
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    max_abs = wav.abs().max().item() if wav.numel() else 0.0
    if max_abs > 0.6:
        wav = wav / max_abs * 0.6
    return wav


def _safe_relpath(path: Path, base: Path | None) -> str:
    if base is None:
        return str(path)
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except Exception:
        return str(path)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Batch convert WAV files to Step-Audio token strings (<audio_...>).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Model root containing Step-Audio-Tokenizer/ (same convention as tts_infer.py).",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Direct path to Step-Audio-Tokenizer/ (overrides --model-path).",
    )
    parser.add_argument(
        "--funasr-model-id",
        type=str,
        default="dengcunqin/speech_paraformer-large_asr_nat-zh-cantonese-en-16k-vocab8501-online",
        help="FunASR model directory name under Step-Audio-Tokenizer/.",
    )
    parser.add_argument(
        "--funasr-device",
        type=str,
        default="cpu",
        choices=["cpu", "mps", "auto"],
        help="Device for FunASR encoder in vq02 (default: cpu).",
    )
    parser.add_argument(
        "--torch-num-threads",
        type=int,
        default=None,
        help="Override torch.set_num_threads; can also use STEPAUDIO_TORCH_NUM_THREADS env.",
    )
    parser.add_argument(
        "--cosy-tokenizer-provider",
        action="append",
        default=None,
        help="ONNXRuntime provider for vq06 (repeatable). Default: CPUExecutionProvider.",
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input-dir", type=str, help="Directory containing WAV files.")
    src.add_argument("--input-list", type=str, help="Text file: one WAV path per line.")
    parser.add_argument(
        "--glob",
        type=str,
        default="**/*.wav",
        help="Glob pattern under --input-dir (default: **/*.wav).",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default=str(repo_root / "output" / "wav2token.jsonl"),
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to output JSONL instead of overwriting.",
    )
    parser.add_argument(
        "--errors-jsonl",
        type=str,
        default=None,
        help="Write failures to this JSONL (default: <output>.errors.jsonl).",
    )
    parser.add_argument(
        "--rel-to",
        type=str,
        default=str(repo_root),
        help="Write paths relative to this base in output JSONL (default: repo root).",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Skip files longer than this many seconds (based on original sr).",
    )
    parser.add_argument(
        "--disable-trim",
        action="store_true",
        help="Disable silence trimming inside wav2token (default: enabled).",
    )
    parser.add_argument(
        "--disable-energy-norm",
        action="store_true",
        help="Disable energy normalization inside wav2token (default: enabled).",
    )

    args = parser.parse_args()

    if args.tokenizer_path:
        tokenizer_path = Path(args.tokenizer_path)
    else:
        if not args.model_path:
            raise SystemExit("Either --tokenizer-path or --model-path must be provided.")
        tokenizer_path = Path(args.model_path) / "Step-Audio-Tokenizer"

    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer path not found: {tokenizer_path}")

    rel_base = Path(args.rel_to) if args.rel_to else None

    if args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.is_dir():
            raise NotADirectoryError(f"--input-dir is not a directory: {input_dir}")
        wav_paths = _iter_audio_files(input_dir, args.glob)
    else:
        list_file = Path(args.input_list)
        if not list_file.is_file():
            raise FileNotFoundError(f"--input-list not found: {list_file}")
        wav_paths = _read_list_file(list_file)

    if not wav_paths:
        print("[batch_wav2token] No input files found.", file=sys.stderr)
        return 2

    providers = args.cosy_tokenizer_provider
    if providers is None:
        providers = ["CPUExecutionProvider"]

    tokenizer = StepAudioTokenizer(
        str(tokenizer_path),
        funasr_model_id=args.funasr_model_id,
        funasr_device=args.funasr_device,
        torch_num_threads=args.torch_num_threads,
        cosy_tokenizer_providers=providers,
    )

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    err_path = Path(args.errors_jsonl) if args.errors_jsonl else Path(f"{out_path}.errors.jsonl")
    err_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if args.append else "w"
    processed = 0
    skipped = 0
    failed = 0
    t0 = time.time()

    with out_path.open(mode, encoding="utf-8") as out_f, err_path.open(mode, encoding="utf-8") as err_f:
        for idx, wav_path in enumerate(wav_paths, 1):
            wav_path = wav_path.expanduser()
            try:
                wav, sr = torchaudio.load(str(wav_path))
                wav = _normalize_mono_wav(wav)

                num_samples = int(wav.shape[1])
                duration_sec = (num_samples / float(sr)) if sr else 0.0
                if args.max_seconds is not None and duration_sec > args.max_seconds:
                    skipped += 1
                    continue

                _, vq02_ori, vq06_ori = tokenizer.wav2token(
                    wav,
                    sr,
                    enable_trim=(not args.disable_trim),
                    energy_norm=(not args.disable_energy_norm),
                )
                token_str = tokenizer.merge_vq0206_to_token_str(vq02_ori, vq06_ori)

                rec = {
                    "wav": _safe_relpath(Path(wav_path), rel_base),
                    "sr": int(sr),
                    "duration_sec": round(duration_sec, 6),
                    "tokens": token_str,
                }
                out_f.write(_json_dumps(rec) + "\n")
                processed += 1
            except Exception as e:
                failed += 1
                err_f.write(
                    _json_dumps(
                        {
                            "wav": _safe_relpath(Path(wav_path), rel_base),
                            "error": str(e),
                        }
                    )
                    + "\n"
                )

            if idx == 1 or idx % 10 == 0 or idx == len(wav_paths):
                elapsed = time.time() - t0
                print(
                    f"[batch_wav2token] {idx}/{len(wav_paths)} "
                    f"ok={processed} skip={skipped} fail={failed} elapsed={elapsed:.1f}s",
                    file=sys.stderr,
                )

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

