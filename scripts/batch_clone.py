#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import soundfile as sf
import torch

from model_loader import ModelSource
from tokenizer import StepAudioTokenizer
from tts import StepAudioTTS


def _json_dumps(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _parse_torch_dtype(name: str) -> torch.dtype:
    name = (name or "").lower()
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported --torch-dtype={name}; choose from {sorted(mapping)}")
    return mapping[name]


def _load_tasks(tasks_path: Path) -> list[dict]:
    tasks: list[dict] = []
    for i, line in enumerate(tasks_path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            tasks.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON on line {i} of {tasks_path}: {e}") from e
    return tasks


def _resolve_out_wav(output_dir: Path, task: dict, fallback_idx: int) -> Path:
    if isinstance(task.get("output_wav"), str) and task["output_wav"].strip():
        p = Path(task["output_wav"].strip())
        return p if p.is_absolute() else (output_dir / p)

    task_id = str(task.get("id", "")).strip()
    if task_id:
        safe = task_id.replace(os.sep, "_").replace("/", "_")
        return output_dir / f"{safe}.wav"

    prompt_audio = str(task.get("prompt_audio_path", "")).strip()
    stem = Path(prompt_audio).stem if prompt_audio else f"task_{fallback_idx:06d}"
    return output_dir / f"{stem}_cloned.wav"


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Batch zero-shot voice cloning using Step-Audio-EditX (load models once, run many tasks).",
    )
    parser.add_argument("--model-path", type=str, required=True, help="Root with Step-Audio-Tokenizer/ and Step-Audio-EditX/.")
    parser.add_argument("--tasks", type=str, required=True, help="JSONL tasks file.")
    parser.add_argument("--output-dir", type=str, default=str(repo_root / "output" / "clone_batch"), help="Output directory for wav files.")

    parser.add_argument(
        "--model-source",
        type=str,
        default="auto",
        choices=["auto", "local", "modelscope", "huggingface"],
        help="Model source (same as tts_infer.py).",
    )
    parser.add_argument(
        "--tokenizer-model-id",
        type=str,
        default="dengcunqin/speech_paraformer-large_asr_nat-zh-cantonese-en-16k-vocab8501-online",
        help="FunASR model ID for the tokenizer (local subdir or hub id depending on your setup).",
    )
    parser.add_argument("--tts-model-id", type=str, default=None, help="Override TTS model id if needed.")
    parser.add_argument("--quantization", type=str, default=None, choices=["int4", "int8", "awq-4bit"])
    parser.add_argument("--device-map", type=str, default="mps", help="Device map for model loading (e.g., mps/cuda/cpu/auto).")
    parser.add_argument("--torch-dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument(
        "--funasr-device",
        type=str,
        default="cpu",
        choices=["cpu", "mps", "auto"],
        help="Device for FunASR encoder in vq02 (default: cpu).",
    )
    parser.add_argument("--skip-existing", action="store_true", help="Skip tasks if output wav already exists.")
    parser.add_argument("--errors-jsonl", type=str, default=None, help="Write failures to this JSONL.")

    args = parser.parse_args()

    tasks_path = Path(args.tasks)
    if not tasks_path.is_file():
        raise FileNotFoundError(f"Tasks file not found: {tasks_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    errors_path = Path(args.errors_jsonl) if args.errors_jsonl else output_dir / f"{tasks_path.name}.errors.jsonl"

    source_mapping = {
        "auto": ModelSource.AUTO,
        "local": ModelSource.LOCAL,
        "modelscope": ModelSource.MODELSCOPE,
        "huggingface": ModelSource.HUGGINGFACE,
    }
    model_source = source_mapping[args.model_source]
    torch_dtype = _parse_torch_dtype(args.torch_dtype)

    model_root = Path(args.model_path)
    tokenizer_root = model_root / "Step-Audio-Tokenizer"
    tts_root = model_root / "Step-Audio-EditX"
    if not tokenizer_root.exists():
        raise FileNotFoundError(f"Missing tokenizer dir: {tokenizer_root}")
    if not tts_root.exists():
        raise FileNotFoundError(f"Missing TTS dir: {tts_root}")

    audio_tokenizer = StepAudioTokenizer(
        str(tokenizer_root),
        model_source=model_source,
        funasr_model_id=args.tokenizer_model_id,
        funasr_device=args.funasr_device,
    )
    tts_engine = StepAudioTTS(
        str(tts_root),
        audio_tokenizer,
        model_source=model_source,
        tts_model_id=args.tts_model_id,
        quantization_config=args.quantization,
        torch_dtype=torch_dtype,
        device_map=args.device_map,
    )

    tasks = _load_tasks(tasks_path)
    if not tasks:
        print("[batch_clone] No tasks found.", file=sys.stderr)
        return 2

    ok = 0
    skipped = 0
    failed = 0
    t0 = time.time()

    with errors_path.open("a", encoding="utf-8") as err_f:
        for i, task in enumerate(tasks, 1):
            prompt_audio_path = str(task.get("prompt_audio_path", "")).strip()
            prompt_text = str(task.get("prompt_text", "")).strip()
            generated_text = str(task.get("generated_text", "")).strip()

            if not prompt_audio_path or not prompt_text or not generated_text:
                failed += 1
                err_f.write(
                    _json_dumps(
                        {
                            "index": i,
                            "task": task,
                            "error": "Missing required fields: prompt_audio_path, prompt_text, generated_text",
                        }
                    )
                    + "\n"
                )
                continue

            out_wav = _resolve_out_wav(output_dir, task, i)
            out_wav.parent.mkdir(parents=True, exist_ok=True)
            if args.skip_existing and out_wav.exists():
                skipped += 1
                continue

            try:
                audio, sr = tts_engine.clone(prompt_audio_path, prompt_text, generated_text)
                if isinstance(audio, torch.Tensor):
                    audio_np = audio.detach().cpu().numpy().squeeze()
                else:
                    audio_np = audio
                sf.write(str(out_wav), audio_np, int(sr))
                ok += 1
            except Exception as e:
                failed += 1
                err_f.write(
                    _json_dumps(
                        {
                            "index": i,
                            "task": task,
                            "output_wav": str(out_wav),
                            "error": str(e),
                        }
                    )
                    + "\n"
                )

            if i == 1 or i % 5 == 0 or i == len(tasks):
                elapsed = time.time() - t0
                print(
                    f"[batch_clone] {i}/{len(tasks)} ok={ok} skip={skipped} fail={failed} elapsed={elapsed:.1f}s",
                    file=sys.stderr,
                )

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

