#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


PROMPT_WAV_NAME = "paralingustic_prompt.wav"
PROMPT_TEXT = "我觉得这个计划大概是可行的，不过还需要再仔细考虑一下。"


def _shell_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    tts_infer_py = repo_root / "tts_infer.py"
    prompt_wav = repo_root / "examples" / PROMPT_WAV_NAME

    parser = argparse.ArgumentParser(
        description="Run zero-shot voice cloning using examples/paralingustic_prompt.wav as the reference.",
    )
    parser.add_argument("--model-path", type=str, required=True, help="Downloaded model root directory.")
    parser.add_argument(
        "--generated-text",
        type=str,
        default=PROMPT_TEXT,
        help="Text to synthesize with the cloned voice (defaults to the prompt text).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(repo_root / "output"),
        help="Directory to save generated wav files.",
    )
    parser.add_argument(
        "--model-source",
        type=str,
        default="auto",
        choices=["auto", "local", "modelscope", "huggingface"],
        help="Model source (same as tts_infer.py).",
    )
    parser.add_argument("--device-map", type=str, default="cuda", help="Device mapping (same as tts_infer.py).")
    parser.add_argument("--torch-dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--quantization", type=str, default=None, choices=["int4", "int8"])
    parser.add_argument("--tokenizer-model-id", type=str, default=None, help="Override tokenizer model id if needed.")
    parser.add_argument("--tts-model-id", type=str, default=None, help="Override TTS model id if needed.")

    args = parser.parse_args()

    if not tts_infer_py.is_file():
        raise FileNotFoundError(f"Missing file: {tts_infer_py}")
    if not prompt_wav.is_file():
        raise FileNotFoundError(f"Missing file: {prompt_wav}")

    cmd: list[str] = [
        sys.executable,
        str(tts_infer_py),
        "--model-path",
        args.model_path,
        "--output-dir",
        args.output_dir,
        "--model-source",
        args.model_source,
        "--device-map",
        args.device_map,
        "--torch-dtype",
        args.torch_dtype,
        "--prompt-text",
        PROMPT_TEXT,
        "--prompt-audio-path",
        str(prompt_wav),
        "--generated-text",
        args.generated_text,
        "--edit-type",
        "clone",
    ]

    if args.quantization:
        cmd += ["--quantization", args.quantization]
    if args.tokenizer_model_id:
        cmd += ["--tokenizer-model-id", args.tokenizer_model_id]
    if args.tts_model_id:
        cmd += ["--tts-model-id", args.tts_model_id]

    print(_shell_join(cmd))
    subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

