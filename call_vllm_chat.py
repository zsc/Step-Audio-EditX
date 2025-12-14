import argparse
from openai import OpenAI
import torchaudio
import torch

from tokenizer import StepAudioTokenizer
from tts import StepAudioTTS
from model_loader import ModelSource
from config.prompts import AUDIO_EDIT_CLONE_SYSTEM_PROMPT_TPL


def wav_path_to_token_str(audio_tokenizer, prompt_wav_path : str):
    prompt_wav, prompt_wav_sr = torchaudio.load(prompt_wav_path)
    if prompt_wav.shape[0] > 1:
        prompt_wav = prompt_wav.mean(dim=0, keepdim=True)  # 将多通道音频转换为单通道

    # volume-normalize avoid clipping
    norm = torch.max(torch.abs(prompt_wav), dim=1, keepdim=True)[0]
    if norm > 0.6: # hard code;  max absolute value is 0.6
        prompt_wav = prompt_wav / norm * 0.6

    _, vq02_codes_ori, vq06_codes_ori = audio_tokenizer.wav2token(prompt_wav, prompt_wav_sr)
    prompt_wav_tokens = audio_tokenizer.merge_vq0206_to_token_str(
        vq02_codes_ori, vq06_codes_ori
    )

    return prompt_wav_tokens # token_str


def prepare_prompt(target_text, prompt_text, prompt_speaker, prompt_wav_tokens):
    prompt = AUDIO_EDIT_CLONE_SYSTEM_PROMPT_TPL.format(
        speaker=prompt_speaker,
        prompt_text=prompt_text,
        prompt_wav_tokens=prompt_wav_tokens
    )
    messages = [
        {
            "role": "system",
            "content": prompt,
        },
        {"role": "user", "content": target_text},
    ]
    return messages

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://127.0.0.1:8000",
        help="VLLM server url.",
    )
    parser.add_argument(
        "--model-name", type=str, default="step-audio-editx-step1", help="Model name."
    )
    args = parser.parse_args()

    server_url = args.server_url + "/v1"  # for chat route
    client = OpenAI(base_url=server_url, api_key="whatever")

    model_path = '/.autodl/stepfun-ai/'
    model_source = ModelSource.LOCAL
    tokenizer_model_id = 'dengcunqin/speech_paraformer-large_asr_nat-zh-cantonese-en-16k-vocab8501-online'
    # Load StepAudioTokenizer
    audio_tokenizer = StepAudioTokenizer(
        "/root/Step-Audio-Tokenizer",
        model_source=model_source,
        funasr_model_id=tokenizer_model_id
    )
    #logger.info("✓ StepAudioTokenizer loaded successfully")

    speaker = ""
    prompt_text = "我觉得这个计划大概是可行的，不过还需要再仔细考虑一下。"
    target_text = "Testing the audio generation capabilities."
    prompt_wav_path = "examples/en_happy_prompt.wav"

    prompt_wav_tokens = wav_path_to_token_str(audio_tokenizer, prompt_wav_path)

    messages = prepare_prompt(target_text, prompt_text, speaker, prompt_wav_tokens)

    completion = client.chat.completions.create(
        model=args.model_name,
        messages=messages,
        max_tokens=512,   # <-- max new tokens
    )
    res = completion.choices[0].message.content
    print(res)
