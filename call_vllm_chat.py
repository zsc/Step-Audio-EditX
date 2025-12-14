import argparse
from openai import OpenAI
import torchaudio

from tokenizer import StepAudioTokenizer
from tts import StepAudioTTS
from model_loader import ModelSource

def get_wav_token_str(prompt_wav_path):
    prompt_wav, _ = torchaudio.load(prompt_wav_path)
    vq0206_codes, vq02_codes_ori, vq06_codes_ori, speech_feat, _, speech_embedding = (
        preprocess_prompt_wav(prompt_wav_path)
    )
    prompt_speaker = self.generate_clone_voice_id(prompt_text, prompt_wav)
    prompt_wav_tokens = self.audio_tokenizer.merge_vq0206_to_token_str(
        vq02_codes_ori, vq06_codes_ori
    )
    token_ids = self._encode_audio_edit_clone_prompt(
        target_text,
        prompt_text,
        prompt_speaker,
        prompt_wav_tokens,
    )

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
    encoder = StepAudioTokenizer(
        os.path.join(model_path, "Step-Audio-Tokenizer"),
        model_source=model_source,
        funasr_model_id=tokenizer_model_id
    )
    #logger.info("✓ StepAudioTokenizer loaded successfully")

    # Initialize common TTS engine directly
    common_tts_engine = StepAudioTTS(
        '/.autodl/stepfun-ai/Step-Audio-EditX',
        encoder,
        model_source=model_source,
        tts_model_id=args.tts_model_id,
        quantization_config=args.quantization,
        torch_dtype=torch_dtype,
        device_map=args.device_map
    )

    speaker = ""
    prompt_text = "我觉得这个计划大概是可行的，不过还需要再仔细考虑一下。"

    messages = [
        {
            "role": "system",
            "content": f'''Generate audio with the following timbre, prosody and speaking style

[speaker_start]
speaker name: {speaker}
speaker prompt text: 
{prompt_text}
speaker audio tokens: 
{prompt_wav_tokens}
[speaker_end]''',
        },
        {"role": "user", "content": ""},
    ]
    completion = client.chat.completions.create(
        model=args.model_name,
        messages=messages,
        max_tokens=512,   # <-- max new tokens
    )
    res = completion.choices[0].message.content
    print(res)
