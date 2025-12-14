import argparse
import os
import re
import sys
import logging
import torch
import torchaudio
import soundfile as sf
import numpy as np

# Ensure we can import local modules
sys.path.append(os.getcwd())

from tokenizer import StepAudioTokenizer
from model_loader import model_loader, ModelSource
from stepvocoder.cosyvoice2.cli.cosyvoice import CosyVoice

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Convert audio tokens to WAV using StepAudio components.")
    
    parser.add_argument(
        "--tokens", 
        type=str, 
        default="<audio_958><audio_101><audio_2216><audio_1028><audio_2892>",
        help="Audio token string (e.g. '<audio_123><audio_456>')."
    )
    parser.add_argument(
        "--prompt-wav", 
        type=str, 
        default="examples/paralingustic_prompt.wav",
        help="Path to the prompt audio WAV file."
    )
    parser.add_argument(
        "--prompt-text", 
        type=str, 
        default=None,
        help="Text corresponding to the prompt audio. If not provided, tries to read from examples/text.txt."
    )
    parser.add_argument(
        "--output-wav", 
        type=str, 
        default="output.wav",
        help="Output WAV file path."
    )
    parser.add_argument(
        "--model-path", 
        type=str, 
        default=None,
        help="Path to the model directory containing 'CosyVoice-300M-25Hz'. If not set, tries to detect or use default locations."
    )
    parser.add_argument(
        "--tokenizer-path", 
        type=str, 
        default="/root/Step-Audio-Tokenizer",
        help="Path for tokenizer resources (encoder path)."
    )
    parser.add_argument(
        "--tokenizer-model-id",
        type=str,
        default="dengcunqin/speech_paraformer-large_asr_nat-zh-cantonese-en-16k-vocab8501-online",
        help="FunASR model ID for the tokenizer."
    )

    return parser.parse_args()



def get_model_path(provided_path):
    """
    Resolve the model path.
    """
    # Common paths to check if not provided
    candidates = [
        "/.autodl/stepfun-ai/",
        "StepAudio/Step-Audio-TTS-3B", # HuggingFace/ModelScope ID
    ]
    
    if provided_path:
        candidates.insert(0, provided_path)
        
    for path in candidates:
        if not path: continue
        
        # If it looks like a local path
        if os.path.exists(path) or path.startswith("/"):
             if os.path.exists(path):
                 return path
             # If it's an absolute path but doesn't exist, we skip unless it was the user provided one?
             # But if user provided it and it doesn't exist, we might want to try to download if it looks like an ID.
             if path == provided_path:
                 # If user provided a path/ID, let model_loader handle it if it's not a simple non-existent local path
                 return path 
        else:
             # It's a model ID
             return path
             
    return "StepAudio/Step-Audio-TTS-3B" # Default fallback

def preprocess_prompt_wav(cosy_model, audio_tokenizer, prompt_wav_path):
    """
    Process prompt audio to extract features and tokens.
    """
    logger.info(f"Loading prompt wav: {prompt_wav_path}")
    prompt_wav, prompt_wav_sr = torchaudio.load(prompt_wav_path)
    if prompt_wav.shape[0] > 1:
        prompt_wav = prompt_wav.mean(dim=0, keepdim=True)

    # volume-normalize avoid clipping
    norm = torch.max(torch.abs(prompt_wav), dim=1, keepdim=True)[0]
    if norm > 0.6: 
        prompt_wav = prompt_wav / norm * 0.6 

    # Extract features using CosyVoice frontend
    speech_feat, speech_feat_len = cosy_model.frontend.extract_speech_feat(
        prompt_wav, prompt_wav_sr
    )
    speech_embedding = cosy_model.frontend.extract_spk_embedding(
        prompt_wav, prompt_wav_sr
    )
    
    # Extract tokens using StepAudioTokenizer
    vq0206_codes, vq02_codes_ori, vq06_codes_ori = audio_tokenizer.wav2token(prompt_wav, prompt_wav_sr)
    
    return vq0206_codes, speech_feat, speech_embedding

def main():
    args = parse_args()

    # 1. Resolve Prompt Text
    if not args.prompt_text:
        logger.warning("No prompt text found or provided. Proceeding without it (Vocoder generation usually relies on audio features).")

    # 2. Load Tokenizer
    logger.info("Initializing StepAudioTokenizer...")
    try:
        # Handle the case where the default path /root/... is not valid on this system
        tokenizer_path = args.tokenizer_path
        if not os.path.exists(tokenizer_path) and tokenizer_path.startswith("/root"):
            # If default /root path doesn't exist, assume we can use current dir or let model_loader handle download
            logger.info(f"Default tokenizer path {tokenizer_path} not found, using current directory as base.")
            tokenizer_path = os.getcwd()
            
        audio_tokenizer = StepAudioTokenizer(
            tokenizer_path,
            model_source=ModelSource.AUTO,
            funasr_model_id=args.tokenizer_model_id
        )
        logger.info("Tokenizer loaded.")
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        return

    # 3. Load CosyVoice Model
    logger.info("Loading CosyVoice model...")
    base_model_path = get_model_path(args.model_path)
    
    # Use model_loader to resolve/download if necessary, but we only want the path
    # We cheat by calling detect_model_source. If it's not local, we trigger a download.
    source = model_loader.detect_model_source(base_model_path)
    final_model_path = base_model_path
    
    if source != ModelSource.LOCAL:
        logger.info(f"Model {base_model_path} detected as {source}. Downloading if not cached...")
        try:
            final_model_path = model_loader._cached_snapshot_download(base_model_path, source)
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            return
    
    # Locate CosyVoice folder
    cosy_model_path = os.path.join(final_model_path, "CosyVoice-300M-25Hz")
    if not os.path.exists(cosy_model_path):
        # Fallback: maybe the user pointed directly to the inner folder?
        if os.path.exists(os.path.join(final_model_path, "cosyvoice.yaml")):
            cosy_model_path = final_model_path
        else:
            logger.warning(f"CosyVoice-300M-25Hz subfolder not found in {final_model_path}. Trying root.")
            cosy_model_path = final_model_path

    try:
        cosy_model = CosyVoice(cosy_model_path)
        logger.info(f"CosyVoice loaded from {cosy_model_path}")
    except Exception as e:
        logger.error(f"Failed to load CosyVoice: {e}")
        return

    # 4. Preprocess Prompt
    try:
        vq0206_codes, speech_feat, speech_embedding = preprocess_prompt_wav(
            cosy_model, audio_tokenizer, args.prompt_wav
        )
    except Exception as e:
        logger.error(f"Failed to process prompt wav: {e}")
        return

    # 5. Parse Input Tokens
    logger.info("Parsing input tokens...")
    # Extract numbers from <audio_N>
    tokens_list = [int(x) for x in re.findall(r"<audio_(\d+)>", args.tokens)]
    
    if not tokens_list:
        logger.error("No valid tokens found in input string. Format should be <audio_123>...")
        return
    
    # Normalize tokens (handle 65536 offset)
    # The tokenizer produces shifted tokens (>65536).
    # The vocoder expects 0-based codebook indices.
    # The input tokens (from LLM) are likely shifted.
    # If the user provides small numbers (e.g. 958), we assume they are already indices.
    
    normalized_tokens = []
    for t in tokens_list:
        if t >= 65536:
            normalized_tokens.append(t - 65536)
        else:
            normalized_tokens.append(t)
            
    token_tensor = torch.tensor([normalized_tokens], dtype=torch.long)
    
    # Shift prompt tokens as well
    prompt_token_tensor = torch.tensor([vq0206_codes], dtype=torch.long)
    prompt_token_tensor = prompt_token_tensor - 65536

    # 6. Generate Audio
    logger.info("Generating audio...")
    try:
        # Ensure types match what StepAudioTTS uses (bfloat16 for features if possible/needed)
        # However, CosyVoice defaults to float32 on CPU.
        # We will cast to the dtype CosyVoice was initialized with (which is float32 by default)
        # implicitly handled by CosyVoice methods or explicitly here.
        
        # tts.py uses bfloat16 explicitly. We should check if we are on CUDA.
        if torch.cuda.is_available():
            speech_feat = speech_feat.to(torch.bfloat16)
            speech_embedding = speech_embedding.to(torch.bfloat16)
        else:
             # On CPU, stay float32
             speech_feat = speech_feat.to(torch.float32)
             speech_embedding = speech_embedding.to(torch.float32)

        out_wav = cosy_model.token2wav_nonstream(
            token_tensor,
            prompt_token_tensor,
            speech_feat,
            speech_embedding,
        )
        
        # 7. Save Output
        out_wav_np = out_wav.detach().cpu().numpy().flatten()
        sf.write(args.output_wav, out_wav_np, 24000)
        logger.info(f"Successfully saved WAV to {args.output_wav}")
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
