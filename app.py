import gradio as gr
import os
import argparse
import torch
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
from datetime import datetime
import torchaudio
import librosa
import soundfile as sf
from typing import Optional
from huggingface_hub import snapshot_download

# Project imports
from tokenizer import StepAudioTokenizer
from tts import StepAudioTTS
from model_loader import ModelSource
from config.edit_config import get_supported_edit_types
from whisper_wrapper import WhisperWrapper
from stepvocoder.cosyvoice2.cli.cosyvoice import CosyVoice

# Configure logging
logger = logging.getLogger(__name__)

# Global constants for prompt options
PROMPT_WAV_OPTIONS = ["denoise_prompt.wav", "en_happy_prompt.wav", "fear_zh_female_prompt.wav", "paralingustic_prompt.wav", "speed_prompt.wav", "vad_prompt.wav", "whisper_prompt.wav", "zero_shot_en_prompt.wav"]
PROMPT_TEXT_MAP = {"zero_shot_en_prompt.wav": "His political stance was conservative, and he was particularly close to margaret thatcher.", "en_happy_prompt.wav": "You know, I just finished that big project and feel so relieved. Everything seems easier and more colorful, what a wonderful feeling!", "fear_zh_female_prompt.wav": "我总觉得，有人在跟着我，我能听到奇怪的脚步声。", "whisper_prompt.wav": "比如在工作间隙，做一些简单的伸展运动，放松一下身体，这样，会让你更有精力.", "paralingustic_prompt.wav": "我觉得这个计划大概是可行的，不过还需要再仔细考虑一下。", "denoise_prompt.wav": "Such legislation was clarified and extended from time to time thereafter. No, the man was not drunk, he wondered how we got tied up with this stranger. Suddenly, my reflexes had gone. It's healthier to cook without sugar.", "speed_prompt.wav": "上次你说鞋子有点磨脚，我给你买了一双软软的鞋垫。", "vad_prompt.wav": "就是说你比如说我一共在这次看病我一共花了一百块钱，其中呢医生的这个劳动价值占了三十块钱。"}

# Save audio to temporary directory
def save_audio(audio_type, audio_data, sr, tmp_dir):
    """Save audio data to a temporary file with timestamp"""
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(tmp_dir, audio_type, f"{current_time}.wav")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    try:
        if isinstance(audio_data, torch.Tensor):
            torchaudio.save(save_path, audio_data, sr)
        else:
            sf.write(audio_data, sr)
        logger.debug(f"Audio saved to: {save_path}")
        return save_path
    except Exception as e:
        logger.error(f"Failed to save audio: {e}")
        raise


class TokenConversionTab:
    """Tab for WAV to Token and Token to WAV conversion"""

    def __init__(self, audio_tokenizer, cosy_model, args):
        self.audio_tokenizer = audio_tokenizer
        self.cosy_model = cosy_model
        self.args = args
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def _get_prompt_wav_path(self, prompt_wav_filename):
        return os.path.join("examples", prompt_wav_filename)

    def _wav_to_token_str(self, input_wav_path, prompt_wav_filename, prompt_text):
        # prompt_wav_filename and prompt_text are not strictly needed for wav2token but kept for consistent UI signature
        _ = prompt_wav_filename # Suppress unused warning
        _ = prompt_text # Suppress unused warning

        if not input_wav_path:
            return "[Error] Input audio cannot be empty."

        try:
            # Load input WAV
            input_wav, input_sr = torchaudio.load(input_wav_path)
            if input_wav.shape[0] > 1:
                input_wav = input_wav.mean(dim=0, keepdim=True)
            
            # Normalize volume
            norm = torch.max(torch.abs(input_wav), dim=1, keepdim=True)[0]
            if norm > 0.6: 
                input_wav = input_wav / norm * 0.6 

            # Extract tokens
            vq0206_codes, _, _ = self.audio_tokenizer.wav2token(input_wav, input_sr)
            
            # Convert to token string format <audio_N>
            token_string = "".join([f"<audio_{code}>" for code in vq0206_codes])
            return token_string
        except Exception as e:
            self.logger.error(f"WAV to token conversion failed: {e}")
            import traceback
            traceback.print_exc()
            return f"[Error] WAV to token conversion failed: {e}"

    def _token_str_to_wav(self, token_string, prompt_wav_filename, prompt_text):
        # prompt_text is not strictly needed for token2wav but kept for consistent UI signature
        _ = prompt_text # Suppress unused warning

        if not token_string:
            return None, "[Error] Token string cannot be empty."
        if not prompt_wav_filename:
            return None, "[Error] Prompt WAV cannot be empty."
        
        prompt_wav_path = self._get_prompt_wav_path(prompt_wav_filename)

        try:
            # Preprocess prompt WAV (similar to token2wav.py's preprocess_prompt_wav)
            prompt_wav, prompt_wav_sr = torchaudio.load(prompt_wav_path)
            if prompt_wav.shape[0] > 1:
                prompt_wav = prompt_wav.mean(dim=0, keepdim=True)

            norm = torch.max(torch.abs(prompt_wav), dim=1, keepdim=True)[0]
            if norm > 0.6: 
                prompt_wav = prompt_wav / norm * 0.6 

            speech_feat, speech_feat_len = self.cosy_model.frontend.extract_speech_feat(
                prompt_wav, prompt_wav_sr
            )
            speech_embedding = self.cosy_model.frontend.extract_spk_embedding(
                prompt_wav, prompt_wav_sr
            )
            vq0206_codes, _, _ = self.audio_tokenizer.wav2token(prompt_wav, prompt_wav_sr)
            
            # Parse input tokens
            tokens_list = [int(x) for x in re.findall(r"<audio_(\d+)>", token_string)]
            if not tokens_list:
                return None, "[Error] No valid tokens found in input string. Format should be <audio_123>..."
            
            normalized_tokens = []
            for t in tokens_list:
                if t >= 65536: # From token2wav.py, to handle shifted tokens
                    normalized_tokens.append(t - 65536)
                else:
                    normalized_tokens.append(t)
            token_tensor = torch.tensor([normalized_tokens], dtype=torch.long)
            
            prompt_token_tensor = torch.tensor([vq0206_codes], dtype=torch.long)
            prompt_token_tensor = prompt_token_tensor - 65536 # Also shifted

            # Prepare for generation (dtype casting)
            if torch.cuda.is_available():
                speech_feat = speech_feat.to(torch.bfloat16)
                speech_embedding = speech_embedding.to(torch.bfloat16)
            else:
                 speech_feat = speech_feat.to(torch.float32)
                 speech_embedding = speech_embedding.to(torch.float32)

            # Generate audio
            out_wav = self.cosy_model.token2wav_nonstream(
                token_tensor,
                prompt_token_tensor,
                speech_feat,
                speech_embedding,
            )
            
            # Save output WAV temporarily
            output_audio_path = save_audio("token2wav_output", out_wav, 24000, self.args.tmp_dir)
            return output_audio_path, "Success"

        except Exception as e:
            self.logger.error(f"Token string to WAV conversion failed: {e}")
            import traceback
            traceback.print_exc()
            return None, f"[Error] Token string to WAV conversion failed: {e}"

    def register_components(self):
        with gr.Tab("Token Conversion") as self.token_conversion_tab: # Store reference for later event binding
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### WAV to Token String")
                    self.wav_to_token_input_audio = gr.Audio(
                        sources=["upload", "microphone"],
                        type="filepath",
                        label="Input Audio (WAV to Token)",
                    )
                    self.wav_to_token_prompt_wav_dropdown = gr.Dropdown(
                        label="Prompt WAV",
                        choices=PROMPT_WAV_OPTIONS,
                        value=PROMPT_WAV_OPTIONS[0] if PROMPT_WAV_OPTIONS else None
                    )
                    self.wav_to_token_prompt_text_output = gr.Textbox(
                        label="Prompt Text (from selected WAV)",
                        interactive=False
                    )
                    self.wav_to_token_button = gr.Button("Convert WAV to Token String", variant="primary")
                    self.wav_to_token_output_text = gr.Textbox(
                        label="Generated Token String",
                        interactive=True, # Make it editable for user to copy/paste
                        lines=5
                    )
                with gr.Column():
                    gr.Markdown("### Token String to WAV")
                    self.token_to_wav_input_text = gr.Textbox(
                        label="Input Token String",
                        value="<audio_958><audio_101><audio_2216><audio_1028><audio_2892>",
                        lines=5
                    )
                    self.token_to_wav_prompt_wav_dropdown = gr.Dropdown(
                        label="Prompt WAV",
                        choices=PROMPT_WAV_OPTIONS,
                        value=PROMPT_WAV_OPTIONS[0] if PROMPT_WAV_OPTIONS else None
                    )
                    self.token_to_wav_prompt_text_output = gr.Textbox(
                        label="Prompt Text (from selected WAV)",
                        interactive=False
                    )
                    self.token_to_wav_button = gr.Button("Convert Token String to WAV", variant="primary")
                    self.token_to_wav_output_audio = gr.Audio(
                        label="Generated WAV",
                        streaming=False, # Ensure it's not streaming for final output
                        autoplay=False
                    )
                    # Add a Textbox for error messages for Token to WAV
                    self.token_to_wav_error_message = gr.Textbox(
                        label="Error/Status",
                        interactive=False,
                        visible=True
                    )

    def register_events(self):
        # Initialize prompt text for WAV to Token tab
        if self.wav_to_token_prompt_wav_dropdown.value:
            self.wav_to_token_prompt_text_output.value = PROMPT_TEXT_MAP.get(self.wav_to_token_prompt_wav_dropdown.value, "")
        # Initialize prompt text for Token to WAV tab
        if self.token_to_wav_prompt_wav_dropdown.value:
            self.token_to_wav_prompt_text_output.value = PROMPT_TEXT_MAP.get(self.token_to_wav_prompt_wav_dropdown.value, "")

        # Update prompt text when prompt WAV changes for WAV to Token
        self.wav_to_token_prompt_wav_dropdown.change(
            fn=lambda x: PROMPT_TEXT_MAP.get(x, ""),
            inputs=[self.wav_to_token_prompt_wav_dropdown],
            outputs=[self.wav_to_token_prompt_text_output]
        )
        # Update prompt text when prompt WAV changes for Token to WAV
        self.token_to_wav_prompt_wav_dropdown.change(
            fn=lambda x: PROMPT_TEXT_MAP.get(x, ""),
            inputs=[self.token_to_wav_prompt_wav_dropdown],
            outputs=[self.token_to_wav_prompt_text_output]
        )

        # WAV to Token conversion
        self.wav_to_token_button.click(
            self._wav_to_token_str,
            inputs=[
                self.wav_to_token_input_audio,
                self.wav_to_token_prompt_wav_dropdown, # Passed for consistent signature, but not used in logic
                self.wav_to_token_prompt_text_output # Passed for consistent signature, but not used in logic
            ],
            outputs=[self.wav_to_token_output_text]
        )

        # Token to WAV conversion
        self.token_to_wav_button.click(
            self._token_str_to_wav,
            inputs=[
                self.token_to_wav_input_text,
                self.token_to_wav_prompt_wav_dropdown,
                self.token_to_wav_prompt_text_output # Passed for consistent signature, but not used in logic
            ],
            outputs=[self.token_to_wav_output_audio, self.token_to_wav_error_message]
        )


class EditxTab:
    """Audio editing and voice cloning interface tab"""

    def __init__(self, args):
        self.args = args
        self.edit_type_list = list(get_supported_edit_types().keys())
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.enable_auto_transcribe = getattr(args, 'enable_auto_transcribe', False)




    def history_messages_to_show(self, messages):
        """Convert message history to gradio chatbot format"""
        show_msgs = []
        for message in messages:
            edit_type = message['edit_type']
            edit_info = message['edit_info']
            source_text = message['source_text']
            target_text = message['target_text']
            raw_audio_part = message['raw_wave']
            edit_audio_part = message['edit_wave']
            type_str = f"{edit_type}-{edit_info}" if edit_info is not None else f"{edit_type}"
            show_msgs.extend([
                {"role": "user", "content": f"任务类型：{type_str}\n文本：{source_text}"},
                {"role": "user", "content": gr.Audio(value=raw_audio_part, interactive=False)},
                {"role": "assistant", "content": f"输出音频：\n文本：{target_text}"},
                {"role": "assistant", "content": gr.Audio(value=edit_audio_part, interactive=False)}
            ])
        return show_msgs

    def generate_clone(self, prompt_text_input, prompt_audio_input, generated_text, edit_type, edit_info, state):
        """Generate cloned audio"""
        self.logger.info("Starting voice cloning process")
        state['history_audio'] = []
        state['history_messages'] = []

        # Input validation
        if not prompt_text_input or prompt_text_input.strip() == "":
            error_msg = "[Error] Uploaded text cannot be empty."
            self.logger.error(error_msg)
            return [{"role": "user", "content": error_msg}], state
        if not prompt_audio_input:
            error_msg = "[Error] Uploaded audio cannot be empty."
            self.logger.error(error_msg)
            return [{"role": "user", "content": error_msg}], state
        if not generated_text or generated_text.strip() == "":
            error_msg = "[Error] Clone content cannot be empty."
            self.logger.error(error_msg)
            return [{"role": "user", "content": error_msg}], state
        if edit_type != "clone":
            error_msg = "[Error] CLONE button must use clone task."
            self.logger.error(error_msg)
            return [{"role": "user", "content": error_msg}], state

        try:
            # Use common_tts_engine for cloning
            output_audio, output_sr = common_tts_engine.clone(
                prompt_audio_input, prompt_text_input, generated_text
            )

            if output_audio is not None and output_sr is not None:
                # Convert tensor to numpy if needed
                if isinstance(output_audio, torch.Tensor):
                    audio_numpy = output_audio.cpu().numpy().squeeze()
                else:
                    audio_numpy = output_audio

                # Load original audio for comparison
                input_audio_data_numpy, input_sample_rate = librosa.load(prompt_audio_input)

                # Create message for history
                cur_assistant_msg = {
                    "edit_type": edit_type,
                    "edit_info": edit_info,
                    "source_text": prompt_text_input,
                    "target_text": generated_text,
                    "raw_wave": (input_sample_rate, input_audio_data_numpy),
                    "edit_wave": (output_sr, audio_numpy),
                }
                state["history_audio"].append((output_sr, audio_numpy, generated_text))
                state["history_messages"].append(cur_assistant_msg)

                show_msgs = self.history_messages_to_show(state["history_messages"])
                self.logger.info("Voice cloning completed successfully")
                return show_msgs, state
            else:
                error_msg = "[Error] Clone failed"
                self.logger.error(error_msg)
                return [{"role": "user", "content": error_msg}], state

        except Exception as e:
            error_msg = f"[Error] Clone failed: {str(e)}"
            self.logger.error(error_msg)
            return [{"role": "user", "content": error_msg}], state
        
    def generate_edit(self, prompt_text_input, prompt_audio_input, generated_text, edit_type, edit_info, state):
        """Generate edited audio"""
        self.logger.info("Starting audio editing process")

        # Input validation
        if not prompt_audio_input:
            error_msg = "[Error] Uploaded audio cannot be empty."
            self.logger.error(error_msg)
            return [{"role": "user", "content": error_msg}], state

        try:
            # Determine which audio to use
            if len(state["history_audio"]) == 0:
                # First edit - use uploaded audio
                audio_to_edit = prompt_audio_input
                text_to_use = prompt_text_input
                self.logger.debug("Using prompt audio, no history found")
            else:
                # Use previous edited audio - save it to temp file first
                sample_rate, audio_numpy, previous_text = state["history_audio"][-1]
                temp_path = save_audio("temp", audio_numpy, sample_rate, self.args.tmp_dir)
                audio_to_edit = temp_path
                text_to_use = previous_text
                self.logger.debug(f"Using previous audio from history, count: {len(state['history_audio'])}")

            # For para-linguistic, use generated_text; otherwise use source text
            if edit_type not in {"paralinguistic"}:
                generated_text = text_to_use

            # Use common_tts_engine for editing
            output_audio, output_sr = common_tts_engine.edit(
                audio_to_edit, text_to_use, edit_type, edit_info, generated_text
            )

            if output_audio is not None and output_sr is not None:
                # Convert tensor to numpy if needed
                if isinstance(output_audio, torch.Tensor):
                    audio_numpy = output_audio.cpu().numpy().squeeze()
                else:
                    audio_numpy = output_audio

                # Load original audio for comparison
                if len(state["history_audio"]) == 0:
                    input_audio_data_numpy, input_sample_rate = librosa.load(prompt_audio_input)
                else:
                    input_sample_rate, input_audio_data_numpy, _ = state["history_audio"][-1]

                # Create message for history
                cur_assistant_msg = {
                    "edit_type": edit_type,
                    "edit_info": edit_info,
                    "source_text": text_to_use,
                    "target_text": generated_text,
                    "raw_wave": (input_sample_rate, input_audio_data_numpy),
                    "edit_wave": (output_sr, audio_numpy),
                }
                state["history_audio"].append((output_sr, audio_numpy, generated_text))
                state["history_messages"].append(cur_assistant_msg)

                show_msgs = self.history_messages_to_show(state["history_messages"])
                self.logger.info("Audio editing completed successfully")
                return show_msgs, state
            else:
                error_msg = "[Error] Edit failed"
                self.logger.error(error_msg)
                return [{"role": "user", "content": error_msg}], state

        except Exception as e:
            error_msg = f"[Error] Edit failed: {str(e)}"
            self.logger.error(error_msg)
            return [{"role": "user", "content": error_msg}], state

    def clear_history(self, state):
        """Clear conversation history"""
        state["history_messages"] = []
        state["history_audio"] = []
        return [], state

    def init_state(self):
        """Initialize conversation state"""
        return {
            "history_messages": [],
            "history_audio": []
        }

    def register_components(self):
        """Register gradio components - maintaining exact layout from original"""
        with gr.Tab("Editx"):
            with gr.Row():
                with gr.Column():
                    self.model_input = gr.Textbox(label="Model Name", value="Step-Audio-EditX", scale=1)
                    self.prompt_text_input = gr.Textbox(label="Prompt Text", value="", scale=1)
                    self.prompt_audio_input = gr.Audio(
                        sources=["upload", "microphone"],
                        format="wav",
                        type="filepath",
                        label="Input Audio",
                    )
                    self.generated_text = gr.Textbox(label="Target Text", lines=1, max_lines=200, max_length=1000)
                with gr.Column():
                    with gr.Row():
                        self.edit_type = gr.Dropdown(label="Task", choices=self.edit_type_list, value="clone")
                        self.edit_info = gr.Dropdown(label="Sub-task", choices=[], value=None)
                    self.chat_box = gr.Chatbot(label="History", type="messages", height=480*1)
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        self.button_tts = gr.Button("CLONE", variant="primary")
                        self.button_edit = gr.Button("EDIT", variant="primary")
                with gr.Column():
                    self.clean_history_submit = gr.Button("Clear History", variant="primary")

            gr.Markdown("---")
            gr.Markdown("""
                **Button Description:**
                - CLONE: Synthesizes audio based on uploaded audio and text, only used for clone mode, will clear history information when used.
                - EDIT: Edits based on uploaded audio, or continues to stack edit effects based on the previous round of generated audio.
                """)
            gr.Markdown("""
                **Operation Workflow:**
                - Upload the audio to be edited on the left side and fill in the corresponding text content of the audio;
                - If the task requires modifying text content (such as clone, para-linguistic), fill in the text to be synthesized in the "target text" field. For all other tasks, keep the uploaded audio text content unchanged;
                - Select tasks and subtasks on the right side (some tasks have no subtasks, such as vad, etc.);
                - Click the "CLONE" or "EDIT" button on the left side, and audio will be generated in the dialog box on the right side.
                """)
            gr.Markdown("""
                **Para-linguistic Description:**
                - Supported tags include: [Breathing] [Laughter] [Surprise-oh] [Confirmation-en] [Uhm] [Surprise-ah] [Surprise-wa] [Sigh] [Question-ei] [Dissatisfaction-hnn]
                - Example:
                    - Fill in "target text" field: "Great, the weather is so nice today." Click the "CLONE" button to get audio.
                    - Change "target text" field to: "Great[Laughter], the weather is so nice today[Surprise-ah]." Click the "EDIT" button to get para-linguistic audio.
                """)

    def register_events(self):
        """Register event handlers"""
        # Create independent state for each session
        state = gr.State(self.init_state())

        self.button_tts.click(self.generate_clone,
            inputs=[self.prompt_text_input, self.prompt_audio_input, self.generated_text, self.edit_type, self.edit_info, state],
            outputs=[self.chat_box, state])
        self.button_edit.click(self.generate_edit,
            inputs=[self.prompt_text_input, self.prompt_audio_input, self.generated_text, self.edit_type, self.edit_info, state],
            outputs=[self.chat_box, state])

        self.clean_history_submit.click(self.clear_history, inputs=[state], outputs=[self.chat_box, state])
        self.edit_type.change(
            fn=self.update_edit_info,
            inputs=self.edit_type,
            outputs=self.edit_info,
        )

        # Add audio transcription event only if enabled
        if self.enable_auto_transcribe:
            self.prompt_audio_input.change(
                fn=self.transcribe_audio,
                inputs=[self.prompt_audio_input, self.prompt_text_input],
                outputs=self.prompt_text_input,
            )

    def update_edit_info(self, category):
        """Update sub-task dropdown based on main task selection"""
        category_items = get_supported_edit_types()
        choices = category_items.get(category, [])
        value = None if len(choices) == 0 else choices[0]
        return gr.Dropdown(label="Sub-task", choices=choices, value=value)

    def transcribe_audio(self, audio_input, current_text):
        """Transcribe audio using Whisper ASR when prompt text is empty"""
        # Only transcribe if current text is empty
        if current_text and current_text.strip():
            return current_text  # Keep existing text
        if not audio_input:
            return ""  # No audio to transcribe
        if whisper_asr is None:
            self.logger.error("Whisper ASR not initialized.")
            return ""

        try:
            # Transcribe audio
            transcribed_text = whisper_asr(audio_input)
            self.logger.info(f"Audio transcribed: {transcribed_text}")
            return transcribed_text

        except Exception as e:
            self.logger.error(f"Failed to transcribe audio: {e}")
            return ""


def launch_demo(args, editx_tab):
    """Launch the gradio demo"""
    with gr.Blocks(
            theme=gr.themes.Soft(), 
            title="🎙️ Step-Audio-EditX",
            css="""
    :root {
        --font: "Helvetica Neue", Helvetica, Arial, sans-serif;
        --font-mono: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    }
    """) as demo:
        gr.Markdown("## 🎙️ Step-Audio-EditX")
        gr.Markdown("Audio Editing and Zero-Shot Cloning using Step-Audio-EditX")

        # Instantiate TokenConversionTab
        token_conversion_tab = TokenConversionTab(common_audio_tokenizer, common_cosy_model, args)

        with gr.Tabs(): # Wrap tabs here
            # Register components for both tabs
            editx_tab.register_components()
            token_conversion_tab.register_components() # Register new tab components here

        # Register events for both tabs
        editx_tab.register_events()
        token_conversion_tab.register_events() # Register new tab events here

    # Launch demo
    demo.queue().launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share if hasattr(args, 'share') else False
    )


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Step-Audio Edit Demo")
    parser.add_argument("--model-path", type=str, default=os.path.join(os.path.expanduser("~"), ".cache", "StepAudioEditX"), help="Local path to model data and tokenizer assets. Models will be downloaded here if not found.")
    parser.add_argument("--server-name", type=str, default="0.0.0.0", help="Demo server name.")
    parser.add_argument("--server-port", type=int, default=7860, help="Demo server port.")
    parser.add_argument("--tmp-dir", type=str, default="/tmp/gradio", help="Save path.")
    parser.add_argument("--share", action="store_true", help="Share gradio app.")

    # Multi-source loading support parameters
    parser.add_argument(
        "--model-source",
        type=str,
        default="auto",
        choices=["auto", "local", "modelscope", "huggingface"],
        help="Model source: auto (detect automatically), local, modelscope, or huggingface"
    )
    parser.add_argument(
        "--tokenizer-model-id",
        type=str,
        default="dengcunqin/speech_paraformer-large_asr_nat-zh-cantonese-en-16k-vocab8501-online",
        help="FunASR model ID for the tokenizer."
    )
    parser.add_argument(
        "--audio-tokenizer-repo",
        type=str,
        default="stepfun-ai/Step-Audio-Tokenizer",
        help="Hugging Face repo for the Step-Audio-Tokenizer."
    )
    parser.add_argument(
        "--tts-model-id",
        type=str,
        default="https://huggingface.co/stepfun-ai/Step-Audio-EditX", # Default to HF model
        help="TTS model ID for online loading (if different from model-path)"
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=["int4", "int8", "awq-4bit"],
        help="Enable quantization for the TTS model to reduce memory usage."
             "Choices: int4 (online), int8 (online), awq-4bit (AWQ 4-bit quantization)."
             "When quantization is enabled, data types are handled automatically by the quantization library."
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="PyTorch data type for model operations. This setting only applies when quantization is disabled. "
             "When quantization is enabled, data types are managed automatically."
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="cuda",
        help="Device mapping for model loading (default: cuda)"
    )
    parser.add_argument(
        "--enable-auto-transcribe",
        action="store_true",
        help="Enable automatic audio transcription when uploading audio files (default: disabled)"
    )

    args = parser.parse_args()

    # If a HuggingFace model ID is provided as a URL, download it to the model_path directory.
    if args.tts_model_id and "huggingface.co" in args.tts_model_id:
        repo_id = args.tts_model_id.replace("https://huggingface.co/", "")
        logger.info(f"Downloading model {repo_id} to {args.model_path}...")
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=args.model_path,
                local_dir_use_symlinks=False, # Use direct copies
                resume_download=True
            )
            logger.info("✓ Model downloaded successfully.")
        except Exception as e:
            logger.error(f"❌ Failed to download model from Hugging Face: {e}")
            exit(1)

    # Download tokenizer model from its own repo if specified
    if args.audio_tokenizer_repo:
        tokenizer_local_path = os.path.join(args.model_path, "Step-Audio-Tokenizer")
        logger.info(f"Downloading audio tokenizer {args.audio_tokenizer_repo} to {tokenizer_local_path}...")
        try:
            snapshot_download(
                repo_id=args.audio_tokenizer_repo,
                local_dir=tokenizer_local_path,
                local_dir_use_symlinks=False,
                resume_download=True
            )
            logger.info("✓ Audio tokenizer downloaded successfully.")
        except Exception as e:
            logger.error(f"❌ Failed to download audio tokenizer: {e}")
            exit(1)

    # Map string arguments to actual types
    source_mapping = {
        "auto": ModelSource.AUTO,
        "local": ModelSource.LOCAL,
        "modelscope": ModelSource.MODELSCOPE,
        "huggingface": ModelSource.HUGGINGFACE
    }
    model_source = source_mapping[args.model_source]

    # Map torch dtype string to actual torch dtype
    dtype_mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }
    torch_dtype = dtype_mapping[args.torch_dtype]

    logger.info(f"Loading models from local path: {args.model_path}")
    logger.info(f"Tokenizer model ID: {args.tokenizer_model_id}")
    logger.info(f"Torch dtype: {args.torch_dtype}")
    logger.info(f"Device map: {args.device_map}")
    if args.tts_model_id:
        logger.info(f"TTS model ID (source repo): {args.tts_model_id}")
    if args.quantization:
        logger.info(f"🔧 {args.quantization.upper()} quantization enabled")

    # Initialize models
    global common_audio_tokenizer, common_tts_engine, common_cosy_model
    try:
        # Load StepAudioTokenizer from the local model_path
        common_audio_tokenizer = StepAudioTokenizer(
            os.path.join(args.model_path, "Step-Audio-Tokenizer"),
            funasr_model_id=args.tokenizer_model_id
        )
        logger.info("✓ StepAudioTokenizer loaded successfully")
        
        # Initialize common TTS engine directly from the local model_path
        tts_model_path_local = os.path.join(args.model_path, "Step-Audio-EditX-AWQ-4bit" if args.quantization == "awq-4bit" else "Step-Audio-EditX")
        common_tts_engine = StepAudioTTS(
            tts_model_path_local,
            common_audio_tokenizer, # Pass the global tokenizer
            model_source=ModelSource.LOCAL, # Always load from local after downloading
            tts_model_id=None, # tts_model_id is not needed as we are loading locally
            quantization_config=args.quantization,
            torch_dtype=torch_dtype,
            device_map=args.device_map
        )
        common_cosy_model = common_tts_engine.cosy_model
        logger.info("✓ StepCommonAudioTTS loaded successfully")
        
        if args.enable_auto_transcribe:
            whisper_asr = WhisperWrapper()
            logger.info("✓ Automatic audio transcription enabled")
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        logger.error("Please check your model paths and source configuration.")
        exit(1)
