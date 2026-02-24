# 在 macOS / Apple Silicon 上用 MPS 跑 `clone`（经验总结）

本文总结本仓库在 **TTS 零样本声音克隆（`clone`）** 场景下，使用 **PyTorch MPS（Apple Silicon GPU）** 的可用路径、组件分工、常见坑与排查点。

## TL;DR（推荐组合）

- **LLM 推理**：用 MPS（`--device-map mps`）
- **声码器（CosyVoice）**：自动走 MPS（cuda > mps > cpu）
- **音频 tokenizer**
  - `vq02`（FunASR streaming encoder）：默认 CPU（更稳）
  - `vq06`（ONNXRuntime）：默认 CPU（`CPUExecutionProvider`）

> 结论：在 Mac 上，通常是 **LLM + vocoder 用 MPS**，**tokenizer/ASR 走 CPU**。如果觉得慢，优先从“音频长度、tokenizer 耗时”入手，而不是强行把所有东西搬到 MPS。

---

## 1) 本仓库的“设备分工”现状（按代码实现）

### 1.1 `StepAudioTTS`（LLM + 调度）

- 入口：`tts.py` 的 `StepAudioTTS.__init__`
- 逻辑：
  - `device_map` 决定 **Transformers 模型加载的 device_map**（见 `model_loader.py -> load_transformers_model(...)`）
  - 同时 `StepAudioTTS` 会根据 `device_map` 推断一个 `self.device`，用于把显式创建的张量（`input_ids`、`vq0206_codes_vocoder` 等）放到正确设备上，避免 *device mismatch*。

### 1.2 `CosyVoice`（vocoder）

- 入口：`stepvocoder/cosyvoice2/cli/cosyvoice.py` 的 `CosyVoice.__init__`
- 逻辑：自动选择 **cuda > mps > cpu**，并把内部 `cosy_impl` 放到该设备。

### 1.3 `StepAudioTokenizer`（参考音频→离散 token）

- 入口：`tokenizer.py` 的 `StepAudioTokenizer.__init__`
- 现状：
  - `vq02`：调用 FunASR streaming encoder（`infer_encoder`），默认 `funasr_device="cpu"`；也支持 `funasr_device="auto"` 尝试在 MPS 可用时切到 `mps`，但不保证稳定（见 `funasr_detach/auto/auto_model.py` 的注释：FunASR 对 mps 支持不佳）。
  - `vq06`：ONNXRuntime 跑 `speech_tokenizer_v1.onnx`，默认 provider 为 `CPUExecutionProvider`（CPU）。

---

## 2) 在 MPS 上跑 `clone` 的推荐命令

### 2.1 直接跑 `tts_infer.py`

```bash
python3 tts_infer.py \
  --model-path where_you_download_dir \
  --prompt-text "（参考音频对应文本）" \
  --prompt-audio-path examples/zero_shot_en_prompt.wav \
  --generated-text "（要合成的目标文本）" \
  --edit-type clone \
  --output-dir ./output/mps \
  --device-map mps \
  --torch-dtype float16
```

说明：
- `--device-map mps`：让 LLM 推理走 MPS（并让 `StepAudioTTS` 的显式张量走同一设备）。
- `--torch-dtype float16`：MPS 上更通用；如果遇到 dtype 报错（尤其是 `bfloat16`），优先改成 `float16` 或 `float32`。

### 2.2 用一键脚本（零样本克隆示例）

```bash
python3 scripts/clone_paralingustic_prompt.py \
  --model-path where_you_download_dir \
  --device-map mps \
  --torch-dtype float16 \
  --output-dir ./output/mps
```

---

## 3) 常见问题与排查要点

### 3.1 “Expected all tensors to be on the same device”

典型原因是：**模型在某个设备，但输入张量/中间张量在另一个设备**。

本仓库已经做过的修复点（便于你二次改动时对照）：
- `tts.py`：`llm.generate(...)` 的输入 `input_ids`、`vq0206_codes_vocoder`、`speech_feat`、`speech_embedding` 都会显式 `.to(self.device)`，避免硬编码 `"cuda"`。

如果你自己新增了张量（比如做额外拼接、后处理），也要确保跟 `self.device` 对齐。

### 3.2 `bfloat16` / dtype 不支持或报错

排查顺序：
1) 先把运行参数改为 `--torch-dtype float16`
2) 仍不行再用 `--torch-dtype float32`（更稳但更慢/更占内存）

### 3.3 量化在 MPS 上“不工作/装不上”

仓库支持 `--quantization int4/int8/awq-4bit`，但：
- `int4/int8` 走 bitsandbytes 路线时通常更偏 CUDA 场景；
- 在 macOS/MPS 上，建议先 **不用量化**，跑通 `float16/float32` 再考虑别的方案。

### 3.4 速度慢：瓶颈多半在 tokenizer/ASR（CPU）

`vq02` 的 FunASR streaming encoder 和 `vq06` 的 ONNXRuntime 推理默认都在 CPU。仓库内的 `output/profile_wav2token*.txt` 也能看到 `infer_encoder` 占比很高。

经验上更有效的优化手段：
- **控制参考音频长度**（README 里也建议 < 30s）
- 如果是批量任务，考虑把“参考音频 → token”做缓存（同一参考音频复用）
- 尝试 `STEPAUDIO_TORCH_NUM_THREADS=...` 控制 CPU 线程数（见 `tokenizer.py`）

---

## 4) 关键实现位置索引（方便回看/二次改）

- `tts.py`：`StepAudioTTS` 的 `self.device` 推断与 `clone/edit` 路径的显式 `.to(self.device)`
- `model_loader.py`：Transformers 模型加载参数（`device_map`、量化、`torch_dtype`）
- `stepvocoder/cosyvoice2/cli/cosyvoice.py`：vocoder 设备自动选择（cuda > mps > cpu）
- `tokenizer.py`：`funasr_device` 的选择与 vq02/vq06 的实际执行路径
- `funasr_detach/auto/auto_model.py` / `funasr_detach/frontends/wav_frontend.py`：移除硬编码 `.cuda()` 后的兼容逻辑（cuda 可用则用 cuda，否则 CPU）

