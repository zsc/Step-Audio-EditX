# 本次会话经验教训（TTS / clone / MPS）

## 0) 推荐“正向使用流程”（Mac / Apple Silicon + MPS + `clone`）

> 目标：在 macOS 上尽量用 **MPS** 跑 LLM/vocoder，tokenizer 走 **CPU**，先跑通再优化。

### 0.1 准备模型目录（本地推理）

确保 `--model-path` 对应目录结构如下（README 同款）：

```text
where_you_download_dir
├── Step-Audio-Tokenizer
└── Step-Audio-EditX
```

### 0.2 跑一个最小 `clone`（推荐参数）

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

预期：
- `./output/mps/` 下生成 `*_cloned.wav`
- 如果 MPS 不可用，会自动/间接回退到 CPU（速度会明显下降）

### 0.3 用脚本快速复现（零样本克隆 demo）

```bash
python3 scripts/clone_paralingustic_prompt.py \
  --model-path where_you_download_dir \
  --output-dir ./output/mps \
  --device-map mps \
  --torch-dtype float16
```

### 0.4 “跑不通”时的最小化处理顺序

1) dtype 报错：先换 `--torch-dtype float16`，还不行再 `float32`
2) device mismatch：确认 `--device-map mps`，以及代码里显式创建的 tensor 有没有 `.to(device)`
3) 太慢：优先缩短参考音频（<30s），或对“参考音频→token”做缓存（批量任务）

### 0.5 批量 `wav2token`（输出 JSONL）

任务：把一堆 wav 转成 `<audio_...>` token 字符串（与 `PROMPT.md` 的格式一致：vq06 已做 +1024，**不**额外 +65536）。

```bash
python3 scripts/batch_wav2token.py \
  --model-path where_you_download_dir \
  --input-dir /path/to/wavs \
  --output-jsonl ./output/wav2token.jsonl
```

输出（每行一个 JSON）：`wav/sr/duration_sec/tokens`；失败会写到 `./output/wav2token.jsonl.errors.jsonl`。

### 0.6 批量 `clone`（一次加载，多条任务）

先准备一个 tasks JSONL（每行一个任务）：

```jsonl
{"id":"0001","prompt_audio_path":"examples/zero_shot_en_prompt.wav","prompt_text":"...","generated_text":"..."}
{"id":"0002","prompt_audio_path":"/abs/path/prompt.wav","prompt_text":"...","generated_text":"...","output_wav":"custom/name.wav"}
```

运行：

```bash
python3 scripts/batch_clone.py \
  --model-path where_you_download_dir \
  --tasks ./tasks.jsonl \
  --output-dir ./output/clone_batch \
  --device-map mps \
  --torch-dtype float16 \
  --skip-existing
```

## 1) 先确认“目录/入口”再写总结

- 用户最初说的是“本目录 `tts/clone/mps`”，但仓库里并不存在 `tts/` 目录；真正入口是 `tts_infer.py` / `tts.py` / `tokenizer.py` / `stepvocoder/...`。
- 经验：**先用 `ls`/`find`/`rg` 把“名字对上代码”**，再开始写文档，不然容易总结到不存在的路径或过时的使用方式。

## 2) 查 MPS 相关，优先 `rg` 全仓 + 读关键类

本次定位 MPS/clone 的路线：

- `rg "\\bmps\\b" -S .`：快速定位所有涉及 mps 的代码点
- 重点文件：
  - `tts.py`：`StepAudioTTS` 的 `device_map` → `self.device` 推断、以及显式张量 `.to(self.device)`（避免硬编码 `"cuda"`）
  - `tokenizer.py`：tokenizer（vq02/vq06）设备与 provider 选择、CPU/MPS 的现实取舍
  - `stepvocoder/cosyvoice2/cli/cosyvoice.py`：vocoder 自动设备选择（cuda > mps > cpu）
  - `funasr_detach/...`：把硬编码 `.cuda()` 改成“有 cuda 才用 cuda，否则 CPU”，避免无 CUDA 环境直接炸

经验：**不要只改“模型加载 device_map”**；还要检查你代码里“手工创建的 tensor/feat/embedding”是否跟模型在同一 device。

## 3) MPS 跑通的现实策略：LLM/Vocoder 用 MPS，Tokenizer 多半 CPU

- LLM 推理：`--device-map mps` 是最核心的切换点
- dtype：在 MPS 上，`float16` 通常比 `bfloat16` 更稳（遇到 dtype 报错先改 `--torch-dtype float16`，不行再 `float32`）
- tokenizer：
  - `vq02`（FunASR streaming encoder）对 MPS 支持不稳定，默认 CPU 更稳
  - `vq06`（ONNXRuntime）默认 `CPUExecutionProvider`，可先保证可用性

经验：**把“能加速的部分”放到 MPS**，不要为了一致性强行把所有组件搬上去（容易踩坑且收益不一定大）。

## 4) 性能瓶颈要用数据说话（profiling / 实测）

仓库里已有 `output/profile_wav2token*.txt` 这类数据，能直观看到 tokenizer/ASR 占用。

经验：
- 先控制输入规模（参考音频时长）再谈优化
- 批处理场景优先做缓存（同一参考音频复用 token）
- CPU 侧可以用 `STEPAUDIO_TORCH_NUM_THREADS=...` 做线程数试探

## 5) 文档/脚本要“可复现、可直接跑”

本次整理并提交了：
- `TTS_CLONE_MPS.md`：面向 MPS+clone 的运行建议、组件分工、常见坑
- `PROMPT.md`：解释 clone prompt 结构（audio token 如何拼入 system prompt）
- `scripts/clone_paralingustic_prompt.py`：把一次常用 clone 流程封装成脚本，减少手敲命令成本

经验：**总结不只是写说明**，最好配一个“能跑通”的脚本或最小命令行示例。

## 6) Git 提交纪律：只提交该提交的

- 仓库里有大体积模型目录（`Step-Audio-EditX/`、`Step-Audio-Tokenizer/`）以及各种本地产物（`output/`、pip 日志等），不适合进 git。
- 操作要点：
  - `git status --porcelain` 先看清“哪些是修改、哪些是未跟踪”
  - `git add <明确的文件>` 精确暂存（不要 `git add .`）
  - 文档/脚本单独提交，方便回滚与审阅
  - 如需长期保持工作区干净，再考虑补 `.gitignore`

## 7) 仍待决的技术债（提醒）

本次会话里 `tokenizer.py` 也做了兼容/性能取舍（FunASR 输入、`funasr_device`、ONNX provider、线程数等），已经以独立 commit 落到主线。

经验：
- **docs/脚本** 和 **功能改动** 分开提交，review/回滚成本最低
- 大体积模型目录（`Step-Audio-EditX/`、`Step-Audio-Tokenizer/`）保持未跟踪/ignore，避免误提交
