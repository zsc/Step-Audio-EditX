# 本次会话经验教训（TTS / clone / MPS）

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

- `tokenizer.py` 目前有未提交改动：涉及 FunASR 输入形式、设备/线程、ONNX provider 默认值等；要不要纳入主线需要你决定（偏“功能/兼容性”而非“文档”范畴）。
- 经验：**docs 提交不要混入大改动**；大改动另起 commit（甚至另起 PR）更好 review。

