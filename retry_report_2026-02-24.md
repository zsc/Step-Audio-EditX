# Step-Audio-EditX GPU 实测报告（更新）

## 环境
- 日期：2026-02-24
- GPU：NVIDIA GeForce RTX 2080 SUPER（7.60 GiB）
- Torch：`2.10.0+cu128`
- Proxy：`no_proxy='*'`

## 已完成代码改动
### 1) 关键性能与稳定性修复
- `tts.py`
  - 新增 `--cosy-device` 透传参数到 `StepAudioTTS(..., cosy_device=...)`
  - 新增 `max_length` 参数，支持用 CLI 控制 LLM 生成长度
  - `CosyVoice` 改为按 `cosy_device` 初始化，CPU 下默认用 `torch.float32`
  - `token2wav_nonstream` 输入张量按 `cosy_device` 组装，避免 GPU/CPU 混用
- `tts_infer.py`
  - 新增 `--max-length`
  - 新增 `--cosy-device`（传递到 TTS 引擎）
- `app.py`
  - 同步新增 `--max-length` 和 `--cosy-device`
- `tokenizer.py` + `funasr_detach/auto/auto_model.py`
  - 让 FunASR 按 `funasr-device` 显式落到目标设备，避免每次推理默认 `.cuda()`

### 2) 运行结果
#### 成功配置（推荐）
```bash
no_proxy='*' python tts_infer.py \
  --model-path /mnt/sda2/Step-Audio-EditX \
  --prompt-text 'His political stance was conservative, and he was particularly close to margaret thatcher.' \
  --prompt-audio-path examples/zero_shot_en_prompt.wav \
  --edit-type clone \
  --generated-text 'Short benchmark sentence to measure runtime path and avoid OOM.' \
  --device-map auto --cosy-device cpu --funasr-device cpu \
  --torch-dtype float16 --max-length 512
```
- 成功产出：`./output_dir/zero_shot_en_prompt_cloned.wav`
- 执行时间：`~29.14s`
- 说明：LLM + 预处理 + vocoder 全链路可跑通。

#### 失败配置（对比）
```bash
--max-length 8192  # 使用默认 8192 时
```
- 报错：`CUDA out of memory`（在 8.6~7.1 GiB 使用量下，8GB 卡不足）

### 3) GPU 利用率（`nvidia-smi` 采样）
- 总体窗口（含模型加载+推理）：
  - `avg gpu`：约 `28.3%`
  - `p95 gpu`：约 `92%`
  - `max gpu`：`93%`
  - 峰值显存：`7613 MiB`
- 业务阶段（`Starting voice cloning process` 到保存音频）：
  - `avg gpu`：约 `45%`
  - `max gpu`：`93%`
  - 持续高负载区间仅约 1~5 秒（模型加载与 I/O 期占用低）

## 结论与建议
1. 你看到的“GPU 效率低”主要是平均值稀释效应：
   - 模型加载阶段大多是内存分配/IO，GPU 利用率接近 0。
   - 真正计算高峰只出现在 token 生成与后续 vocoder 阶段。
2. 8GB 卡上建议默认不要用 `--max-length 8192`，改成 `256/512/1024` 做分段生成。
3. `--funasr-device cpu` + `--cosy-device cpu` 是当前卡上最稳配置，防止 LLM、funasr、vocoder 抢同一块 GPU。
4. 如果还希望再提速：
   - 尽量一次性输入更长文本（一次性触发更长 `decode`）；
   - 缓存已编译模型和预处理结果，复用进程内模型实例，避免反复重复加载。
