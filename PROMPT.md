# Prompt 结构说明（面向 `clone` / 声音克隆）

这份文档解释本仓库在 **声音克隆（clone / zero-shot TTS）** 时，如何把「参考音频的离散 audio token」和「文本」一起**拼成 LLM 的输入 prompt**。核心点：**audio token 是离散的 token**，在 prompt 里会被序列化成形如 `<audio_123>` 的“特殊 token 字符串”，再像拼字符串一样直接拼接到 system prompt 中。

相关实现文件：
- `config/prompts.py`：`AUDIO_EDIT_CLONE_SYSTEM_PROMPT_TPL`
- `tokenizer.py`：`StepAudioTokenizer.merge_vq0206_to_token_str`
- `tts.py`：`StepAudioTTS.clone`、`StepAudioTTS._encode_audio_edit_clone_prompt`

---

## 1) 参考音频 → 离散 audio token → “字符串化”

### 1.1 两路 VQ code
`StepAudioTokenizer.wav2token()` 会从参考音频提取两路离散码（两个 codebook）：
- `vq02`：1024 类（常见取值范围 `0~1023`）
- `vq06`：4096 类（常见取值范围 `0~4095`）

在拼接前会对 `vq06` 做 **+1024** 偏移，避免和 `vq02` 的 token id 空间冲突（见 `merge_vq0206_to_token_str`）。

### 1.2 2:3 交织（merge）
`merge_vq0206_to_token_str(vq02, vq06)` 的逻辑是：每个 chunk 取
- `vq02[i:i+2]` 两个
- `vq06[j:j+3]` 三个（并先整体 +1024）

交织后的 token id 序列可以用伪代码表示为：

```python
vq06_shift = [1024 + x for x in vq06]
merged = []
i = j = 0
while i < len(vq02) - 1 and j < len(vq06_shift) - 2:
    merged += vq02[i:i+2] + vq06_shift[j:j+3]
    i += 2
    j += 3
```

### 1.3 “像字符串一样拼”：`<audio_*>` 直接无分隔拼接
最后把 `merged` 序列**序列化成字符串**：

```python
audio_token_str = "".join([f"<audio_{x}>" for x in merged])
```

注意这里是 **`"" .join(...)`**：**没有空格、没有逗号**，纯粹把一个个 `<audio_*>` token 当作字符串片段直接拼起来。

原因：这些 `<audio_*>` 在 Step-Audio 的 tokenizer 词表里是“整体 token”（不是按字符切），因此写成连续的 `<audio_1><audio_2>...` 依然会被编码成离散 token 序列。

---

## 2) clone 的 system prompt：把参考信息塞进 system 角色

### 2.1 模板
`config/prompts.py` 里的 `AUDIO_EDIT_CLONE_SYSTEM_PROMPT_TPL`：

```text
Generate audio with the following timbre, prosody and speaking style

[speaker_start]
speaker name: {speaker}
speaker prompt text: 
{prompt_text}
speaker audio tokens: 
{prompt_wav_tokens}
[speaker_end]
```

### 2.2 填充字段
在 `tts.py -> StepAudioTTS.clone()` 里：
- `speaker`：由 `generate_clone_voice_id(prompt_text, prompt_wav)` 生成（参考文本 + 参考音频片段 hash）
- `prompt_text`：参考音频对应的文本（用户输入/ASR）
- `prompt_wav_tokens`：上一节得到的 `<audio_*>` 拼接字符串

也就是说，clone 的“条件信息”（声音特征）**不是放在 human turn**，而是放在 **system prompt** 里，让模型把它当作全局约束。

---

## 3) 最终送进 LLM 的“对话式 prompt”是怎么拼的

在 `tts.py -> StepAudioTTS._encode_audio_edit_clone_prompt()` 里，最终输入给 LLM 的结构是：

1) 一个 `system` turn，内容是上面模板填充后的字符串  
2) 一个 `human` turn，内容只有你要合成的 `target_text`（目标文本）  
3) 追加一个空的 `assistant` 头，让 `generate()` 从这里开始续写（续写的就是 audio token）

如果只看“字符串层面”的语义，等价于：

```text
system
{AUDIO_EDIT_CLONE_SYSTEM_PROMPT_TPL.format(...)}

human
{target_text}

assistant
```

实现层面会把这些字符串分别 `tokenizer.encode(...)`，并用若干特殊 id 包一层“对话边界”（代码里是 `history = [1]` 起始，然后在每段前后插入 `[4] ... [3]` 之类的边界 token）。核心仍然是：**先把文本（包含 `<audio_*>`）按字符串拼好，再交给 tokenizer 编成离散 token 序列**。

---

## 4) 一句话总结

clone prompt 的拼接思路就是：
- 参考音频 → 离散码 → `"<audio_...><audio_...>..."` 这种“无分隔字符串”
- 把这串 audio token 放进 system 模板
- human 只放目标文本
- 让 LLM 在 assistant 位置续写（输出离散 audio token）
