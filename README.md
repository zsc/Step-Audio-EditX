# Step-Audio-EditX
<p align="center">
  <img src="assets/logo.png"  height=100>
</p>

<div align="center">
    <a href="https://stepaudiollm.github.io/step-audio-editx/"><img src="https://img.shields.io/static/v1?label=Demo%20Page&message=Web&color=green"></a> &ensp;
  <a href="https://arxiv.org/abs/2511.03601"><img src="https://img.shields.io/static/v1?label=Tech%20Report&message=Arxiv&color=red"></a> &ensp;
  <a href="https://huggingface.co/stepfun-ai/Step-Audio-EditX"><img src="https://img.shields.io/static/v1?label=Step-Audio-EditX&message=HuggingFace&color=yellow"></a> &ensp;
    <a href="https://modelscope.cn/models/stepfun-ai/Step-Audio-EditX"><img src="https://img.shields.io/static/v1?label=Step-Audio-EditX&message=ModelScope&color=blue"></a> &ensp;
  <a href="https://huggingface.co/spaces/stepfun-ai/Step-Audio-EditX"><img src="https://img.shields.io/static/v1?label=Space%20Playground&message=HuggingFace&color=yellow"></a> &ensp;
</div>

## 🔥🔥🔥 最新动态！！！
* 2025年11月28日: 🚀 新模型发布：现已支持**日语**和**韩语**。
* 2025年11月23日: 📊 [Step-Audio-Edit-Benchmark](https://github.com/stepfun-ai/Step-Audio-Edit-Benchmark) 已发布！
* 2025年11月19日: ⚙️ 我们发布了模型的**新版本**，**支持多音字发音控制**，并提升了情感、说话风格和副语言编辑的性能。
* 2025年11月12日: 📦 我们发布了 **Step-Audio-EditX** ([HuggingFace](https://huggingface.co/stepfun-ai/Step-Audio-EditX); [ModelScope](https://modelscope.cn/models/stepfun-ai/Step-Audio-EditX)) 和 **Step-Audio-Tokenizer**([HuggingFace](https://huggingface.co/stepfun-ai/Step-Audio-Tokenizer); [ModelScope](https://modelscope.cn/models/stepfun-ai/Step-Audio-Tokenizer)) 的**优化推理代码**和**模型权重**
* 2025年11月07日: ✨ [Demo 页面](https://stepaudiollm.github.io/step-audio-editx/) ; 🎮 [HF Space 体验区](https://huggingface.co/spaces/stepfun-ai/Step-Audio-EditX)
* 2025年11月06日: 👋 我们发布了 [Step-Audio-EditX](https://arxiv.org/abs/2511.03601) 的技术报告。

## 简介
我们开源了 Step-Audio-EditX，这是一个强大的基于**30亿参数**大语言模型的**强化学习**音频模型，专门用于富有表现力和迭代式音频编辑。它在情感、说话风格和副语言编辑方面表现出色，同时还具备强大的零样本文本转语音（TTS）能力。

## 📑 开源计划
- [x] 推理代码
- [x] 在线演示（Gradio）
- [x] Step-Audio-Edit-Benchmark
- [x] 模型检查点
  - [x] Step-Audio-Tokenizer
  - [x] Step-Audio-EditX
  - [ ] Step-Audio-EditX-Int4
- [ ] 训练代码
  - [ ] SFT 训练
  - [ ] PPO 训练
- [ ] ⏳ 功能支持计划
  - [ ] 编辑功能
    - [x] 多音字发音控制
    - [ ] 更多副语言标签（[咳嗽、哭泣、重音等]）
    - [ ] 填充词移除
  - [ ] 其他语言
    - [x] 日语、韩语
    - [ ] 阿拉伯语、法语、俄语、西班牙语等

## 功能特性
- **零样本 TTS**
  - 在普通话、英语、四川话和粤语方面具有出色的零样本 TTS 克隆效果。
  - 要使用方言或其他语言，只需在文本前添加 **`[Sichuanese]`** / **`[Cantonese]`** / **`[Japanese]`** / **`[Korean]`** 标签。
  - 🔥 多音字发音控制，只需将多音字替换为拼音即可。
    - **[我也想过过过儿过过的生活]** -> **[我也想guo4guo4guo1儿guo4guo4的生活]**

- **情感和说话风格编辑**
  - 对情感和风格进行非常有效的迭代控制，支持**数十种**编辑选项。
    - 情感编辑：[ *愤怒*、*快乐*、*悲伤*、*兴奋*、*恐惧*、*惊讶*、*厌恶* 等 ]
    - 说话风格编辑：[ *撒娇*、*老年*、*儿童*、*耳语*、*严肃*、*豪爽*、*夸张* 等]
    - 更多情感和说话风格编辑功能正在开发中，敬请期待！🚀

- **副语言编辑**
  - 精确控制 10 种类型的副语言特征，使合成音频更加自然、类人且富有表现力。
  - 支持的标签：
    - [ *呼吸*、*笑声*、*惊讶-哦*、*确认-嗯*、*犹豫-呃*、*惊讶-啊*、*惊讶-哇*、*叹息*、*疑问-诶*、*不满-哼* ]

- **可用标签**
<table>
  <tr>
    <td rowspan="8" style="vertical-align: middle; text-align:center;" align="center">情感</td>
    <td align="center"><b>快乐</b></td>
    <td align="center">表达快乐</td>
    <td align="center"><b>愤怒</b></td>
    <td align="center">表达愤怒</td>
  </tr>
  <tr>
    <td align="center"><b>悲伤</b></td>
    <td align="center">表达悲伤</td>
    <td align="center"><b>恐惧</b></td>
    <td align="center">表达恐惧</td>
  </tr>
  <tr>
    <td align="center"><b>惊讶</b></td>
    <td align="center">表达惊讶</td>
    <td align="center"><b>困惑</b></td>
    <td align="center">表达困惑</td>
  </tr>
  <tr>
    <td align="center"><b>共情</b></td>
    <td align="center">表达共情和理解</td>
    <td align="center"><b>尴尬</b></td>
    <td align="center">表达尴尬</td>
  </tr>
  <tr>
    <td align="center"><b>兴奋</b></td>
    <td align="center">表达兴奋和热情</td>
    <td align="center"><b>沮丧</b></td>
    <td align="center">表达沮丧或气馁的情绪</td>
  </tr>
  <tr>
    <td align="center"><b>钦佩</b></td>
    <td align="center">表达钦佩或尊敬</td>
    <td align="center"><b>冷漠</b></td>
    <td align="center">表达冷漠和漠不关心</td>
  </tr>
  <tr>
    <td align="center"><b>厌恶</b></td>
    <td align="center">表达厌恶或反感</td>
    <td align="center"><b>幽默</b></td>
    <td align="center">表达幽默或俏皮</td>
  </tr>
  <tr>
  </tr>
  <tr>
    <td rowspan="17" style="vertical-align: middle; text-align:center;" align="center">说话风格</td>
    <td align="center"><b>严肃</b></td>
    <td align="center">以严肃或庄重的方式说话</td>
    <td align="center"><b>傲慢</b></td>
    <td align="center">以傲慢的方式说话</td>
  </tr>
  <tr>
    <td align="center"><b>儿童</b></td>
    <td align="center">以儿童般的方式说话</td>
    <td align="center"><b>老年</b></td>
    <td align="center">以老年人的声音方式说话</td>
  </tr>
  <tr>
    <td align="center"><b>少女</b></td>
    <td align="center">以轻快、年轻女性的方式说话</td>
    <td align="center"><b>清纯</b></td>
    <td align="center">以清纯、天真的方式说话</td>
  </tr>
  <tr>
    <td align="center"><b>御姐</b></td>
    <td align="center">以成熟、自信女性的方式说话</td>
    <td align="center"><b>甜美</b></td>
    <td align="center">以甜美、可爱的方式说话</td>
  </tr>
  <tr>
    <td align="center"><b>夸张</b></td>
    <td align="center">以夸张、戏剧性的方式说话</td>
    <td align="center"><b>空灵</b></td>
    <td align="center">以柔和、飘渺、梦幻的方式说话</td>
  </tr>
  <tr>
    <td align="center"><b>耳语</b></td>
    <td align="center">以耳语、非常轻柔的方式说话</td>
    <td align="center"><b>豪爽</b></td>
    <td align="center">以豪爽、外向、直率的方式说话</td>
  </tr>
  <tr>
    <td align="center"><b>朗诵</b></td>
    <td align="center">以清晰、有节奏、朗诵诗歌的方式说话</td>
    <td align="center"><b>撒娇</b></td>
    <td align="center">以甜美、俏皮、可爱的方式说话</td>
  </tr>
  <tr>
    <td align="center"><b>温暖</b></td>
    <td align="center">以温暖、友好的方式说话</td>
    <td align="center"><b>害羞</b></td>
    <td align="center">以害羞、胆怯的方式说话</td>
  </tr>
  <tr>
    <td align="center"><b>安慰</b></td>
    <td align="center">以安慰、令人安心的方式说话</td>
    <td align="center"><b>权威</b></td>
    <td align="center">以权威、命令的方式说话</td>
  </tr>
  <tr>
    <td align="center"><b>聊天</b></td>
    <td align="center">以随意、对话的方式说话</td>
    <td align="center"><b>广播</b></td>
    <td align="center">以电台广播的方式说话</td>
  </tr>
  <tr>
    <td align="center"><b>深情</b></td>
    <td align="center">以发自内心、深情的方式说话</td>
    <td align="center"><b>温柔</b></td>
    <td align="center">以温柔、柔和的方式说话</td>
  </tr>
  <tr>
    <td align="center"><b>讲故事</b></td>
    <td align="center">以叙述性、有声读物风格的方式说话</td>
    <td align="center"><b>生动</b></td>
    <td align="center">以活泼、富有表现力的方式说话</td>
  </tr>
  <tr>
    <td align="center"><b>主持</b></td>
    <td align="center">以节目主持/主持人的方式说话</td>
    <td align="center"><b>新闻</b></td>
    <td align="center">以新闻广播的方式说话</td>
  </tr>
  <tr>
    <td align="center"><b>广告</b></td>
    <td align="center">以精致、高端商业配音的方式说话</td>
    <td align="center"><b>咆哮</b></td>
    <td align="center">以大声、深沉、咆哮的方式说话</td>
  </tr>
  <tr>
    <td align="center"><b>低语</b></td>
    <td align="center">以安静、低沉的方式说话</td>
    <td align="center"><b>喊叫</b></td>
    <td align="center">以大声、尖锐、喊叫的方式说话</td>
  </tr>
  <tr>
    <td align="center"><b>深沉</b></td>
    <td align="center">以深沉、低音的语调说话</td>
    <td align="center"><b>响亮</b></td>
    <td align="center">以响亮、高音的语调说话</td>
  </tr>
  <tr>
  </tr>
  <tr>
  </tr>
  <tr>
    <td rowspan="5" style="vertical-align: middle; text-align:center;" align="center">副语言</td>
    <td align="center"><b>呼吸</b></td>
    <td align="center">呼吸声</td>
    <td align="center"><b>笑声</b></td>
    <td align="center">笑声或笑的声音</td>
  </tr>
  <tr>
    <td align="center"><b>犹豫-呃</b></td>
    <td align="center">犹豫声："呃"</td>
    <td align="center"><b>叹息</b></td>
    <td align="center">叹息声</td>
  </tr>
  <tr>
    <td align="center"><b>惊讶-哦</b></td>
    <td align="center">表达惊讶："哦"</td>
    <td align="center"><b>惊讶-啊</b></td>
    <td align="center">表达惊讶："啊"</td>
  </tr>
  <tr>
    <td align="center"><b>惊讶-哇</b></td>
    <td align="center">表达惊讶："哇"</td>
    <td align="center"><b>确认-嗯</b></td>
    <td align="center">确认声："嗯"</td>
  </tr>
  <tr>
    <td align="center"><b>疑问-诶</b></td>
    <td align="center">疑问声："诶"</td>
    <td align="center"><b>不满-哼</b></td>
    <td align="center">不满声："哼"</td>
  </tr>
</table>

## 功能请求与愿望清单
💡 我们欢迎所有新功能的想法！如果您希望看到某个功能被添加到项目中，请在 [Discussions](https://github.com/stepfun-ai/Step-Audio-EditX/discussions) 部分发起讨论。

我们将在这里收集社区反馈，并将受欢迎的建议纳入我们的未来开发计划。感谢您的贡献！

## 演示示例

<table>
  <tr>
    <th style="vertical-align : middle;text-align: center">任务</th>
    <th style="vertical-align : middle;text-align: center">文本</th>
    <th style="vertical-align : middle;text-align: center">原始音频</th>
    <th style="vertical-align : middle;text-align: center">编辑后</th>
  </tr>

  <tr>
    <td align="center"> 情感-恐惧</td>
    <td align="center"> 我总觉得，有人在跟着我，我能听到奇怪的脚步声。</td>
    <td align="center">

  [fear_zh_female_prompt.webm](https://github.com/user-attachments/assets/a088c059-032c-423f-81d6-3816ba347ff5) 
  </td>
    <td align="center">
      
  [fear_zh_female_output.webm](https://github.com/user-attachments/assets/917494ac-5913-4949-8022-46cf55ca05dd)
  </td>
  </tr>


  <tr>
    <td align="center"> 风格-耳语</td>
    <td align="center"> 比如在工作间隙，做一些简单的伸展运动，放松一下身体，这样，会让你更有精力。</td>
    <td align="center">
      
  [whisper_prompt.webm](https://github.com/user-attachments/assets/ed9e22f1-1bac-417b-913a-5f1db31f35c9)
  </td>
    <td align="center">
      
  [whisper_output.webm](https://github.com/user-attachments/assets/e0501050-40db-4d45-b380-8bcc309f0b5f)
  </td>
  </tr>

  <tr>
    <td align="center"> 风格-撒娇</td>
    <td align="center"> 我今天想喝奶茶，可是不知道喝什么口味，你帮我选一下嘛，你选的都好喝～</td>
    <td align="center">

  [act_coy_prompt.webm](https://github.com/user-attachments/assets/74d60625-5b3c-4f45-becb-0d3fe7cc4b3f)
  </td>
    <td align="center"> 

  [act_coy_output.webm](https://github.com/user-attachments/assets/b2f74577-56c2-4997-afd6-6bf47d15ea51)
  </td>
  </tr>


  <tr>
    <td align="center"> 副语言</td>
    <td align="center"> 你这次又忘记带钥匙了 [不满-哼]，真是拿你没办法。</td>
    <td align="center">
      
  [paralingustic_prompt.webm](https://github.com/user-attachments/assets/21e831a3-8110-4c64-a157-60e0cf6735f0)
  </td>
    <td align="center">
      
  [paralingustic_output.webm](https://github.com/user-attachments/assets/a82f5a40-c6a3-409b-bbe6-271180b20d7b)
  </td>
  </tr>


  <tr>
    <td align="center"> 降噪</td>
    <td align="center"> Such legislation was clarified and extended from time to time thereafter. No, the man was not drunk, he wondered how we got tied up with this stranger. Suddenly, my reflexes had gone. It's healthier to cook without sugar.</td>
    <td align="center">
      
  [denoising_prompt.webm](https://github.com/user-attachments/assets/70464bf4-ebde-44a3-b2a6-8c292333319b)
  </td>
    <td align="center">
      
  [denoising_output.webm](https://github.com/user-attachments/assets/7cd0ae8d-1bf0-40fc-9bcd-f419bd4b2d21)
  </td>
  </tr>

  <tr>
    <td align="center"> 语速-加快</td>
    <td align="center"> 上次你说鞋子有点磨脚，我给你买了一双软软的鞋垫。</td>
    <td align="center">
      
  [speed_faster_prompt.webm](https://github.com/user-attachments/assets/db46609e-1b98-48d8-99c8-e166cfdfc6e3)
  </td>
    <td align="center">
      
  [speed_faster_output.webm](https://github.com/user-attachments/assets/0fbc14ca-dd4a-4362-aadc-afe0629f4c9f)
  </td>
  </tr>
  
</table>


更多示例，请参见 [演示页面](https://stepaudiollm.github.io/step-audio-editx/)。

## 模型下载

| 模型   | 🤗 Hugging Face | ModelScope |
|-------|-------|-------|
| Step-Audio-EditX | [stepfun-ai/Step-Audio-EditX](https://huggingface.co/stepfun-ai/Step-Audio-EditX) | [stepfun-ai/Step-Audio-EditX](https://modelscope.cn/models/stepfun-ai/Step-Audio-EditX) |
| Step-Audio-Tokenizer | [stepfun-ai/Step-Audio-Tokenizer](https://huggingface.co/stepfun-ai/Step-Audio-Tokenizer) | [stepfun-ai/Step-Audio-Tokenizer](https://modelscope.cn/models/stepfun-ai/Step-Audio-Tokenizer) |


## 模型使用
### 📜 运行要求
下表展示了运行 Step-Audio-EditX 模型的要求（批次大小 = 1）：

|     模型    | 参数量 |  设置<br/>(采样频率) | GPU 最佳显存  |
|------------|------------|--------------------------------|----------------|
| Step-Audio-EditX   | 3B|         41.6Hz          |       12 GB        |

* 首选执行设备：
  * 支持 CUDA 的 NVIDIA GPU，或支持 MPS 的 Apple Silicon。
  * 模型在单张 L40S GPU 上测试通过。
  * 12GB 只是一个临界值，16GB GPU 显存会更安全。
* 在某些环境中，Tokenizer/ASR 在 CPU 上运行可能更稳定。
* 测试通过的操作系统：Linux

### 🔧 依赖与安装
- Python >= 3.10.0（建议使用 [Anaconda](https://www.anaconda.com/download/#linux) 或 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)）
- [PyTorch >= 2.4.1-cu121](https://pytorch.org/)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

> [!提示]
> 如果网络/代理设置导致安装/运行时依赖下载失败，请在需要直接访问上游的步骤中使用 `no_proxy='*'`。

```bash
git clone https://github.com/stepfun-ai/Step-Audio-EditX.git
conda create -n stepaudioedit python=3.10
conda activate stepaudioedit

cd Step-Audio-EditX
pip install -r requirements.txt

git lfs install
git clone https://huggingface.co/stepfun-ai/Step-Audio-Tokenizer
git clone https://huggingface.co/stepfun-ai/Step-Audio-EditX

```

下载模型后，your_download_dir 应具有以下结构：
```
your_download_dir
├── Step-Audio-Tokenizer
├── Step-Audio-EditX
```

#### 使用 Docker 运行

您可以使用提供的 Dockerfile 设置运行 Step-Audio-EditX 所需的环境。

```bash
# 构建 docker
docker build . -t step-audio-editx

# 运行 docker
docker run --rm --gpus all \
    -v /your/code/path:/app \
    -v /your/model/path:/model \
    -p 7860:7860 \
    step-audio-editx
```

### 🚦 推理设备设置
- 在 `tts_infer.py` 中使用以下标志控制计算设备分配：
  - `--device-map`：控制 LLM 放置位置（`auto/cuda/mps/cpu`）。
  - `--cosy-device`：控制 CosyVoice/声码器放置位置（`auto/cuda/mps/cpu`）。
  - `--funasr-device`：控制 tokenizer/编码放置位置（`auto/cuda/mps/cpu`）。
  - `--max-length`：控制最大生成长度（默认 `8192`）。
- 对于 8GB 级别的 GPU，较低的 `--max-length`（例如 `1024`）可以降低 OOM 风险。
- 当 tokenizer 在 MPS 上不稳定时，`--funasr-device cpu` 通常更安全。

#### 本地推理演示
> [!提示]
> 为了获得最佳性能，每次推理的音频长度请保持在 30 秒以内。

```bash
# 零样本克隆
# 生成的音频文件路径为 output/fear_zh_female_prompt_cloned.wav
python3 tts_infer.py \
    --model-path your_download_dir \
    --prompt-text "我总觉得，有人在跟着我，我能听到奇怪的脚步声。"\
    --prompt-audio "examples/fear_zh_female_prompt.wav"\
    --generated-text "可惜没有如果，已经发生的事情终究是发生了。" \
    --edit-type "clone" \
    --device-map auto \
    --cosy-device auto \
    --funasr-device cpu \
    --max-length 1024 \
    --output-dir ./output 

python3 tts_infer.py \
    --model-path your_download_dir \
    --prompt-text "His political stance was conservative, and he was particularly close to margaret thatcher."\
    --prompt-audio "examples/zero_shot_en_prompt.wav"\
    --generated-text "Underneath the courtyard is a large underground exhibition room which connects the two buildings.	" \
    --edit-type "clone" \
    --device-map auto \
    --cosy-device auto \
    --funasr-device cpu \
    --max-length 1024 \
    --output-dir ./output 

# 编辑
# 每次编辑迭代会有一个或多个对应的 wav 文件，例如：output/fear_zh_female_prompt_edited_iter1.wav、output/fear_zh_female_prompt_edited_iter2.wav、...
# 情感；恐惧
python3 tts_infer.py \
    --model-path your_download_dir \
    --prompt-text "我总觉得，有人在跟着我，我能听到奇怪的脚步声。" \
    --prompt-audio "examples/fear_zh_female_prompt.wav"\
    --edit-type "emotion" \
    --edit-info "fear" \
    --n-edit-iter 2 \
    --output-dir ./output 

# 情感；快乐
python3 tts_infer.py \
    --model-path your_download_dir \
    --prompt-text "You know, I just finished that big project and feel so relieved. Everything seems easier and more colorful, what a wonderful feeling!" \
    --prompt-audio "examples/en_happy_prompt.wav"\
    --edit-type "emotion" \
    --edit-info "happy" \
    --n-edit-iter 2 \
    --output-dir ./output 

# 风格；耳语
# 对于耳语风格，编辑迭代次数应设置大于 1 以获得更好的效果。
python3 tts_infer.py \
    --model-path your_download_dir \
    --prompt-text "比如在工作间隙，做一些简单的伸展运动，放松一下身体，这样，会让你更有精力." \
    --prompt-audio "examples/whisper_prompt.wav" \
    --edit-type "style" \
    --edit-info "whisper" \
    --n-edit-iter 2 \
    --output-dir ./output 

# 副语言
# 支持的标签：呼吸、笑声、惊讶-哦、确认-嗯、犹豫-呃、惊讶-啊、惊讶-哇、叹息、疑问-诶、不满-哼
python3 tts_infer.py \
    --model-path your_download_dir \
    --prompt-text "我觉得这个计划大概是可行的，不过还需要再仔细考虑一下。" \
    --prompt-audio "examples/paralingustic_prompt.wav" \
    --generated-text "我觉得这个计划大概是可行的，[犹豫-呃]不过还需要再仔细考虑一下。" \
    --edit-type "paralinguistic" \
    --output-dir ./output 

# 降噪
# 不需要提示文本。
python3 tts_infer.py \
    --model-path your_download_dir \
    --prompt-audio "examples/denoise_prompt.wav"\
    --edit-type "denoise" \
    --output-dir ./output 

# 语音活动检测
# 不需要提示文本。
python3 tts_infer.py \
    --model-path your_download_dir \
    --prompt-audio "examples/vad_prompt.wav" \
    --edit-type "vad" \
    --output-dir ./output 

# 语速
# 支持的 edit-info：faster、slower、more faster、more slower
python3 tts_infer.py \
    --model-path your_download_dir \
    --prompt-text "上次你说鞋子有点磨脚，我给你买了一双软软的鞋垫。" \
    --prompt-audio "examples/speed_prompt.wav" \
    --edit-type "speed" \
    --edit-info "faster" \
    --output-dir ./output 

```



#### 启动 Web 演示
启动本地服务器进行在线推理。
假设您有一张至少 12GB 显存的 GPU 可用，并且已经下载了所有模型。

```bash
# Step-Audio-EditX 演示
python app.py --model-path your_download_dir --model-source local

# 运行时量化内存优化选项
# 对于显存有限的系统，您可以使用量化来减少内存使用：

# INT8 量化
python app.py --model-path your_download_dir --model-source local --quantization int8

# INT4 量化
python app.py --model-path your_download_dir --model-source local --quantization int4

# 使用预量化的 AWQ 模型
python app.py --model-path path/to/quantized/model --model-source local --quantization awq-4bit

# 自定义设置示例：
python app.py --model-path your_download_dir --model-source local --torch-dtype float16 --enable-auto-transcribe
```

### 📚 入门文档
- `TTS_CLONE_MPS.md`：MPS/克隆工作流程、设备分配策略和故障排除说明的实用指南。
- `PROMPT.md`：克隆提示如何组装（`<audio_*>` 令牌、合并逻辑、模板使用）。
- `retry_report_2026-02-24.md`：本分支的最新运行报告，包含成功/失败的命令和 GPU-CPU 观察结果。
- `experience.md`：环境问题和修复的历史记录。
- `quantization/README.md`：量化设置和选项。
- `scripts/clone_paralingustic_prompt.py` / `scripts/batch_clone.py`：可运行的批量/快速入门助手。

### 🔄 模型量化（可选）

对于显存有限的用户，您可以创建模型的量化版本以减少内存需求：

```bash
# 创建 AWQ 4 位量化模型
python quantization/awq_quantize.py --model_path path/to/Step-Audio-EditX

# 高级量化选项
python quantization/awq_quantize.py
```

有关详细的量化选项和参数，请参见 [quantization/README.md](quantization/README.md)。


## 技术细节
<img src="assets/architechture.png" width=900>
Step-Audio-EditX 包含三个主要组件：

- 双码本音频 tokenizer，将参考音频或输入音频转换为离散令牌。
- 生成双码本令牌序列的音频大语言模型。
- 音频解码器，使用流匹配方法将音频大语言模型预测的双码本令牌序列转换回音频波形。

音频编辑支持对所有声音进行情感和说话风格的迭代控制，利用 SFT 和 PPO 训练期间的大裕度数据。

## 评估

### Step-Audio-EditX 与闭源模型的对比

- Step-Audio-EditX 在零样本克隆和情感控制方面均优于 Minimax 和 Doubao。
- Step-Audio-EditX 的情感编辑在一次迭代后显著改善了所有三个模型的情感控制音频输出。经过进一步迭代，它们的整体性能持续提高。

<div align="center">
<img src="assets/emotion-eval.png" width=800 >
</div>

### 在闭源模型上的泛化能力
- 对于情感和说话风格编辑，领先闭源系统的内置声音具有相当强的上下文能力，使它们能够部分传达文本中的情感。经过 Step-Audio-EditX 单次编辑后，所有语音模型的情感和风格准确性均显著提高。在接下来的两次迭代中观察到进一步改善，有力地证明了我们的模型具有强大的泛化能力。

- 对于副语言编辑，经过 Step-Audio-EditX 编辑后，副语言再现的性能与闭源模型的内置声音在直接合成原生副语言内容时达到的性能相当。（**sub** 表示将副语言标签替换为原生词）


<div align="center">

  <table border="1" cellspacing="0" cellpadding="5" style="border-collapse: collapse; font-family: sans-serif; width: auto;">
    <caption><b>表：情感、说话风格和副语言编辑在闭源模型上的泛化能力。</b></caption>
    <thead>
      <tr>
        <th rowspan="2" align="center" style="vertical-align: bottom;">语言</th>
        <th rowspan="2" align="center" style="vertical-align: bottom;">模型</th>
        <th colspan="4" style="border-bottom: 1px solid black;">情感 ↑</th>
        <th colspan="4" style="border-bottom: 1px solid black;">说话风格 ↑</th>
        <th colspan="3" style="border-bottom: 1px solid black; border-left: 1px solid black;">副语言 ↑</th>
      </tr>
      <tr>
        <th>迭代<sub>0</sub></th>
        <th>迭代<sub>1</sub></th>
        <th>迭代<sub>2</sub></th>
        <th>迭代<sub>3</sub></th>
        <th style="border-left: 1px solid #ccc;">迭代<sub>0</sub></th>
        <th>迭代<sub>1</sub></th>
        <th>迭代<sub>2</sub></th>
        <th>迭代<sub>3</sub></th>
        <th style="border-left: 1px solid black;">迭代<sub>0</sub></th>
        <th>替换</th>
        <th>迭代<sub>1</sub></th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td rowspan="4" align="center" style="font-weight: bold; vertical-align: middle;">中文</td>
        <td align="left">MiniMax-2.6-hd</td>
        <td align="center">71.6</td>
        <td align="center">78.6</td>
        <td align="center">81.2</td>
        <td align="center"><b>83.4</b></td>
        <td align="center" style="border-left: 1px solid #ccc;">36.7</td>
        <td align="center">58.8</td>
        <td align="center">63.1</td>
        <td align="center"><b>67.3</b></td>
        <td align="center" style="border-left: 1px solid black;">1.73</td>
        <td align="center">2.80</td>
        <td align="center">2.90</td>
      </tr>
      <tr>
        <td align="left">Doubao-Seed-TTS-2.0</td>
        <td align="center">67.4</td>
        <td align="center">77.8</td>
        <td align="center">80.6</td>
        <td align="center"><b>82.8</b></td>
        <td align="center" style="border-left: 1px solid #ccc;">38.2</td>
        <td align="center">60.2</td>
        <td align="center"><b>65.0</b></td>
        <td align="center">64.9</td>
        <td align="center" style="border-left: 1px solid black;">1.67</td>
        <td align="center">2.81</td>
        <td align="center">2.90</td>
      </tr>
      <tr>
        <td align="left">GPT-4o-mini-TTS</td>
        <td align="center">62.6</td>
        <td align="center">76.0</td>
        <td align="center">77.0</td>
        <td align="center"><b>81.8</b></td>
        <td align="center" style="border-left: 1px solid #ccc;">45.9</td>
        <td align="center">64.0</td>
        <td align="center">65.7</td>
        <td align="center"><b>69.7</b></td>
        <td align="center" style="border-left: 1px solid black;">1.71</td>
        <td align="center">2.88</td>
        <td align="center">2.93</td>
      </tr>
      <tr style="border-bottom: 1px solid black;">
        <td align="left">ElevenLabs-v2</td>
        <td align="center">60.4</td>
        <td align="center">74.6</td>
        <td align="center">77.4</td>
        <td align="center"><b>79.2</b></td>
        <td align="center" style="border-left: 1px solid #ccc;">43.8</td>
        <td align="center">63.3</td>
        <td align="center">69.7</td>
        <td align="center"><b>70.8</b></td>
        <td align="center" style="border-left: 1px solid black;">1.70</td>
        <td align="center">2.71</td>
        <td align="center">2.92</td>
      </tr>
      <tr>
        <td rowspan="4" align="center" style="font-weight: bold; vertical-align: middle;">英文</td>
        <td align="left">MiniMax-2.6-hd</td>
        <td align="center">55.0</td>
        <td align="center">64.0</td>
        <td align="center">64.2</td>
        <td align="center"><b>66.4</b></td>
        <td align="center" style="border-left: 1px solid #ccc;">51.9</td>
        <td align="center">60.3</td>
        <td align="center">62.3</td>
        <td align="center"><b>64.3</b></td>
        <td align="center" style="border-left: 1px solid black;">1.72</td>
        <td align="center">2.87</td>
        <td align="center">2.88</td>
      </tr>
      <tr>
        <td align="left">Doubao-Seed-TTS-2.0</td>
        <td align="center">53.8</td>
        <td align="center">65.8</td>
        <td align="center">65.8</td>
        <td align="center"><b>66.2</b></td>
        <td align="center" style="border-left: 1px solid #ccc;">47.0</td>
        <td align="center">62.0</td>
        <td align="center"><b>62.7</b></td>
        <td align="center">62.3</td>
        <td align="center" style="border-left: 1px solid black;">1.72</td>
        <td align="center">2.75</td>
        <td align="center">2.92</td>
      </tr>
      <tr>
        <td align="left">GPT-4o-mini-TTS</td>
        <td align="center">56.8</td>
        <td align="center">61.4</td>
        <td align="center">64.8</td>
        <td align="center"><b>65.2</b></td>
        <td align="center" style="border-left: 1px solid #ccc;">52.3</td>
        <td align="center">62.3</td>
        <td align="center">62.4</td>
        <td align="center"><b>63.4</b></td>
        <td align="center" style="border-left: 1px solid black;">1.90</td>
        <td align="center">2.90</td>
        <td align="center">2.88</td>
      </tr>
      <tr style="border-bottom: 1px solid black;">
        <td align="left">ElevenLabs-v2</td>
        <td align="center">51.0</td>
        <td align="center">61.2</td>
        <td align="center">64.0</td>
        <td align="center"><b>65.2</b></td>
        <td align="center" style="border-left: 1px solid #ccc;">51.0</td>
        <td align="center">62.1</td>
        <td align="center">62.6</td>
        <td align="center"><b>64.0</b></td>
        <td align="center" style="border-left: 1px solid black;">1.93</td>
        <td align="center">2.87</td>
        <td align="center">2.88</td>
      </tr>
      <tr>
        <td rowspan="4" align="center" style="font-weight: bold; vertical-align: middle;">平均</td>
        <td align="left">MiniMax-2.6-hd</td>
        <td align="center">63.3</td>
        <td align="center">71.3</td>
        <td align="center">72.7</td>
        <td align="center"><b>74.9</b></td>
        <td align="center" style="border-left: 1px solid #ccc;">44.2</td>
        <td align="center">59.6</td>
        <td align="center">62.7</td>
        <td align="center"><b>65.8</b></td>
        <td align="center" style="border-left: 1px solid black;">1.73</td>
        <td align="center">2.84</td>
        <td align="center">2.89</td>
      </tr>
      <tr>
        <td align="left">Doubao-Seed-TTS-2.0</td>
        <td align="center">60.6</td>
        <td align="center">71.8</td>
        <td align="center">73.2</td>
        <td align="center"><b>74.5</b></td>
        <td align="center" style="border-left: 1px solid #ccc;">42.6</td>
        <td align="center">61.1</td>
        <td align="center"><b>63.9</b></td>
        <td align="center">63.6</td>
        <td align="center" style="border-left: 1px solid black;">1.70</td>
        <td align="center">2.78</td>
        <td align="center">2.91</td>
      </tr>
      <tr>
        <td align="left">GPT-4o-mini-TTS</td>
        <td align="center">59.7</td>
        <td align="center">68.7</td>
        <td align="center">70.9</td>
        <td align="center"><b>73.5</b></td>
        <td align="center" style="border-left: 1px solid #ccc;">49.1</td>
        <td align="center">63.2</td>
        <td align="center">64.1</td>
        <td align="center"><b>66.6</b></td>
        <td align="center" style="border-left: 1px solid black;">1.81</td>
        <td align="center">2.89</td>
        <td align="center">2.90</td>
      </tr>
      <tr>
        <td align="left">ElevenLabs-v2</td>
        <td align="center">55.7</td>
        <td align="center">67.9</td>
        <td align="center">70.7</td>
        <td align="center"><b>72.2</b></td>
        <td align="center" style="border-left: 1px solid #ccc;">47.4</td>
        <td align="center">62.7</td>
        <td align="center">66.1</td>
        <td align="center"><b>67.4</b></td>
        <td align="center" style="border-left: 1px solid black;">1.82</td>
        <td align="center">2.79</td>
        <td align="center">2.90</td>
      </tr>
    </tbody>
  </table>

</div>


## 致谢

本项目的部分代码和数据来自：
* [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
* [transformers](https://github.com/huggingface/transformers)
* [FunASR](https://github.com/modelscope/FunASR)
* [NVSpeech](https://huggingface.co/datasets/amphion/Emilia-NV)

感谢所有开源项目对本项目的贡献！

## 许可协议
+ 本开源仓库中的代码根据 [Apache 2.0](LICENSE) 许可证授权。

## 引用

```
@misc{yan2025stepaudioeditxtechnicalreport,
      title={Step-Audio-EditX Technical Report}, 
      author={Chao Yan and Boyong Wu and Peng Yang and Pengfei Tan and Guoqiang Hu and Yuxin Zhang and Xiangyu and Zhang and Fei Tian and Xuerui Yang and Xiangyu Zhang and Daxin Jiang and Gang Yu},
      year={2025},
      eprint={2511.03601},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2511.03601}, 
}
```


## ⚠️ 使用免责声明
- 请勿将此模型用于任何未经授权的活动，包括但不限于：
  - 未经许可的声音克隆
  - 身份冒充
  - 欺诈
  - 深度伪造或任何其他非法目的
- 使用本模型时，请确保遵守当地法律法规，并遵循道德准则。
- 模型开发者不对本技术的任何滥用或不当使用负责。

我们倡导负责任的生成式 AI 研究，并敦促社区在 AI 开发和应用中坚持安全和道德标准。如果您对本模型的使用有任何疑虑，请随时与我们联系。

## Star 历史
[![Star History Chart](https://api.star-history.com/svg?repos=stepfun-ai/Step-Audio-EditX&type=Date)](https://star-history.com/#stepfun-ai/Step-Audio-EditX&Date)
