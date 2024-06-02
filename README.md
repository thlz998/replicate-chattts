# ChatTTS Cog model

[![Replicate](https://replicate.com/thlz998/chat-tts/badge)](https://replicate.com/thlz998/chat-tts) 

This is an implementation of the [ChatTTS](https://github.com/2noise/ChatTTS) as a Cog model.

First, download the pre-trained weights:

    cog run script/download-weights 

Then, you can run predictions:

    cog predict -i text="Hello world" -i voice=2222 -i custom_voice=0 -i skip_refine=0 -i temperature=0.3 -i top_p=0.7 -i top_k=20 -i prompt=""

**Parameters:**

| Parameter      | Type    | Required | Default | Description                                                                                                                                              |
|----------------|---------|----------|---------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| text           | `str`   | Yes      | -       | The text to be synthesized into speech.                                                                                                                   |
| voice          | `int`   | No       | 2222    | A number that determines the voice tone. Options are 2222, 7869, 6653, 4099, 5099. You can choose one of these options, or pass in any number for random selection. |
| prompt         | `str`   | No       | Empty   | This sets laughter, pauses, etc., for example, `[oral_2][laugh_0][break_6]`.                                                                               |
| temperature    | `float` | No       | 0.3     | Temperature value for sampling.                                                                                                                           |
| top_p          | `float` | No       | 0.7     | Top p value for nucleus sampling.                                                                                                                         |
| top_k          | `int`   | No       | 20      | Top k value for top-k sampling.                                                                                                                           |
| skip_refine    | `int`   | No       | 0       | 1 means to skip text refining, 0 means not to skip.                                                                                                       |
| custom_voice   | `int`   | No       | 0       | Seed value for custom voice tone generation, must be a positive integer. If set, it will take precedence and the `voice` parameter will be ignored.       |

# ChatTTS Cog 模型

[![Replicate](https://replicate.com/thlz998/chat-tts/badge)](https://replicate.com/thlz998/chat-tts) 

这是一个 [ChatTTS](https://github.com/2noise/ChatTTS) 的 Cog 模型实现。

首先，下载预训练权重：

    cog run script/download-weights 

然后，你可以运行预测：

    cog predict -i text="Hello world" -i voice=2222 -i custom_voice=0 -i skip_refine=0 -i temperature=0.3 -i top_p=0.7 -i top_k=20 -i prompt=""

**参数说明：**

| 参数名         | 类型    | 必填     | 默认值 | 描述                                                                                                                                             |
|----------------|---------|----------|---------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| text           | `str`   | 是       | -       | 要合成为语音的文本。                                                                                                                               |
| voice          | `int`   | 否       | 2222    | 用于确定声音音调的数字。选项有 2222、7869、6653、4099、5099。你可以选择其中一个选项，或者传递任何数字进行随机选择。                                                                      |
| prompt         | `str`   | 否       | 空      | 设置笑声、停顿等。例如，`[oral_2][laugh_0][break_6]`。                                                                                            |
| temperature    | `float` | 否       | 0.3     | 采样时的温度值。                                                                                                                                  |
| top_p          | `float` | 否       | 0.7     | 核心采样的 top p 值。                                                                                                                              |
| top_k          | `int`   | 否       | 20      | top-k 采样的 top k 值。                                                                                                                            |
| skip_refine    | `int`   | 否       | 0       | 1 表示跳过文本精炼，0 表示不跳过。                                                                                                                 |
| custom_voice   | `int`   | 否       | 0       | 用于定制声音音调生成的种子值，必须是正整数。如果设置了这个值，将优先使用，并忽略 `voice` 参数。                                                     |