# ChatTTS Cog model

[![Replicate](https://replicate.com/thlz998/chat-tts/badge)](https://replicate.com/thlz998/chat-tts) 

This is an implementation of the [ChatTTS](https://github.com/2noise/ChatTTS) as a Cog model.

First, download the pre-trained weights:

    cog run script/download-weights 

Then, you can run predictions:

    cog predict -i text="Hello world" -i voice=2222 -i custom_voice=0 -i skip_refine=0 -i temperature=0.3 -i top_p=0.7 -i top_k=20 -i prompt=""
