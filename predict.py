import os, hashlib, webbrowser, time, torch
from datetime import datetime
from dotenv import load_dotenv
from modelscope import snapshot_download
from typing import List, Tuple
from pathlib import Path
from cog import BasePredictor, Path, Input
import soundfile as sf
from flask import Flask, request, render_template, jsonify,  send_from_directory
import ChatTTS
import numpy as np
import hashlib
import ChatTTS
class Predictor(BasePredictor):
    def __init__(self):
        self.chat = ChatTTS.Chat()
        MODEL_DIR = 'models'
        self.CHATTTS_DIR = snapshot_download('pzc163/chatTTS',cache_dir=MODEL_DIR)
        self.chat.load_models(source="local",local_path=self.CHATTTS_DIR)

    def predict(
        self,
        text: str = Input("Text to generate speech from."),
        prompt: str = Input("Optional pattern to guide text synthesis", default=''),
        custom_voice: int = Input("Optional custom voice value", default=0),
        voice: int = Input("Voice selector", default=2222),
        temperature: float = Input("Temperature", default=0.3),
        top_p: float = Input("Top P", default=0.7),
        top_k: int = Input("Top K", default=20),
        skip_refine: int = Input("Whether to skip the refine text stage", choices=[0, 1], default=0),
    ) -> Path:
        std, mean = torch.load(f'{self.CHATTTS_DIR}/asset/spk_stat.pt').chunk(2)
        torch.manual_seed(voice)

        rand_spk = self.chat.sample_random_speaker()
        md5_hash = hashlib.md5()
        md5_hash.update(f"{text}-{voice}-{skip_refine}-{prompt}".encode('utf-8'))
        datename = datetime.now().strftime('%Y%m%d-%H_%M_%S')
        filename = f"/tmp/{datename}-{md5_hash.hexdigest()}.wav"

        wavs = self.chat.infer([t for t in text.split("\n") if t.strip()], use_decoder=True, skip_refine_text=True if int(skip_refine) == 1 else False, params_infer_code={'spk_emb': rand_spk, 'temperature': temperature, 'top_P': top_p, 'top_K': top_k}, params_refine_text= {'prompt': prompt}, do_text_normalization=False)

        combined_wavdata = np.array([], dtype=wavs[0][0].dtype) 

        for wavdata in wavs:
            combined_wavdata = np.concatenate((combined_wavdata, wavdata[0]))
        
        sf.write(filename, combined_wavdata, 24000)
        return Path(filename)