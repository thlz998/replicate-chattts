import os
import hashlib
import datetime
import numpy as np
import torch
import soundfile as sf
from cog import BasePredictor, Input
from modelscope import snapshot_download
import ChatTTS
import time
import base64
from io import BytesIO

# The ChatTTS model setup
MODEL_DIR="models"
CHATTTS_DIR = snapshot_download('pzc163/chatTTS', cache_dir=MODEL_DIR)
chat = ChatTTS.Chat()
chat.load_models(source="local", local_path=CHATTTS_DIR)
# std and mean global variables
std, mean = torch.load(f'{CHATTTS_DIR}/asset/spk_stat.pt').chunk(2)

WAVS_DIR_PATH=Path("wavs")
WAVS_DIR_PATH.mkdir(parents=True, exist_ok=True)
WAVS_DIR=WAVS_DIR_PATH.as_posix()

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        pass  # No setup needed as everything is loaded globally

    @torch.inference_mode()
    def predict(self,
                text: str = Input(description="Text to be synthesized", default="Hello world!"),
                voice: int = Input(description="Voice identifier", default=2222),
                custom_voice: int = Input(description="Custom voice identifier", default=0),
                skip_refine: int = Input(description="Skip refine text step", default=0),
                temperature: float = Input(description="Temperature for sampling", default=0.3),
                top_p: float = Input(description="Top-p sampling parameter", default=0.7),
                top_k: int = Input(description="Top-k sampling parameter", default=20),
                prompt: str = Input(description="Prompt for refining text", default=''),
    ) -> dict:
        rand_spk = chat.sample_random_speaker()
        torch.manual_seed(voice)
        
        # Generate a filename based on input parameters
        md5_hash = hashlib.md5()
        request_signature = f"{text}-{voice}-{skip_refine}-{prompt}"
        md5_hash.update(request_signature.encode('utf-8'))
        datename = datetime.datetime.now().strftime('%Y%m%d-%H_%M_%S')
        filename = datename + '-' + md5_hash.hexdigest() + ".wav"
        
        # Perform TTS inference
        start_time = time.time()
        wavs = chat.infer(
            [t for t in text.split("\n") if t.strip()],
            use_decoder=True,
            skip_refine_text=bool(skip_refine),
            params_infer_code={
                'spk_emb': rand_spk,
                'temperature': temperature,
                'top_P': top_p,
                'top_K': top_k
            },
            params_refine_text={'prompt': prompt},
            do_text_normalization=False
        )
        end_time = time.time()
        inference_time = round(end_time - start_time, 2)

        # Combine wav files
        combined_wavdata = np.concatenate([wavdata[0] for wavdata in wavs])
        sample_rate = 24000
        audio_duration = round(len(combined_wavdata) / sample_rate, 2)

        # Convert audio data to Base64
        buffer = BytesIO()
        sf.write(buffer, combined_wavdata, sample_rate, format='WAV')
        encoded_audio = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Create response dict
        response = {
            "code": 0,
            "msg": "ok",
            "audio_files": [{
                "filename": filename,
                "audio_data": encoded_audio,
                "inference_time": inference_time,
                "audio_duration": audio_duration
            }]
        }
        
        return response