import os
import hashlib
import datetime
import numpy as np
import torch
from cog import BasePredictor, Input, Path
from modelscope import snapshot_download
import ChatTTS
import soundfile as sf
import time

# The ChatTTS model setup
MODEL_DIR="models"
CHATTTS_DIR = snapshot_download('pzc163/chatTTS', cache_dir=MODEL_DIR)
chat = ChatTTS.Chat()
chat.load_models(source="local", local_path=CHATTTS_DIR)

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.std, self.mean = torch.load(f'{CHATTTS_DIR}/asset/spk_stat.pt').chunk(2)
    
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
    ) -> Path:
        if custom_voice > 0:
            voice = custom_voice
        
        torch.manual_seed(voice)
        rand_spk = chat.sample_random_speaker()
        
        # Calculate the file hash
        md5_hash = hashlib.md5()
        md5_hash.update(f"{text}-{voice}-{skip_refine}-{prompt}".encode('utf-8'))
        datename = datetime.datetime.now().strftime('%Y%m%d-%H_%M_%S')
        filename = datename + '-' + md5_hash.hexdigest() + ".wav"
        
        # Inference
        start_time = time.time()
        
        wavs = chat.infer(
            [t for t in text.split("\n") if t.strip()],
            use_decoder=True,
            skip_refine_text=True if int(skip_refine) == 1 else False,
            params_infer_code={'spk_emb': rand_spk, 'temperature': temperature, 'top_P': top_p, 'top_K': top_k},
            params_refine_text={'prompt': prompt},
            do_text_normalization=False
        )
        
        end_time = time.time()
        inference_time = round(end_time - start_time, 2)
        
        combined_wavdata = np.concatenate([wav[0] for wav in wavs])
        
        sample_rate = 24000  # Assuming 24kHz sample rate
        audio_duration = round(len(combined_wavdata) / sample_rate, 2)
        
        # Save the resulting WAV file
        output_path = Path("/tmp") / filename
        sf.write(output_path, combined_wavdata, sample_rate)
        
        # Create response dict
        response = {
            "code": 0,
            "msg": "ok",
            "audio_files": [{
                "filename": filename,
                "audio_data": str(output_path),
                "output_path": output_path,
                "inference_time": inference_time,
                "audio_duration": audio_duration
            }]
        }
        
        # Return the path of the generated audio file
        return response