import os
import torch
import hashlib
import datetime
import numpy as np
from cog import BasePredictor, Input, Path
import ChatTTS
import time
import soundfile as sf  # Ensure to import soundfile for saving audio

# Constants
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, 'asset/spk_stat.pt')

# Check if model files exist
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model files not found at {MODEL_DIR}. Please ensure the model files are present.")

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.std, self.mean = torch.load(MODEL_PATH).chunk(2)
        self.chat = ChatTTS.Chat()
        self.chat.load_models(source="local", local_path=MODEL_DIR)

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
        if custom_voice > 0:
            voice = custom_voice

        torch.manual_seed(voice)
        rand_spk = self.chat.sample_random_speaker()

        # Calculate the file hash
        md5_hash = hashlib.md5()
        md5_hash.update(f"{text}-{voice}-{skip_refine}-{prompt}".encode('utf-8'))
        datename = datetime.datetime.now().strftime('%Y%m%d-%H_%M_%S')
        filename = datename + '-' + md5_hash.hexdigest() + ".wav"

        # Inference
        start_time = time.time()

        wavs = self.chat.infer(
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
        sf.write(output_path, combined_wavdata, sample_rate)  # Save audio data to output path

        # Create response dict
        response = {
            "code": 0,
            "msg": "ok",
            "audio_files": [{
                "filename": filename,
                "audio_data": str(output_path),  # Ensure this is a string path
                "inference_time": inference_time,
                "audio_duration": audio_duration
            }]
        }

        return response