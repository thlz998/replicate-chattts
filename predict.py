import hashlib
import os
import datetime
import numpy as np
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision('high')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from cog import BasePredictor, Input, Path
import ChatTTS
import time
import soundfile as sf
from modelscope import snapshot_download


# Constants
MODEL_DIR = "models"


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        MODEL_DIR = "models"
        CHATTTS_DIR = snapshot_download('pzc163/chatTTS', cache_dir=MODEL_DIR)
        self.chat = ChatTTS.Chat()
        self.chat.load_models(source="local", local_path=CHATTTS_DIR)
        print("模型加载成功，初始化完成")
    @torch.inference_mode()
    def predict(self,
                text: str = Input(description="Text to be synthesized", default="Hello world!"),
                voice: int = Input(
                    description="Voice identifier",
                    default=2222,
                    ge=0,
                    choices=[2222, 7869, 6653, 4099, 5099],
                ),
                custom_voice: int = Input(description="Custom voice identifier", default=0, ge=0),
                skip_refine: int = Input(description="Skip refine text step", default=0, choices=[0, 1]),
                temperature: float = Input(description="Temperature for sampling", default=0.3, ge=0.0, le=1.0),
                top_p: float = Input(description="Top-p sampling parameter", default=0.7, ge=0.0, le=1.0),
                top_k: int = Input(description="Top-k sampling parameter", default=20, ge=0),
                prompt: str = Input(description="Prompt for refining text", default=''),
    ) -> dict:
        # Use custom_voice if set and greater than 0
        if custom_voice > 0:
            voice = custom_voice

        print(f'{voice=},{custom_voice=}')

        # Set the random seed for reproducibility
        torch.manual_seed(voice)
        
        # Sample a random speaker embedding
        rand_spk = self.chat.sample_random_speaker()

        # Prepare filename using MD5 hash
        md5_hash = hashlib.md5()
        md5_hash.update(f"{text}-{voice}-{skip_refine}-{prompt}".encode('utf-8'))
        datename = datetime.datetime.now().strftime('%Y%m%d-%H_%M_%S')
        filename = datename + '-' + md5_hash.hexdigest() + ".wav"

        # Start timing the inference process
        start_time = time.time()
        print(f'{start_time=}')

        # Perform inference
        wavs = self.chat.infer(
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

        # Calculate inference time
        end_time = time.time()
        inference_time = round(end_time - start_time, 2)
        print(f"推理时长: {inference_time} 秒")

        # Combine all audio segments
        combined_wavdata = np.concatenate([wav[0] for wav in wavs])

        # Calculate audio duration
        sample_rate = 24000  # Assuming 24kHz sample rate
        audio_duration = round(len(combined_wavdata) / sample_rate, 2)
        print(f"音频时长: {audio_duration} 秒")

        # Save audio file
        sf.write(filename, combined_wavdata, sample_rate)
        audio_files = [{
            "filename": Path(filename),
            "inference_time": inference_time,
            "audio_duration": audio_duration
        }]
        
        return {"audio_files": audio_files}
