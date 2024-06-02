import hashlib
import os
import datetime
import numpy as np

torch._dynamo.config.suppress_errors = True
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision('high')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from cog import BasePredictor, Input, Path
import torch
import torch._dynamo
import ChatTTS
import time
import soundfile as sf
from modelscope import snapshot_download

# Constants
class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        MODEL_DIR = "models"
        CHATTTS_DIR = snapshot_download('pzc163/chatTTS', cache_dir=MODEL_DIR)
        self.chat = ChatTTS.Chat()
        self.chat.load_models(source="local", local_path=CHATTTS_DIR)

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
        print(f'{voice=},{custom_voice=}')


        torch.manual_seed(voice)
        
        rand_spk = self.chat.sample_random_speaker()

        audio_files = []
        md5_hash = hashlib.md5()
        md5_hash.update(f"{text}-{voice}-{skip_refine}-{prompt}".encode('utf-8'))
        datename=datetime.datetime.now().strftime('%Y%m%d-%H_%M_%S')
        filename = datename+'-'+md5_hash.hexdigest() + ".wav"
        start_time = time.time()

        
        wavs = self.chat.infer(
            [t for t in text.split("\n") if t.strip()],
            use_decoder=True,
            skip_refine_text=True if int(skip_refine) == 1 else False,
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
        inference_time = end_time - start_time
        inference_time_rounded = round(inference_time, 2)
        print(f"推理时长: {inference_time_rounded} 秒")

        combined_wavdata = np.array([], dtype=wavs[0][0].dtype)  # 确保dtype与你的wav数据类型匹配

        for wavdata in wavs:
            combined_wavdata = np.concatenate((combined_wavdata, wavdata[0]))


        sample_rate = 24000  # Assuming 24kHz sample rate
        audio_duration = len(combined_wavdata) / sample_rate
        audio_duration_rounded = round(audio_duration, 2)
        print(f"音频时长: {audio_duration_rounded} 秒")

        sf.write(filename, combined_wavdata, 24000)
        audio_files.append({
            "filename": Path(filename),
            "inference_time": inference_time_rounded,
            "audio_duration": audio_duration_rounded
        })
        return {"audio_files": audio_files}