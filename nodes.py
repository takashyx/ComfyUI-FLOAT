import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

import folder_paths
import comfy.model_management as mm
import time
import torchaudio
import torchvision.utils as vutils
import torch
import numpy as np
import comfy.utils
import comfy.sd

from .generate import InferenceAgent
from .options.base_options import BaseOptionsJson

class LoadFloatModels:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (['float.pth'],)
            }
        }

    RETURN_TYPES = ("FLOAT_PIPE",)
    RETURN_NAMES = ("float_pipe",)
    FUNCTION = "loadmodel"
    CATEGORY = "FLOAT"
    DESCRIPTION = "Models are auto-downloaded to /ComfyUI/models/float"

    def loadmodel(self, model):
        # download models if not exist
        float_models_dir = os.path.join(folder_paths.models_dir, "float")
        os.makedirs(float_models_dir, exist_ok=True)

        wav2vec2_base_960h_models_dir = os.path.join(float_models_dir,"wav2vec2-base-960h") 
        wav2vec_english_speech_emotion_recognition_models_dir = os.path.join(float_models_dir,"wav2vec-english-speech-emotion-recognition") 
        float_model_path = os.path.join(float_models_dir,"float.pth")

        if not os.path.exists(float_model_path) or not os.path.isdir(wav2vec2_base_960h_models_dir) or not os.path.isdir(wav2vec_english_speech_emotion_recognition_models_dir):
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="yuvraj108c/float", local_dir=float_models_dir, local_dir_use_symlinks=False)

        # Load the model and convert to MPS
        if os.path.exists(float_model_path):
            # Load the model
            model_data = comfy.utils.load_torch_file(float_model_path)
            
            # Convert model tensors to MPS
            device = 'mps' if torch.backends.mps.is_available() else 'cpu'
            for key in model_data:
                if isinstance(model_data[key], torch.Tensor):
                    model_data[key] = model_data[key].to(device)
            
            # Save the converted model
            torch.save(model_data, float_model_path)

        # use custom dictionary instead of original parser for arguments
        opt = BaseOptionsJson
        opt.rank = 'mps' if torch.backends.mps.is_available() else 'cpu'
        opt.ngpus = 1
        opt.ckpt_path = float_model_path
        opt.pretrained_dir = float_models_dir
        opt.wav2vec_model_path = wav2vec2_base_960h_models_dir
        opt.audio2emotion_path = wav2vec_english_speech_emotion_recognition_models_dir
        agent = InferenceAgent(opt)

        return (agent,)

class FloatProcess:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ref_image": ("IMAGE",),
                "ref_audio": ("AUDIO",),
                "float_pipe": ("FLOAT_PIPE",),
                "a_cfg_scale": ("FLOAT", {"default": 2.0,"min": 1.0, "step": 0.1}),
                "r_cfg_scale": ("FLOAT", {"default": 1.0,"min": 1.0, "step": 0.1}),
                "e_cfg_scale": ("FLOAT", {"default": 1.0,"min": 1.0, "step": 0.1}),
                "fps": ("FLOAT", {"default": 25, "step": 1}),
                "emotion": (['none', 'angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'], {"default": "none"}),
                "crop": ("BOOLEAN",{"default":False},),
                "seed": ("INT", {"default": 62064758300528, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "floatprocess"
    CATEGORY = "FLOAT"
    DESCRIPTION = "Float Processing"

    def floatprocess(self, ref_image, ref_audio, float_pipe, a_cfg_scale, r_cfg_scale, e_cfg_scale, fps, emotion, crop, seed):
        # save audio
        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)
        audio_save_path = os.path.join(temp_dir, f"{int(time.time())}.wav")
        torchaudio.save(audio_save_path, ref_audio['waveform'].squeeze(0), ref_audio["sample_rate"])
        
        # save image
        if ref_image.shape[0] != 1:
            raise Exception("Only a single image is supported.")
        ref_image_bchw = ref_image.permute(0, 3, 1, 2)
        image_save_path = os.path.join(temp_dir, f"{int(time.time())}.png")
        vutils.save_image(ref_image_bchw[0], image_save_path)
        
        # Move model to device only during inference
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        float_pipe.G.to(device)
        
        try:
            float_pipe.opt.fps = fps
            images_bhwc = float_pipe.run_inference(
                None,
                image_save_path,
                audio_save_path,
                a_cfg_scale = a_cfg_scale,
                r_cfg_scale = r_cfg_scale,
                e_cfg_scale = e_cfg_scale,
                emo 		= None if emotion == "none" else emotion,
                no_crop 	= not crop,
                seed 		= seed
            )
        finally:
            # Always move model back to CPU after inference
            float_pipe.G.to('cpu')
            # Clear any cached memory
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        return (images_bhwc,)

NODE_CLASS_MAPPINGS = {
    "LoadFloatModels": LoadFloatModels,
    "FloatProcess": FloatProcess
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadFloatModels": "Load FLOAT Models",
    "FloatProcess": "FLOAT Process"
}