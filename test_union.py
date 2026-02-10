import sys
import os
import torch
import numpy as np
from PIL import Image

# Add project root to path
sys.path.append(os.getcwd())

from posts.infrastructure import ai_models
ai_models.OPENPOSE_AVAILABLE = True

from posts.infrastructure.ai_models import AIModelManager
from posts.infrastructure.custom_pipeline import StableDiffusionXLUnionPipeline
from posts.domain.value_objects import DeviceConfig, ImageDimensions
from posts.config.ai_config import CONTROLNET_MODE_CANNY, CONTROLNET_MODE_OPENPOSE

def create_dummy_image(width=1024, height=1024):
    return Image.new("RGB", (width, height), (0, 0, 0))

def test_pipeline():
    print("=== Verifying ControlNet Union Implementation ===")
    
    device_config = DeviceConfig.from_cuda_availability()
    print(f"Device: {device_config.device}")
    
    # Test 1: Load with OpenPose (Double layer)
    print("\n--- Test 1: AIModelManager with use_openpose=True ---")
    manager = AIModelManager(device_config, use_openpose=True)
    
    try:
        pipe, compel = manager.load_models()
        
        print(f"Pipeline type: {type(pipe)}")
        if isinstance(pipe, StableDiffusionXLUnionPipeline):
            print("✅ Pipeline class correct")
        else:
            print("❌ Pipeline class mismatch")
            
        print(f"ControlNet type: {type(pipe.controlnet)}")
        import inspect
        print(f"ControlNet MRO: {inspect.getmro(type(pipe.controlnet))}")
        
        from diffusers.models import MultiControlNetModel as DMCM
        print(f"Is instance of diffusers.models.MultiControlNetModel? {isinstance(pipe.controlnet, DMCM)}")
        
        if hasattr(pipe.controlnet, 'nets'):
             print(f"Inner ControlNet type: {type(pipe.controlnet.nets[0])}")
        
        # Test Generation call with dummy data
        print("\n--- Test 2: Dry Run Generation (1 step) ---")
        
        prompt = "a photo of a cat"
        negative_prompt = "bad quality"
        
        # Mock embeddings
        conditioning, pooled = compel([prompt, negative_prompt])
        
        # Mock control images
        dummy_img = create_dummy_image()
        control_images = [dummy_img, dummy_img] # Canny, Pose
        control_modes = [CONTROLNET_MODE_CANNY, CONTROLNET_MODE_OPENPOSE] # [0, 4]
        
        print(f"Control Modes: {control_modes}")
        
        with torch.inference_mode():
             # We just want to check if it crashes during forward pass
             # We run just 1 step
             pipe(
                prompt_embeds=conditioning[0:1],
                negative_prompt_embeds=conditioning[1:2],
                pooled_prompt_embeds=pooled[0:1],
                negative_pooled_prompt_embeds=pooled[1:2],
                image=control_images,
                control_modes=control_modes,
                num_inference_steps=1,
                guidance_scale=1.0,
             )
        print("✅ Generation call finished without error")
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        manager.cleanup()
        print("Cleanup done")

if __name__ == "__main__":
    test_pipeline()
