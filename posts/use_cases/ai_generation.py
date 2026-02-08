import os
import torch
from PIL import Image
from typing import Optional, List
import traceback

from ..config import DEFAULT_INFERENCE_STEPS, DEFAULT_CFG_SCALE, DEFAULT_DENOISE_STRENGTH, NEGATIVE_PROMPTS
from ..domain.entities import ImageAnalysis
from ..infrastructure import (
    analyze_image_complexity,
    compute_adaptive_canny_thresholds,
    preprocess_image,
    make_canny_condition,
    make_openpose_condition,
    smart_resize,
    cleanup_gpu_memory,
    ai_pipeline_context,
    create_hires_refiner,
    OPENPOSE_AVAILABLE,
)
from ..adapters.prompt_builder import build_enhanced_prompt
from .image_analysis import compute_optimal_strength


def generate_ai_image(
    image_path: str,
    prompt: str,
    style: str = "auto",
    strength_canny: Optional[float] = None,
    strength_openpose: float = 0.35,
    use_openpose: bool = True,
    use_hires: bool = False,
    inference_steps: int = DEFAULT_INFERENCE_STEPS,
    cfg_scale: float = DEFAULT_CFG_SCALE,
    denoise_strength: float = DEFAULT_DENOISE_STRENGTH,
    strength: Optional[float] = None
) -> bool:
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    
    original_image = None
    preprocessed_image = None
    canny_image = None
    openpose_image = None
    control_images = []
    result = None
    
    try:
        print(f"Processing: {image_path}")
        print(f"Style: {style} | Steps: {inference_steps} | CFG: {cfg_scale}")
        print(f"Max resolution: 1024px | Hires: {use_hires}")
        
        original_image = Image.open(image_path).convert("RGB")
        original_image = smart_resize(original_image)
        
        image_analysis = analyze_image_complexity(original_image)
        preprocessed_image = preprocess_image(original_image, image_analysis)
        
        print(f"Image analysis: complexity={image_analysis.is_complex}, details={image_analysis.has_fine_details}, entropy={image_analysis.entropy:.2f}")
        
        if strength_canny is None:
            strength_value = compute_optimal_strength(image_analysis, style)
            strength_canny = float(strength_value)
        else:
            print(f"Using manual ControlNet strength: {strength_canny:.3f}")
        
        canny_thresholds = compute_adaptive_canny_thresholds(preprocessed_image)
        canny_image = make_canny_condition(preprocessed_image, canny_thresholds)
        control_images = [canny_image]
        control_scales = [strength_canny]
        
        if use_openpose and OPENPOSE_AVAILABLE:
            try:
                openpose_image = make_openpose_condition(preprocessed_image)
                if openpose_image:
                    control_images.append(openpose_image)
                    control_scales.append(strength_openpose)
                    print(f"OpenPose ControlNet strength: {strength_openpose:.3f}")
            except Exception as e:
                print(f"OpenPose skipped: {e}")
        elif use_openpose and not OPENPOSE_AVAILABLE:
            print("OpenPose unavailable (dependencies missing)")
        
        final_prompt = build_enhanced_prompt(prompt, style, image_analysis)
        print(f"Enhanced prompt: {final_prompt[:150]}...")
        
        final_control_scale = control_scales if len(control_scales) > 1 else control_scales[0]
        
        with ai_pipeline_context(use_openpose=use_openpose and OPENPOSE_AVAILABLE) as (pipe, compel):
            device = pipe.device
            dtype = pipe.unet.dtype
            
            with torch.no_grad():
                conditioning, pooled = compel([final_prompt, NEGATIVE_PROMPTS])
                conditioning = conditioning.to(dtype)
                pooled = pooled.to(dtype)
            
            print("Generating image with maximum quality settings...")
            with torch.inference_mode(), torch.no_grad():
                latents = pipe(
                    prompt_embeds=conditioning[0:1],
                    negative_prompt_embeds=conditioning[1:2],
                    pooled_prompt_embeds=pooled[0:1],
                    negative_pooled_prompt_embeds=pooled[1:2],
                    image=control_images,
                    controlnet_conditioning_scale=final_control_scale,
                    guidance_scale=cfg_scale,
                    num_inference_steps=inference_steps,
                    output_type="latent",
                ).images[0]
                
                del control_images[:]
                del control_scales
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                cleanup_gpu_memory()
                
                print("Decoding latents to image...")
                latents = latents.unsqueeze(0).to(pipe.vae.dtype)
                image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
                
                del latents
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                cleanup_gpu_memory()
                
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.cpu().permute(0, 2, 3, 1).float().numpy()
                result = Image.fromarray((image[0] * 255).round().astype("uint8"))
                
                del image
                cleanup_gpu_memory()
            
            if use_hires:
                print(f"Applying hires fix (scale: 2.0x)...")
                with torch.no_grad():
                    conditioning_hires, pooled_hires = compel([final_prompt, NEGATIVE_PROMPTS])
                    conditioning_hires = conditioning_hires.to(dtype)
                    pooled_hires = pooled_hires.to(dtype)
                
                result = create_hires_refiner(
                    pipe,
                    result,
                    conditioning_hires[0:1],
                    pooled_hires[0:1],
                    conditioning_hires[1:2],
                    pooled_hires[1:2],
                    inference_steps,
                    cfg_scale,
                    denoise_strength
                )
                
                conditioning_hires = conditioning_hires.cpu()
                pooled_hires = pooled_hires.cpu()
                del conditioning_hires, pooled_hires
                cleanup_gpu_memory()
            
            conditioning = conditioning.cpu()
            pooled = pooled.cpu()
            del conditioning, pooled
            cleanup_gpu_memory()
        
        result.save(image_path, quality=98, optimize=True, subsampling=0)
        print("✅ Image generation completed successfully")
        return True
    
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        traceback.print_exc()
        return False
    
    finally:
        for obj in [original_image, preprocessed_image, canny_image, openpose_image, result]:
            if obj is not None:
                del obj
        
        if control_images:
            control_images.clear()
        
        cleanup_gpu_memory()
        import time
        time.sleep(0.2)
        cleanup_gpu_memory()
        print("Memory cleanup completed")
