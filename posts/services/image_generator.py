import os
import torch
from PIL import Image
from typing import Optional, List, Dict, Any, Union
import traceback
import time

from ..config import (
    DEFAULT_INFERENCE_STEPS,
    DEFAULT_CFG_SCALE,
    DEFAULT_DENOISE_STRENGTH,
    NEGATIVE_PROMPTS,
    CONTROLNET_MODE_CANNY,
    CONTROLNET_MODE_OPENPOSE,
)
from ..domain.entities import ImageAnalysis
from ..domain.value_objects import DeviceConfig
from ..infrastructure import (
    analyze_image_complexity,
    compute_adaptive_canny_thresholds,
    preprocess_image,
    make_canny_condition,
    make_openpose_condition,
    smart_resize,
    cleanup_gpu_memory,
    ai_pipeline_context,
    text_to_image_pipeline_context,
    create_hires_refiner,
    OPENPOSE_AVAILABLE,
    ADetailer,
)
from ..adapters.prompt_builder import build_enhanced_prompt
from ..use_cases.image_analysis import compute_optimal_strength


class ImageGeneratorService:
    @staticmethod
    def generate(
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
        strength: Optional[float] = None,
        use_adetailer: bool = True,
        adetailer_strength: float = 0.4,
    ) -> Dict[str, Any]:
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
        
        original_image = None
        preprocessed_image = None
        canny_image = None
        openpose_image = None
        control_images = []
        result = None
        
        response = {
            "success": False,
            "error": None,
            "was_auto_detected": False,
            "strength": strength_canny
        }

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
                if strength is not None:
                     strength_canny = strength
                     print(f"Using provided strength: {strength_canny:.3f}")
                else:
                    strength_val = compute_optimal_strength(image_analysis, style)
                    strength_canny = float(strength_val)
                    response["was_auto_detected"] = True
                    print(f"Auto-calculated control strength: {strength_canny:.2f}")
            else:
                print(f"Using manual ControlNet strength: {strength_canny:.3f}")
            
            response["strength"] = strength_canny

            canny_thresholds = compute_adaptive_canny_thresholds(preprocessed_image)
            canny_image = make_canny_condition(preprocessed_image, canny_thresholds)
            control_images = [canny_image]
            control_scales = [strength_canny]
            
            control_modes = [CONTROLNET_MODE_CANNY]
            
            if use_openpose and OPENPOSE_AVAILABLE:
                try:
                    openpose_image = make_openpose_condition(preprocessed_image)
                    if openpose_image:
                        control_images.append(openpose_image)
                        control_scales.append(strength_openpose)
                        control_modes.append(CONTROLNET_MODE_OPENPOSE)
                        print(f"OpenPose ControlNet strength: {strength_openpose:.3f}")
                    else:
                        raise ValueError("OpenPose returned None (no person detected)")
                except Exception as e:
                    print(f"OpenPose skipped: {e}")
                    # Fallback: maintain pipeline consistency
                    # The pipeline expects 2 images if initialized with use_openpose=True
                    # We provide a blank image with 0.0 scale
                    blank = Image.new("RGB", canny_image.size, (0, 0, 0))
                    control_images.append(blank)
                    control_scales.append(0.0)
                    control_modes.append(CONTROLNET_MODE_OPENPOSE)
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
                        control_modes=control_modes,
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
                
                if use_adetailer:
                    print(f"Applying ADetailer (strength: {adetailer_strength})...")
                    # We can use the same pipe mostly, but adetailer creates its own InpaintPipeline
                    # which reuses components. 
                    # We need to make sure we don't clear memory yet.
                    
                    # We need device config for ADetailer
                    # In this context, we can derive it or create new
                    # pipe.device is available
                    
                    # However, ADetailer class init expects DeviceConfig object.
                    # Let's create it.
                    device_config = DeviceConfig.from_cuda_availability()
                    adetailer = ADetailer(device_config)
                    
 
                    # Apply Enhanced ADetailer (Standard + Kentus + MonetEinsley)
                    result = ImageGeneratorService._apply_enhanced_adetailer(
                        adetailer=adetailer,
                        image=result,
                        base_pipe=pipe,
                        prompt=final_prompt,
                        negative_prompt=NEGATIVE_PROMPTS,
                        strength=adetailer_strength,
                        inference_steps=inference_steps
                    )
                    
                    # Clean up adetailer specific things if any (mostly handled by garbage collection unless we explicitly close things)

                    # ADetailer holds model ref, we might want to clear it if it consumes VRAM (the YOLO model)
                    if hasattr(adetailer, 'models'):
                        adetailer.models.clear()
                    del adetailer
                    cleanup_gpu_memory()

                conditioning = conditioning.cpu()
                pooled = pooled.cpu()
                del conditioning, pooled
                cleanup_gpu_memory()
            
            result.save(image_path, quality=98, optimize=True, subsampling=0)
            print("✅ Image generation completed successfully")
            
            response["success"] = True
            return response
        
        except Exception as e:
            print(f"❌ Error during generation: {e}")
            traceback.print_exc()
            response["error"] = str(e)
            return response
        
        finally:
            for obj in [original_image, preprocessed_image, canny_image, openpose_image, result]:
                if obj is not None:
                    del obj
            
            if control_images:
                control_images.clear()
            
            cleanup_gpu_memory()
            time.sleep(0.2)
            cleanup_gpu_memory()
            print("Memory cleanup completed")
    
    @staticmethod
    def generate_from_text(
        output_path: str,
        prompt: str,
        style: str = "auto",
        inference_steps: int = DEFAULT_INFERENCE_STEPS,
        cfg_scale: float = DEFAULT_CFG_SCALE,
        width: int = 1024,
        height: int = 1024,
        use_adetailer: bool = True,
        adetailer_strength: float = 0.4,
    ) -> Dict[str, Any]:
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
        
        result = None
        
        response = {
            "success": False,
            "error": None,
        }

        try:
            print(f"Text-to-Image generation")
            print(f"Style: {style} | Steps: {inference_steps} | CFG: {cfg_scale}")
            print(f"Size: {width}x{height}")
            
            final_prompt = build_enhanced_prompt(prompt, style, None)
            print(f"Enhanced prompt: {final_prompt[:150]}...")
            
            with text_to_image_pipeline_context() as (pipe, compel):
                device = pipe.device
                dtype = pipe.unet.dtype
                
                with torch.no_grad():
                    conditioning, pooled = compel([final_prompt, NEGATIVE_PROMPTS])
                    conditioning = conditioning.to(dtype)
                    pooled = pooled.to(dtype)
                
                print("Generating image with Z-Image from text...")
                with torch.inference_mode(), torch.no_grad():
                    latents = pipe(
                        prompt_embeds=conditioning[0:1],
                        negative_prompt_embeds=conditioning[1:2],
                        pooled_prompt_embeds=pooled[0:1],
                        negative_pooled_prompt_embeds=pooled[1:2],
                        guidance_scale=cfg_scale,
                        num_inference_steps=inference_steps,
                        width=width,
                        height=height,
                        output_type="latent",
                    ).images[0]
                    
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
                
                
                if use_adetailer:
                    print(f"Applying ADetailer (strength: {adetailer_strength})...")
                    device_config = DeviceConfig.from_cuda_availability()
                    adetailer = ADetailer(device_config)
                    
                    # Apply Enhanced ADetailer (Standard + Kentus + MonetEinsley)
                    result = ImageGeneratorService._apply_enhanced_adetailer(
                        adetailer=adetailer,
                        image=result,
                        base_pipe=pipe,
                        prompt=final_prompt,
                        negative_prompt=NEGATIVE_PROMPTS,
                        strength=adetailer_strength,
                        inference_steps=inference_steps
                    )
                    
                    if hasattr(adetailer, 'models'):
                        adetailer.models.clear()
                    del adetailer
                    cleanup_gpu_memory()

                conditioning = conditioning.cpu()
                pooled = pooled.cpu()
                del conditioning, pooled
                cleanup_gpu_memory()
            
            result.save(output_path, quality=98, optimize=True, subsampling=0)
            print("✅ Text-to-Image generation completed successfully")
            
            response["success"] = True
            return response
        
        except Exception as e:
            print(f"❌ Error during Text-to-Image generation: {e}")
            traceback.print_exc()
            response["error"] = str(e)
            return response
        
        finally:
            if result is not None:
                del result
            
            cleanup_gpu_memory()
            time.sleep(0.2)
            cleanup_gpu_memory()
            print("Memory cleanup completed")


    @staticmethod
    def _apply_enhanced_adetailer(
        adetailer: Any,
        image: Image.Image,
        base_pipe: Any,
        prompt: str,
        negative_prompt: str,
        strength: float,
        inference_steps: int
    ) -> Image.Image:
        """
        Helper to apply a sequence of ADetailer models.
        """
        steps = max(20, int(inference_steps * 0.6))
        
        # 1. Standard Person/Face/Hand (Bingsu/adetailer)
        standard_models = [
            ("person_yolov8m-seg.pt", "detailed person, high quality, realistic", "deformed, blurred, bad anatomy"),
            ("face_yolov8m.pt", "detailed face, high quality, realistic eyes, sharp focus", "bad eyes, deformed, blurred"),
            ("hand_yolov8n.pt", "detailed hands, anatomical, high quality", "malformed hands, extra fingers, missing fingers, bad anatomy"),
        ]
        
        for model_name, det_prompt, det_neg in standard_models:
            image = adetailer.apply_adetailer(
                image=image,
                base_pipe=base_pipe,
                prompt=prompt,
                negative_prompt=negative_prompt,
                strength=strength,
                steps=steps,
                model_name=model_name,
                repo_id="Bingsu/adetailer",
                detailer_prompt=det_prompt,
                detailer_negative_prompt=det_neg
            )

        # 2. Kentus/Adetailer Enhancement Parts
        kentus_models = [
            ("Eyeful_v1.pt", "detailed eyes, realistic iris, sharp focus", "bad eyes, blurred", 0.35),
            ("lips_v1.pt", "detailed lips, realistic mouth", "blurred, deformed", 0.3),
            ("female_breast_v3.pt", "detailed breasts, anatomy", "deformed, blurred", 0.35),
            ("belly_seg_v1.pt", "detailed belly, navel, anatomy", "deformed", 0.3),
            # NSFW / Specific parts (Use with caution or conditionally, enabling for now as requested)
            ("vagina_v2.5.pt", "explicit, anatomy", "deformed, blurred", 0.35),
            ("penis_V2.pt", "explicit, anatomy", "deformed, blurred", 0.35),
            ("assdetailer-seg.pt", "detailed buttocks, anatomy", "deformed", 0.35),
        ]
        
        for model_name, det_prompt, det_neg, model_strength in kentus_models:
             image = adetailer.apply_adetailer(
                image=image,
                base_pipe=base_pipe,
                prompt=prompt,
                negative_prompt=negative_prompt,
                strength=model_strength if model_strength else strength,
                steps=steps,
                model_name=model_name,
                repo_id="Kentus/Adetailer",
                detailer_prompt=det_prompt,
                detailer_negative_prompt=det_neg
            )
            
        # 3. MonetEinsley/ADetailer_CM (Feet)
        image = adetailer.apply_adetailer(
            image=image,
            base_pipe=base_pipe,
            prompt=prompt,
            negative_prompt=negative_prompt,
            strength=0.35, # Feet often need careful handling
            steps=steps,
            model_name="foot_yolov8x_v2.pt",
            repo_id="MonetEinsley/ADetailer_CM",
            detailer_prompt="detailed feet, toes, realistic",
            detailer_negative_prompt="malformed feet, extra toes, missing toes, bad anatomy"
        )
        
        return image
