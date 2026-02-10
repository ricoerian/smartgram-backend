import torch
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from typing import List, Tuple, Dict, Any, Optional
import os
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

from diffusers import StableDiffusionXLInpaintPipeline
from diffusers.utils import load_image

from ..config import BASE_MODEL_ID, VAE_ID
from ..domain.value_objects import DeviceConfig

class ADetailer:
    def __init__(self, device_config: DeviceConfig):
        self.device_config = device_config
        self.models = {}
        self.inpaint_pipe = None
        
    def _load_yolo_model(self, model_name: str = "face_yolov8n.pt", repo_id: str = "Bingsu/adetailer"):
        # Create a unique key for the model based on name and repo
        model_key = f"{repo_id}/{model_name}"
        if model_key in self.models:
            return self.models[model_key]
            
        print(f"Loading ADetailer model: {model_name} from {repo_id}")
        try:
            # First try to load from local ultralytics cache or download if needed
            # We can use hf_hub_download to get the path
            model_path = hf_hub_download(repo_id=repo_id, filename=model_name)
            model = YOLO(model_path)
            self.models[model_key] = model
            return model
        except Exception as e:
            print(f"Failed to load ADetailer model {model_name} from {repo_id}: {e}")
            return None

    def _get_inpaint_pipeline(self, base_pipe):
        if self.inpaint_pipe is not None:
             return self.inpaint_pipe
             
        print("Initializing Inpaint pipeline for ADetailer...")
        
        # Reuse components from the main pipeline to save VRAM
        # We need to ensure we don't accidentally move things to CPU if base_pipe is using them
        
        components = {
            "vae": base_pipe.vae,
            "text_encoder": base_pipe.text_encoder,
            "text_encoder_2": base_pipe.text_encoder_2,
            "tokenizer": base_pipe.tokenizer,
            "tokenizer_2": base_pipe.tokenizer_2,
            "unet": base_pipe.unet,
            "scheduler": base_pipe.scheduler,
        }
        
        self.inpaint_pipe = StableDiffusionXLInpaintPipeline(**components)
        
        if torch.cuda.is_available():
            self.inpaint_pipe.to(self.device_config.device)
            # Enable memory efficient attention if available
            try:
                self.inpaint_pipe.enable_xformers_memory_efficient_attention()
            except:
                pass
                
        return self.inpaint_pipe

    def detect_people(self, image: Image.Image, confidence: float = 0.3) -> List[Dict[str, Any]]:
        """
        Convenience method to detect people using the preferred model.
        """
        return self.detect_objects(image, model_name="person_yolov8m-seg.pt", confidence=confidence)

    def detect_objects(self, image: Image.Image, model_name: str = "face_yolov8n.pt", confidence: float = 0.3, repo_id: str = "Bingsu/adetailer") -> List[Dict[str, Any]]:
        model = self._load_yolo_model(model_name, repo_id=repo_id)
        if not model:
            return []
            
        # Run inference
        results = model(image, conf=confidence)
        detections = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0].cpu().numpy() # [x1, y1, x2, y2]
                detections.append({
                    "box": b,
                    "confidence": float(box.conf.cpu().numpy()[0]),
                    "label": int(box.cls.cpu().numpy()[0])
                })
                
        return detections

    def _create_soft_mask(self, width: int, height: int, padding: int = 8) -> Image.Image:
        """
        Creates a soft mask for blending.
        """
        mask = Image.new("L", (width, height), 0)
        import PIL.ImageDraw as ImageDraw
        draw = ImageDraw.Draw(mask)
        
        # Draw a filled ellipse or rounded rectangle for better blending (standard ADetailer practice)
        # Using a rectangle with heavy blur is often safer than ellipse for general bounding boxes
        draw.rectangle([padding, padding, width - padding, height - padding], fill=255)
        
        # Apply gaussian blur to soften edges
        mask = mask.filter(ImageFilter.GaussianBlur(10))
        return mask

    def apply_adetailer(
        self, 
        image: Image.Image, 
        base_pipe: Any, 
        prompt: str, 
        negative_prompt: str,
        strength: float = 0.4, 
        steps: int = 20,
        model_name: str = "face_yolov8n.pt",
        repo_id: str = "Bingsu/adetailer",
        detailer_prompt: str = "",
        detailer_negative_prompt: str = ""
    ) -> Image.Image:
        """
        Applies ADetailer to the given image.
        """
        print(f"Starting ADetailer process with model: {model_name} from {repo_id}")
        detections = self.detect_objects(image, model_name, repo_id=repo_id)
        
        if not detections:
            print(f"No objects detected for ADetailer ({model_name}).")
            return image
            
        print(f"ADetailer: Found {len(detections)} objects.")
        
        # Setup inpaint pipeline
        pipe = self._get_inpaint_pipeline(base_pipe)
        
        result_image = image.copy()
        width, height = result_image.size
        
        # Construct effective prompts
        effective_prompt = prompt
        if detailer_prompt:
             effective_prompt = f"{detailer_prompt}, {prompt}"
        
        effective_negative_prompt = negative_prompt
        if detailer_negative_prompt:
             effective_negative_prompt = f"{detailer_negative_prompt}, {negative_prompt}"

        # Sort detections by size (largest first) to handle overlaps better
        detections.sort(key=lambda x: (x["box"][2] - x["box"][0]) * (x["box"][3] - x["box"][1]), reverse=True)

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = map(int, det["box"])
            
            # Add padding
            padding = 32
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(width, x2 + padding)
            y2 = min(height, y2 + padding)
            
            crop_width = x2 - x1
            crop_height = y2 - y1
            
            # Crop
            crop = result_image.crop((x1, y1, x2, y2))
            
            # Create soft mask
            mask = self._create_soft_mask(crop_width, crop_height)
            
            print(f"Processing object {i+1}/{len(detections)}: box=[{x1}, {y1}, {x2}, {y2}]")

            # ADetailer logic: resize small faces to adequate resolution for inpainting (usually 512-768ish)
            target_size = 768
            original_crop_size = crop.size
            
            need_resize = max(original_crop_size) < target_size
            if need_resize:
                 scale_factor = target_size / max(original_crop_size)
                 new_w = int(original_crop_size[0] * scale_factor)
                 new_h = int(original_crop_size[1] * scale_factor)
                 new_w = new_w - (new_w % 8)
                 new_h = new_h - (new_h % 8)
                 crop_resized = crop.resize((new_w, new_h), Image.LANCZOS)
                 mask_resized = mask.resize((new_w, new_h), Image.LANCZOS)
            else:
                 crop_resized = crop
                 mask_resized = mask
                 w, h = crop_resized.size
                 w = w - (w % 8)
                 h = h - (h % 8)
                 if w != crop_resized.size[0] or h != crop_resized.size[1]:
                    crop_resized = crop_resized.crop((0,0,w,h))
                    mask_resized = mask_resized.crop((0,0,w,h))
            
            generator = torch.Generator(device=pipe.device).manual_seed(42) # optional seed
            
            with torch.inference_mode():
                # Inpainting
                inpainted = pipe(
                    prompt=effective_prompt,
                    negative_prompt=effective_negative_prompt,
                    image=crop_resized,
                    mask_image=mask_resized,
                    num_inference_steps=steps,
                    strength=strength, 
                    guidance_scale=7.5,
                    output_type="pil",
                ).images[0]
            
            # Resize back to original crop size (to handle both scaling and modulo-8 cropping)
            if inpainted.size != original_crop_size:
                inpainted = inpainted.resize(original_crop_size, Image.LANCZOS)
            
            # Paste back using the mask for blending
            result_image.paste(inpainted, (x1, y1), mask=mask)
            
        return result_image
