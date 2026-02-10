import torch
import time
from contextlib import contextmanager
from typing import Optional, Tuple, Any

from diffusers import (
    ControlNetModel,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    DPMSolverMultistepScheduler,
    AutoencoderKL,
)
from compel import Compel, ReturnedEmbeddingsType
from huggingface_hub.utils import LocalEntryNotFoundError

from ..config import (
    BASE_MODEL_ID,
    VAE_ID,
    CONTROLNET_UNION_ID,
    HIRES_SCALE,
)
from ..domain.value_objects import DeviceConfig, ImageDimensions
from .memory_manager import cleanup_gpu_memory
from .image_processing import OPENPOSE_AVAILABLE
from .custom_controlnet import UnionMultiControlNetModel
from .custom_pipeline import StableDiffusionXLUnionPipeline


def _load_pretrained(model_class, model_id, **kwargs):
    """
    Helper to load a model from local cache first, falling back to download/check
    if not found or incomplete.
    """
    try:
        # Try loading with local_files_only=True
        return model_class.from_pretrained(model_id, local_files_only=True, **kwargs)
    except (OSError, EnvironmentError, LocalEntryNotFoundError):
        # Fallback to standard loading (will check/download)
        print(f"Model {model_id} not found locally or incomplete. Downloading/Checking...")
        return model_class.from_pretrained(model_id, local_files_only=False, **kwargs)



class AIModelManager:
    def __init__(self, device_config: DeviceConfig, use_openpose: bool = False):
        self.device_config = device_config
        self.use_openpose = use_openpose and OPENPOSE_AVAILABLE
        self.pipe = None
        self.compel = None
        self.controlnets = []
        self.vae = None
    
    def load_models(self) -> Tuple[Any, Any]:
        print("Initializing AI pipeline with maximum quality settings...")
        
        cleanup_gpu_memory()
        
        # Load the single Union ControlNet model
        controlnet_union = _load_pretrained(
            ControlNetModel,
            CONTROLNET_UNION_ID,
            torch_dtype=self.device_config.dtype,
            use_safetensors=True,
            low_cpu_mem_usage=True
        )
        
        # Prepare the controlnet(s) for the pipeline
        if self.use_openpose:
            # If using both Canny (implied as base) and OpenPose, we need two "instances"
            # We use the custom MultiControlNet to wrap the SAME model execution twice
            # This allows passing different class_labels (modes) to each "layer"
            # Note: We pass the same model instance to save VRAM. 
            self.controlnets = UnionMultiControlNetModel([controlnet_union, controlnet_union])
            # self.controlnets will be used in pipeline init
        else:
            # Just single usage (Canny)
            self.controlnets = controlnet_union
        
        self.vae = _load_pretrained(
            AutoencoderKL,
            VAE_ID,
            torch_dtype=self.device_config.dtype,
            use_safetensors=True,
            low_cpu_mem_usage=True
        )
        
        self.pipe = _load_pretrained(
            StableDiffusionXLUnionPipeline,
            BASE_MODEL_ID,
            controlnet=self.controlnets,
            vae=self.vae,
            torch_dtype=self.device_config.dtype,
            use_safetensors=True,
            variant="fp16",
            add_watermarker=False,
        )
        
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            use_karras_sigmas=True,
            algorithm_type="dpmsolver++",
            solver_order=2
        )
        
        self.pipe.vae.enable_tiling()
        self.pipe.vae.enable_slicing()
        self.pipe.vae.config.force_upcast = False
        
        if torch.cuda.is_available():
            self.pipe.to(self.device_config.device)
            self.pipe.unet.to(memory_format=torch.channels_last)
            self.pipe.enable_attention_slicing(slice_size=1)
            self.pipe.enable_vae_slicing()
            
            self.compel = Compel(
                tokenizer=[self.pipe.tokenizer, self.pipe.tokenizer_2],
                text_encoder=[self.pipe.text_encoder, self.pipe.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True],
                device=self.device_config.device
            )
            
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                print("xFormers memory efficient attention enabled")
            except Exception:
                print("xFormers not available, using standard attention")
        else:
            self.compel = Compel(
                tokenizer=[self.pipe.tokenizer, self.pipe.tokenizer_2],
                text_encoder=[self.pipe.text_encoder, self.pipe.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True],
            )
        
        print(f"Pipeline ready | Device: {self.device_config.device} | Maximum quality mode active")
        return self.pipe, self.compel
    
    def cleanup(self) -> None:
        print("Unloading AI pipeline...")
        
        if self.compel is not None:
            del self.compel
        
        if self.pipe is not None:
            components = ['controlnet', 'unet', 'vae', 'text_encoder', 'text_encoder_2']
            for comp_name in components:
                if hasattr(self.pipe, comp_name):
                    comp = getattr(self.pipe, comp_name)
                    if comp is not None:
                        if isinstance(comp, list):
                            for item in comp:
                                if item is not None and hasattr(item, 'to'):
                                    item.to('cpu')
                                del item
                        else:
                            if hasattr(comp, 'to'):
                                comp.to('cpu')
                            del comp
                        setattr(self.pipe, comp_name, None)
            del self.pipe
        
        # Helper to clear a single model or list/container of models
        def clear_model(m):
            if m is not None:
                if isinstance(m, (list, tuple)):
                    for item in m:
                        clear_model(item)
                elif hasattr(m, 'nets'): # MultiControlNet
                     for item in m.nets:
                         clear_model(item)
                else:
                    if hasattr(m, 'to'):
                        m.to('cpu')
                    del m

        # Clear controlnets
        if hasattr(self, 'controlnets'):
             clear_model(self.controlnets)
             if isinstance(self.controlnets, list):
                 self.controlnets.clear()
             else:
                 self.controlnets = []
        
        if self.vae is not None:
            if hasattr(self.vae, 'to'):
                self.vae.to('cpu')
            del self.vae
        
        cleanup_gpu_memory()
        time.sleep(0.25)
        cleanup_gpu_memory()
        print("Pipeline fully unloaded and memory cleared")


@contextmanager
def ai_pipeline_context(use_openpose: bool = False):
    device_config = DeviceConfig.from_cuda_availability()
    manager = AIModelManager(device_config, use_openpose)
    
    try:
        pipe, compel = manager.load_models()
        yield pipe, compel
    finally:
        manager.cleanup()

class TextToImageModelManager:
    def __init__(self, device_config: DeviceConfig):
        self.device_config = device_config
        self.pipe = None
        self.compel = None
        self.vae = None
    
    def load_models(self) -> Tuple[Any, Any]:
        print("Initializing Text-to-Image pipeline...")
        
        cleanup_gpu_memory()
        
        self.vae = _load_pretrained(
            AutoencoderKL,
            VAE_ID,
            torch_dtype=self.device_config.dtype,
            use_safetensors=True,
            low_cpu_mem_usage=True
        )
        
        self.pipe = _load_pretrained(
            StableDiffusionXLPipeline,
            BASE_MODEL_ID,
            vae=self.vae,
            torch_dtype=self.device_config.dtype,
            use_safetensors=True,
            variant="fp16",
            add_watermarker=False,
        )
        
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            use_karras_sigmas=True,
            algorithm_type="dpmsolver++",
            solver_order=2
        )
        
        self.pipe.vae.enable_tiling()
        self.pipe.vae.enable_slicing()
        self.pipe.vae.config.force_upcast = False
        
        if torch.cuda.is_available():
            self.pipe.to(self.device_config.device)
            self.pipe.unet.to(memory_format=torch.channels_last)
            self.pipe.enable_attention_slicing(slice_size=1)
            self.pipe.enable_vae_slicing()
            
            self.compel = Compel(
                tokenizer=[self.pipe.tokenizer, self.pipe.tokenizer_2],
                text_encoder=[self.pipe.text_encoder, self.pipe.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True],
                device=self.device_config.device
            )
            
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                print("xFormers memory efficient attention enabled")
            except Exception:
                print("xFormers not available, using standard attention")
        else:
            self.compel = Compel(
                tokenizer=[self.pipe.tokenizer, self.pipe.tokenizer_2],
                text_encoder=[self.pipe.text_encoder, self.pipe.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True],
            )
        
        print(f"Text-to-Image pipeline ready | Device: {self.device_config.device}")
        return self.pipe, self.compel
    
    def cleanup(self) -> None:
        print("Unloading Text-to-Image pipeline...")
        
        if self.compel is not None:
            del self.compel
        
        if self.pipe is not None:
            components = ['unet', 'vae', 'text_encoder', 'text_encoder_2']
            for comp_name in components:
                if hasattr(self.pipe, comp_name):
                    comp = getattr(self.pipe, comp_name)
                    if comp is not None:
                        if hasattr(comp, 'to'):
                            comp.to('cpu')
                        del comp
                    setattr(self.pipe, comp_name, None)
            del self.pipe
        
        if self.vae is not None:
            if hasattr(self.vae, 'to'):
                self.vae.to('cpu')
            del self.vae
        
        cleanup_gpu_memory()
        time.sleep(0.25)
        cleanup_gpu_memory()
        print("Text-to-Image pipeline fully unloaded and memory cleared")
@contextmanager
def text_to_image_pipeline_context():
    device_config = DeviceConfig.from_cuda_availability()
    manager = TextToImageModelManager(device_config)
    
    try:
        pipe, compel = manager.load_models()
        yield pipe, compel
    finally:
        manager.cleanup()


def create_hires_refiner(
    pipe: Any,
    image: Any,
    prompt_embeds: torch.Tensor,
    pooled_embeds: torch.Tensor,
    negative_embeds: torch.Tensor,
    negative_pooled: torch.Tensor,
    steps: int,
    cfg: float,
    denoise: float
) -> Any:
    dimensions = ImageDimensions(width=image.size[0], height=image.size[1])
    target_dims = dimensions.scale(HIRES_SCALE).align_to_multiple(64)
    
    cleanup_gpu_memory()
    
    refiner = StableDiffusionXLImg2ImgPipeline(
        vae=pipe.vae,
        text_encoder=pipe.text_encoder,
        text_encoder_2=pipe.text_encoder_2,
        tokenizer=pipe.tokenizer,
        tokenizer_2=pipe.tokenizer_2,
        unet=pipe.unet,
        scheduler=pipe.scheduler,
    )
    
    if torch.cuda.is_available():
        refiner.enable_attention_slicing(slice_size=1)
        refiner.enable_vae_slicing()
        try:
            refiner.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    
    refiner.vae.config.force_upcast = False
    
    hires_steps = max(steps // 2, 20)
    
    with torch.inference_mode():
        refined = refiner(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_embeds,
            negative_prompt_embeds=negative_embeds,
            negative_pooled_prompt_embeds=negative_pooled,
            image=image,
            strength=denoise,
            num_inference_steps=hires_steps,
            guidance_scale=cfg,
            width=target_dims.width,
            height=target_dims.height,
        ).images[0]
    
    del refiner
    cleanup_gpu_memory()
    
    return refined
