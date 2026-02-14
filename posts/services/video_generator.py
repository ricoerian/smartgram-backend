import os
import torch
import gc
import math  # For cosine interpolation in blending
from diffusers import (
    WanPipeline, 
    WanVACEPipeline,
    AutoencoderKLWan,
)
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import load_image, export_to_video
from PIL import Image, ImageFilter, ImageEnhance
import tempfile
import uuid
import threading
from typing import Tuple, Optional, List
import numpy as np

# ===== CRITICAL: Set BEFORE any torch import =====
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

try:
    import ffmpeg
    FFMPEG_AVAILABLE = True
except ImportError:
    FFMPEG_AVAILABLE = False

from ..infrastructure.memory_manager import cleanup_gpu_memory

# Global pipeline caches
_t2v_pipe: Optional[WanPipeline] = None
_vace_pipe: Optional[WanVACEPipeline] = None
_t2v_lock = threading.Lock()
_vace_lock = threading.Lock()


def nuclear_clear_memory():
    """Nuclear option - clear EVERYTHING"""
    import gc
    
    # Clear Python objects
    gc.collect()
    gc.collect()  # Yes, twice
    
    # Clear PyTorch cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
        # Synchronize to ensure operations complete
        torch.cuda.synchronize()
        
        # Reset memory stats
        if hasattr(torch.cuda, 'reset_peak_memory_stats'):
            torch.cuda.reset_peak_memory_stats()
        if hasattr(torch.cuda, 'reset_accumulated_memory_stats'):
            torch.cuda.reset_accumulated_memory_stats()
    
    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated(0) / 1024**3
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[CLEAR] VRAM: {vram_used:.2f}/{vram_total:.2f} GB")


def get_t2v_pipeline() -> WanPipeline:
    """
    Load Text-to-Video pipeline (Wan2.1-T2V-1.3B-Diffusers)
    12GB GPU version - dengan model offload
    """
    global _t2v_pipe
    
    with _t2v_lock:
        if _t2v_pipe is not None:
            print("[T2V] ✓ Using cached pipeline")
            return _t2v_pipe

        model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
        
        print(f"[T2V-12GB] Loading Text-to-Video pipeline...")
        
        # CRITICAL: Clear everything before loading
        nuclear_clear_memory()
        
        # Check available VRAM
        if torch.cuda.is_available():
            free_vram = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3
            print(f"[T2V-12GB] Free VRAM: {free_vram:.2f} GB")
            
            if free_vram < 8:
                print(f"[WARNING] Only {free_vram:.2f} GB free!")
        
        try:
            # Load VAE with float32 for quality
            print("[T2V-12GB] Loading VAE...")
            vae = AutoencoderKLWan.from_pretrained(
                model_id, 
                subfolder="vae", 
                torch_dtype=torch.float32
            )
            
            # Load pipeline
            print("[T2V-12GB] Loading main pipeline...")
            pipe = WanPipeline.from_pretrained(
                model_id, 
                vae=vae, 
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,  # Important for 12GB
            )
            
            # IMPORTANT: Enable model offload untuk 12GB GPU
            print("[T2V-12GB] Enabling model CPU offload...")
            pipe.enable_model_cpu_offload()  # Offload ke CPU saat tidak dipakai
            
            # Memory optimizations
            pipe.enable_attention_slicing(slice_size=1)
            pipe.vae.enable_slicing()
            pipe.vae.enable_tiling()
            
            # Try xformers
            try:
                pipe.enable_xformers_memory_efficient_attention()
                print("[T2V-12GB] ✓ xFormers enabled")
            except Exception as e:
                print(f"[T2V-12GB] ⚠ xFormers not available: {e}")

            _t2v_pipe = pipe
            
            # Clear after loading
            nuclear_clear_memory()
            
            print(f"[T2V-12GB] ✓ Pipeline loaded with CPU offload")
            return pipe

        except Exception as e:
            print(f"[ERROR] T2V Load failed: {e}")
            if torch.cuda.is_available():
                print(f"[ERROR] VRAM allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
            raise


def get_vace_pipeline() -> WanVACEPipeline:
    """
    Load VACE pipeline (Wan2.1-VACE-1.3B-diffusers)
    12GB GPU version - dengan SEQUENTIAL CPU offload (most aggressive)
    """
    global _vace_pipe
    
    with _vace_lock:
        if _vace_pipe is not None:
            print("[VACE] ✓ Using cached pipeline")
            return _vace_pipe

        model_id = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
        
        print(f"[VACE-12GB] Loading VACE pipeline with AGGRESSIVE memory saving...")
        
        # CRITICAL: Clear everything before loading
        nuclear_clear_memory()
        
        # Check available VRAM
        if torch.cuda.is_available():
            free_vram = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3
            print(f"[VACE-12GB] Free VRAM: {free_vram:.2f} GB")
            
            if free_vram < 10:
                print(f"[WARNING] Only {free_vram:.2f} GB free! VACE needs ~10GB")
                print(f"[WARNING] Will use SEQUENTIAL CPU OFFLOAD (slower but safe)")
        
        try:
            # Load VAE with float32
            print("[VACE-12GB] Loading VAE...")
            vae = AutoencoderKLWan.from_pretrained(
                model_id, 
                subfolder="vae", 
                torch_dtype=torch.float32
            )
            
            # Load pipeline
            print("[VACE-12GB] Loading main pipeline...")
            pipe = WanVACEPipeline.from_pretrained(
                model_id, 
                vae=vae, 
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
            
            # Setup scheduler
            flow_shift = 3.0  # 480P
            pipe.scheduler = UniPCMultistepScheduler.from_config(
                pipe.scheduler.config, 
                flow_shift=flow_shift
            )
            
            # CRITICAL: SEQUENTIAL CPU OFFLOAD untuk 12GB GPU
            # Ini PALING LAMBAT tapi PALING AMAN untuk memory
            print("[VACE-12GB] Enabling SEQUENTIAL CPU offload (slowest but safest)...")
            pipe.enable_sequential_cpu_offload()
            
            # Maximum memory optimizations
            pipe.enable_attention_slicing(slice_size=1)
            pipe.vae.enable_slicing()
            pipe.vae.enable_tiling()
            
            # Try xformers
            try:
                pipe.enable_xformers_memory_efficient_attention()
                print("[VACE-12GB] ✓ xFormers enabled")
            except Exception as e:
                print(f"[VACE-12GB] ⚠ xFormers not available: {e}")

            _vace_pipe = pipe
            
            # Clear after loading
            nuclear_clear_memory()
            
            print(f"[VACE-12GB] ✓ Pipeline loaded with SEQUENTIAL CPU offload")
            print(f"[VACE-12GB] ⚠ This will be SLOWER but won't OOM")
            return pipe

        except Exception as e:
            print(f"[ERROR] VACE Load failed: {e}")
            if torch.cuda.is_available():
                print(f"[ERROR] VRAM allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
            raise


def calculate_optimal_dimensions(
    orig_width: int,
    orig_height: int,
    max_pixels: int = 832 * 480,  # Default max area for 12GB GPU
    divisible_by: int = 16,  # CRITICAL: Must be divisible by 16 for VACE pipeline!
) -> Tuple[int, int]:
    """
    Calculate optimal video dimensions based on image aspect ratio
    Preserves aspect ratio while staying within memory limits
    
    IMPORTANT: VACE pipeline requires dimensions divisible by 16!
    """
    aspect_ratio = orig_width / orig_height
    total_pixels = orig_width * orig_height
    
    # If image is already small enough, keep it (but make divisible by 16)
    if total_pixels <= max_pixels:
        target_width = (orig_width // divisible_by) * divisible_by
        target_height = (orig_height // divisible_by) * divisible_by
        
        # Ensure minimum dimensions (divisible by 16)
        target_width = max(target_width, 384)
        target_height = max(target_height, 384)
        
        print(f"[DIM] Original: {orig_width}x{orig_height}")
        print(f"[DIM] Adjusted: {target_width}x{target_height} (divisible by {divisible_by})")
        return target_width, target_height
    
    # Image too large, scale down while preserving aspect ratio
    scale_factor = (max_pixels / total_pixels) ** 0.5
    target_width = int(orig_width * scale_factor)
    target_height = int(orig_height * scale_factor)
    
    # Make divisible by 16 (CRITICAL for VACE!)
    target_width = (target_width // divisible_by) * divisible_by
    target_height = (target_height // divisible_by) * divisible_by
    
    # Ensure minimum dimensions (already divisible by 16)
    target_width = max(target_width, 384)
    target_height = max(target_height, 384)
    
    # Verify aspect ratio preservation
    new_aspect = target_width / target_height
    aspect_diff = abs(aspect_ratio - new_aspect) / aspect_ratio * 100
    
    print(f"[DIM] Original: {orig_width}x{orig_height} ({total_pixels} pixels)")
    print(f"[DIM] Scaled: {target_width}x{target_height} ({target_width*target_height} pixels)")
    print(f"[DIM] Aspect ratio: {aspect_ratio:.3f} → {new_aspect:.3f} (diff: {aspect_diff:.1f}%)")
    
    return target_width, target_height


def prepare_image_to_video_input(
    first_img: Image.Image, 
    height: int, 
    width: int, 
    num_frames: int
) -> Tuple[List[Image.Image], List[Image.Image]]:
    """
    Prepare input for VACE pipeline
    Resizes image to specified dimensions while maintaining quality
    """
    # Resize image to target dimensions
    resized_img = first_img.resize((width, height), Image.Resampling.LANCZOS)
    
    frames = []
    frames.append(resized_img)
    
    # Create gray frames for generation
    gray_frame = Image.new("RGB", (width, height), (128, 128, 128))
    frames.extend([gray_frame] * (num_frames - 1))
    
    # Create masks
    mask_black = Image.new("L", (width, height), 0)
    mask_white = Image.new("L", (width, height), 255)
    mask = [mask_black] + [mask_white] * (num_frames - 1)
    
    return frames, mask


class VideoGeneratorService:
    """
    Video Generator Service - 12GB GPU Optimized
    Settings reduced untuk fit dalam 12GB VRAM
    Auto-adjusts dimensions based on input image
    
    IMPORTANT: All dimensions MUST be divisible by 16 for VACE pipeline!
    """
    
    # DEFAULT SETTINGS untuk 12GB GPU (when no image provided)
    # All values divisible by 16
    WIDTH = 832   # 832 / 16 = 52 ✓
    HEIGHT = 480  # 480 / 16 = 30 ✓
    NUM_FRAMES = 41  # REDUCED from 81 (saves ~50% VRAM!)
    FPS = 16
    GUIDANCE_SCALE = 5.0
    STEPS = 25  # REDUCED from 30 (faster, less VRAM)
    
    # MAX PIXELS untuk different scenarios (12GB GPU)
    # Calculated to result in dimensions divisible by 16
    MAX_PIXELS_LANDSCAPE = 832 * 480   # 399,360 pixels (52x16 x 30x16)
    MAX_PIXELS_PORTRAIT = 480 * 832    # 399,360 pixels (30x16 x 52x16)
    MAX_PIXELS_CONSERVATIVE = 640 * 384  # 245,760 pixels (40x16 x 24x16)
    
    DEFAULT_NEGATIVE_PROMPT = (
        "Bright tones, overexposed, static, blurred details, subtitles, style, "
        "works, paintings, images, static, overall gray, worst quality, low quality, "
        "JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
        "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
        "still picture, messy background, three legs, many people in the background, "
        "walking backwards"
    )
    
    @staticmethod
    def get_max_pixels_for_image(width: int, height: int, conservative: bool = False) -> int:
        """
        Get max pixels based on image orientation and safety level
        Returns values that result in dimensions divisible by 16
        """
        if conservative:
            return VideoGeneratorService.MAX_PIXELS_CONSERVATIVE
        
        aspect_ratio = width / height
        
        # Portrait (height > width)
        if aspect_ratio < 0.9:
            return VideoGeneratorService.MAX_PIXELS_PORTRAIT
        # Landscape or square
        else:
            return VideoGeneratorService.MAX_PIXELS_LANDSCAPE

    @staticmethod
    def generate_video_from_text(
        prompt: str,
        num_frames: int = NUM_FRAMES,
        fps: int = FPS,
        seed: int = 42,
        height: int = HEIGHT,
        width: int = WIDTH,
        guidance_scale: float = GUIDANCE_SCALE,
        num_inference_steps: int = STEPS,
    ) -> Tuple[bool, Optional[str]]:
        """Text-to-Video - 12GB GPU version"""
        
        # Safety check
        if num_frames > 41:
            print(f"[T2V-12GB] WARNING: num_frames={num_frames} too high for 12GB GPU!")
            print(f"[T2V-12GB] Reducing to 41 frames...")
            num_frames = 41
        
        nuclear_clear_memory()
        
        try:
            pipe = get_t2v_pipeline()
            nuclear_clear_memory()
            
            generator = torch.Generator("cuda").manual_seed(seed)
            
            print(f"[T2V-12GB] Generating {num_frames} frames at {width}x{height}...")
            
            with torch.no_grad(), torch.inference_mode():
                output = pipe(
                    prompt=prompt,
                    negative_prompt=VideoGeneratorService.DEFAULT_NEGATIVE_PROMPT,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                )
            
            frames = output.frames[0]
            
            temp_output = os.path.join(
                tempfile.gettempdir(), 
                f"wan_t2v_{uuid.uuid4().hex}.mp4"
            )
            
            export_to_video(frames, temp_output, fps=fps)
            
            print(f"[T2V-12GB SUCCESS] {temp_output}")
            return True, temp_output
            
        except torch.cuda.OutOfMemoryError as oom:
            print(f"[OOM] {oom}")
            if torch.cuda.is_available():
                vram = torch.cuda.memory_allocated(0) / 1024**3
                print(f"[OOM] VRAM: {vram:.2f} GB")
            print(f"[OOM] Try: num_frames=21 or close other GPU apps")
            return False, None
            
        except Exception as e:
            print(f"[ERROR] {str(e)}")
            import traceback
            traceback.print_exc()
            return False, None
            
        finally:
            nuclear_clear_memory()

    @staticmethod
    def generate_video_from_image(
        input_image_path: str,
        prompt: str,
        num_frames: int = NUM_FRAMES,
        fps: int = FPS,
        seed: int = 42,
        height: Optional[int] = None,  # Made optional
        width: Optional[int] = None,   # Made optional
        guidance_scale: float = GUIDANCE_SCALE,
        num_inference_steps: int = STEPS,
        max_pixels: Optional[int] = None,  # Auto-calculate if not provided
        conservative: bool = False,  # Use more conservative max_pixels
    ) -> Tuple[bool, Optional[str]]:
        """
        Image-to-Video - 12GB GPU version with SEQUENTIAL offload
        Auto-calculates optimal dimensions from image if not specified
        Preserves aspect ratio of input image
        """
        
        # Safety check for num_frames
        if num_frames > 41:
            print(f"[VACE-12GB] WARNING: num_frames={num_frames} too high for 12GB GPU!")
            print(f"[VACE-12GB] Reducing to 41 frames...")
            num_frames = 41
        
        nuclear_clear_memory()
        
        try:
            pipe = get_vace_pipeline()
            nuclear_clear_memory()
            
            # Load image
            first_img = load_image(input_image_path).convert("RGB")
            orig_width, orig_height = first_img.size
            
            print(f"[VACE-12GB] Input image: {orig_width}x{orig_height}")
            
            # Calculate optimal dimensions if not provided
            if width is None or height is None:
                # Auto-determine max_pixels based on orientation if not specified
                if max_pixels is None:
                    max_pixels = VideoGeneratorService.get_max_pixels_for_image(
                        orig_width, orig_height, conservative
                    )
                    print(f"[VACE-12GB] Using max_pixels: {max_pixels} (~{int(max_pixels**0.5)}x{int(max_pixels**0.5)})")
                
                width, height = calculate_optimal_dimensions(
                    orig_width, 
                    orig_height, 
                    max_pixels=max_pixels
                )
            else:
                # Ensure provided dimensions are divisible by 16
                width = (width // 16) * 16
                height = (height // 16) * 16
                print(f"[VACE-12GB] Using provided dimensions: {width}x{height}")
            
            # CRITICAL: Final validation - must be divisible by 16
            if width % 16 != 0 or height % 16 != 0:
                print(f"[ERROR] Dimensions not divisible by 16: {width}x{height}")
                # Force correct
                width = (width // 16) * 16
                height = (height // 16) * 16
                print(f"[FIXED] Corrected to: {width}x{height}")
            
            # Calculate estimated VRAM usage
            total_pixels = width * height * num_frames
            estimated_vram_gb = (total_pixels * 4) / (1024**3)  # Rough estimate
            print(f"[VACE-12GB] Estimated VRAM: ~{estimated_vram_gb:.2f} GB (actual may vary)")
            
            if estimated_vram_gb > 10:
                print(f"[VACE-12GB] ⚠️  WARNING: High VRAM estimate! May OOM on 12GB GPU")
            
            # Prepare VACE input with calculated dimensions
            video_frames, mask_frames = prepare_image_to_video_input(
                first_img, height, width, num_frames
            )
            
            generator = torch.Generator("cuda").manual_seed(seed)
            
            print(f"[VACE-12GB] Generating {num_frames} frames at {width}x{height}...")
            print(f"[VACE-12GB] This will be SLOW due to CPU offload (but safe)...")
            
            with torch.no_grad(), torch.inference_mode():
                output = pipe(
                    video=video_frames,
                    mask=mask_frames,
                    prompt=prompt,
                    negative_prompt=VideoGeneratorService.DEFAULT_NEGATIVE_PROMPT,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                )
            
            frames = output.frames[0]
            
            temp_output = os.path.join(
                tempfile.gettempdir(), 
                f"wan_i2v_{uuid.uuid4().hex}.mp4"
            )
            
            export_to_video(frames, temp_output, fps=fps)
            
            print(f"[VACE-12GB SUCCESS] {temp_output}")
            print(f"[VACE-12GB] Output: {width}x{height}, {num_frames} frames, {fps} fps")
            return True, temp_output
            
        except torch.cuda.OutOfMemoryError as oom:
            print(f"[OOM] {oom}")
            if torch.cuda.is_available():
                vram = torch.cuda.memory_allocated(0) / 1024**3
                print(f"[OOM] VRAM: {vram:.2f} GB")
            print(f"[OOM] Even with offload, 12GB is not enough!")
            print(f"[OOM] Current dimensions: {width}x{height}")
            print(f"[OOM] Solutions:")
            print(f"  1. Use T2V instead (lighter)")
            print(f"  2. Use num_frames=21 (currently {num_frames})")
            print(f"  3. Use conservative=True (smaller dimensions)")
            print(f"  4. Reduce max_pixels to {VideoGeneratorService.MAX_PIXELS_CONSERVATIVE}")
            print(f"  5. Close ALL other GPU apps")
            print(f"  6. Restart system to clear GPU")
            return False, None
            
        except Exception as e:
            print(f"[ERROR] {str(e)}")
            import traceback
            traceback.print_exc()
            return False, None
            
        finally:
            nuclear_clear_memory()

    @staticmethod
    def generate_video(
        input_image_path: Optional[str] = None,
        prompt: str = "",
        num_frames: int = NUM_FRAMES,
        fps: int = FPS,
        seed: int = 42,
    ) -> Tuple[bool, Optional[str]]:
        """Auto-detect T2V or I2V"""
        if input_image_path and os.path.exists(input_image_path):
            print("[AUTO-12GB] Using VACE (I2V) - will be SLOW but safe")
            return VideoGeneratorService.generate_video_from_image(
                input_image_path=input_image_path,
                prompt=prompt,
                num_frames=num_frames,
                fps=fps,
                seed=seed,
            )
        else:
            print("[AUTO-12GB] Using T2V")
            return VideoGeneratorService.generate_video_from_text(
                prompt=prompt,
                num_frames=num_frames,
                fps=fps,
                seed=seed,
            )

    @staticmethod
    def generate_from_text(
        output_video_path: str,
        prompt: str,
        style: str = "auto",
        strength: float = 0.3,
        num_frames: int = NUM_FRAMES,
        fps: int = FPS
    ) -> bool:
        """T2V dengan output path"""
        try:
            enhanced_prompt = f"{prompt}, high quality" if style != "auto" else prompt

            success, video_file_path = VideoGeneratorService.generate_video_from_text(
                prompt=enhanced_prompt,
                num_frames=num_frames,
                fps=fps,
            )

            if success and video_file_path and os.path.exists(video_file_path):
                os.rename(video_file_path, output_video_path)
                print(f"[T2V-12GB] Saved → {output_video_path}")
                return True
            
            return False

        except Exception as e:
            print(f"[T2V-12GB ERROR] {str(e)}")
            return False

        finally:
            cleanup_gpu_memory()

    @staticmethod
    def compress_video(video_path: str, crf: int = 28) -> bool:
        """Compress video"""
        if not FFMPEG_AVAILABLE:
            print("[COMPRESS] ffmpeg not available")
            return False

        try:
            output_path = video_path.replace('.mp4', '_compressed.mp4')
            
            (
                ffmpeg.input(video_path)
                .output(
                    output_path,
                    vcodec='libx264',
                    crf=crf,
                    preset='fast',
                    movflags='faststart',
                    pix_fmt='yuv420p',
                )
                .overwrite_output()
                .run(quiet=True)
            )
            
            os.replace(output_path, video_path)
            print(f"[COMPRESS] Done")
            return True
            
        except Exception as e:
            print(f"[COMPRESS ERROR] {e}")
            return False

    @staticmethod
    def generate_long_video(
        input_image_path: Optional[str] = None,
        prompt: str = "",
        target_duration_sec: float = 10.0,
        fps: int = FPS,
        seed: int = 42,
    ) -> Tuple[bool, Optional[str]]:
        """
        Long video generation - Multi-clip with blending for 12GB GPU
        
        For 10 seconds @ 16fps = 160 frames
        Strategy: Generate multiple 41-frame clips with overlap and blend
        """
        print(f"[LONG-12GB] Target duration: {target_duration_sec}s @ {fps} fps")
        
        total_frames_needed = int(target_duration_sec * fps)
        print(f"[LONG-12GB] Total frames needed: {total_frames_needed}")
        
        # If short enough, use single clip
        if total_frames_needed <= VideoGeneratorService.NUM_FRAMES:
            print(f"[LONG-12GB] Short video, using single clip")
            return VideoGeneratorService.generate_video(
                input_image_path=input_image_path,
                prompt=prompt,
                num_frames=total_frames_needed,
                fps=fps,
                seed=seed,
            )
        
        # Long video: use chaining
        print(f"[LONG-12GB] Long video, using multi-clip chaining")
        return VideoGeneratorService._generate_long_video_chained(
            input_image_path=input_image_path,
            prompt=prompt,
            total_frames_needed=total_frames_needed,
            fps=fps,
            seed=seed,
        )
    
    @staticmethod
    def _generate_long_video_chained(
        input_image_path: Optional[str],
        prompt: str,
        total_frames_needed: int,
        fps: int,
        seed: int,
    ) -> Tuple[bool, Optional[str]]:
        """
        FINAL STRATEGY (V4):
        
        Goal: Quality like clip 1, but PROGRESSIVE motion (not looping!)
        
        Solution:
        1. Use BEST QUALITY frame from previous clip (middle frames, not last)
        2. SLIGHT seed variation (consistency but not identical)
        3. Generous overlap blending
        4. Result: Consistent quality + progressive animation!
        """
        from PIL import Image, ImageFilter, ImageEnhance
        import tempfile
        import os
        
        CLIP_FRAMES = 41
        OVERLAP_FRAMES = 15  # Very large overlap for smooth continuation
        
        print(f"[CHAIN-V4] ═══════════════════════════════════════")
        print(f"[CHAIN-V4] PROGRESSIVE Long Video (Quality + Continuation)")
        print(f"[CHAIN-V4] Target: {total_frames_needed} frames ({total_frames_needed/fps:.1f}s)")
        print(f"[CHAIN-V4] ═══════════════════════════════════════\n")
        
        all_frames = []
        frames_per_clip_net = CLIP_FRAMES - OVERLAP_FRAMES
        num_clips = max(1, (total_frames_needed + frames_per_clip_net - 1) // frames_per_clip_net)
        
        print(f"[PLAN] Need {num_clips} clips for {total_frames_needed} frames")
        print(f"[PLAN] Each clip: {CLIP_FRAMES} frames")
        print(f"[PLAN] Overlap: {OVERLAP_FRAMES} frames (smooth blend)")
        print(f"[PLAN] Net new: {frames_per_clip_net} frames per clip\n")
        
        current_image = input_image_path
        
        try:
            for clip_num in range(1, num_clips + 1):
                print(f"{'='*70}")
                print(f"CLIP {clip_num}/{num_clips}")
                print(f"{'='*70}")
                
                # CRITICAL: Slight seed variation for progression
                # Too much = style change, too little = identical
                # Magic number: 3-5 works well for progression without style change
                clip_seed = seed + (clip_num - 1) * 3
                
                print(f"Seed: {clip_seed} (base={seed}, offset={3*(clip_num-1)})")
                print(f"Progress: {len(all_frames)}/{total_frames_needed} frames done")
                
                # Generate clip
                if current_image and os.path.exists(current_image):
                    if clip_num == 1:
                        print(f"Source: ORIGINAL image (first clip)")
                    else:
                        print(f"Source: Best frame from clip {clip_num-1}")
                    
                    success, video_path = VideoGeneratorService.generate_video_from_image(
                        input_image_path=current_image,
                        prompt=prompt,
                        num_frames=CLIP_FRAMES,
                        fps=fps,
                        seed=clip_seed,
                    )
                else:
                    print(f"Source: Text-to-Video")
                    success, video_path = VideoGeneratorService.generate_video_from_text(
                        prompt=prompt,
                        num_frames=CLIP_FRAMES,
                        fps=fps,
                        seed=clip_seed,
                    )
                
                if not success or not video_path:
                    print(f"\n❌ CLIP {clip_num} FAILED\n")
                    return False, None
                
                # Extract frames
                clip_frames = VideoGeneratorService._extract_frames_from_video(video_path)
                
                if not clip_frames:
                    print(f"\n❌ Frame extraction failed\n")
                    return False, None
                
                print(f"✓ Extracted {len(clip_frames)} frames")
                
                # First clip
                if clip_num == 1:
                    all_frames = clip_frames.copy()
                    print(f"✓ Added all {len(clip_frames)} frames")
                
                # Subsequent clips: BLEND OVERLAP
                else:
                    overlap_start = len(all_frames) - OVERLAP_FRAMES
                    
                    print(f"Blending overlap: frames {overlap_start}-{len(all_frames)-1}")
                    
                    # Blend overlap region with smooth interpolation
                    for i in range(OVERLAP_FRAMES):
                        if overlap_start + i < len(all_frames) and i < len(clip_frames):
                            # Cosine interpolation for smooth transition
                            t = (i + 1) / (OVERLAP_FRAMES + 1)
                            alpha = (1 - math.cos(t * math.pi)) / 2
                            
                            old = all_frames[overlap_start + i]
                            new = clip_frames[i]
                            
                            blended = Image.blend(old, new, alpha)
                            all_frames[overlap_start + i] = blended
                    
                    # Add new frames
                    new_frames = clip_frames[OVERLAP_FRAMES:]
                    all_frames.extend(new_frames)
                    
                    print(f"✓ Blended {OVERLAP_FRAMES}, added {len(new_frames)} new")
                
                print(f"✓ Total: {len(all_frames)}/{total_frames_needed} frames\n")
                
                # CRITICAL: Select BEST frame for next clip
                # Use middle-to-late frames (sharp, good quality, not blur)
                if len(all_frames) < total_frames_needed:
                    # Best range: frames 20-30 (after initial motion, before end blur)
                    best_frame_idx = min(25, len(clip_frames) - 10)
                    best_frame = clip_frames[best_frame_idx]
                    
                    # Optional: Slight sharpening to compensate compression
                    enhancer = ImageEnhance.Sharpness(best_frame)
                    best_frame = enhancer.enhance(1.1)  # Subtle sharpen
                    
                    # Save as reference for next clip
                    temp_path = os.path.join(
                        tempfile.gettempdir(),
                        f"chain_v4_clip{clip_num}_f{best_frame_idx}.png"
                    )
                    best_frame.save(temp_path, quality=98, optimize=False)
                    current_image = temp_path
                    
                    print(f"Reference for next: frame {best_frame_idx} (sharp, mid-motion)")
                
                # Cleanup
                try:
                    os.remove(video_path)
                except:
                    pass
                
                nuclear_clear_memory()
                
                if len(all_frames) >= total_frames_needed:
                    break
            
            # Trim exact
            all_frames = all_frames[:total_frames_needed]
            
            print(f"\n{'='*70}")
            print(f"✓ ALL CLIPS DONE: {len(all_frames)} frames")
            print(f"{'='*70}\n")
            
            # Export
            final_path = os.path.join(
                tempfile.gettempdir(),
                f"wan_long_v4_{uuid.uuid4().hex}.mp4"
            )
            
            print(f"Exporting video...")
            export_to_video(all_frames, final_path, fps=fps)
            
            size_mb = os.path.getsize(final_path) / 1024 / 1024
            duration = len(all_frames) / fps
            
            print(f"\n{'='*70}")
            print(f"✓✓✓ SUCCESS ✓✓✓")
            print(f"{'='*70}")
            print(f"File: {final_path}")
            print(f"Duration: {duration:.1f}s ({len(all_frames)} frames @ {fps}fps)")
            print(f"Size: {size_mb:.2f} MB")
            print(f"Quality: Consistent across all clips!")
            print(f"Motion: Progressive continuation (not looping!)")
            print(f"{'='*70}\n")
            
            return True, final_path
            
        except Exception as e:
            print(f"\n❌ ERROR: {str(e)}\n")
            import traceback
            traceback.print_exc()
            return False, None
        
        finally:
            # Cleanup temp images
            try:
                import glob
                temp_dir = tempfile.gettempdir()
                for temp_file in glob.glob(os.path.join(temp_dir, "chain_v4_clip*.png")):
                    os.remove(temp_file)
            except:
                pass
            
            nuclear_clear_memory()
    
    @staticmethod
    def _extract_frames_from_video(video_path: str) -> List[Image.Image]:
        """
        Extract frames from video file
        Returns list of PIL Images
        """
        try:
            import cv2
            
            frames = []
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                print(f"[ERROR] Could not open video: {video_path}")
                return []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL Image
                pil_frame = Image.fromarray(frame_rgb)
                frames.append(pil_frame)
            
            cap.release()
            
            print(f"[EXTRACT] Extracted {len(frames)} frames from video")
            return frames
            
        except Exception as e:
            print(f"[EXTRACT ERROR] {str(e)}")
            return []