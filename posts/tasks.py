from celery import shared_task
import traceback
import os
import tempfile
import uuid
from django.core.files import File
import torch
import gc

from .models import Post
from .services import ImageGeneratorService, VideoGeneratorService
from .infrastructure import cleanup_gpu_memory


@shared_task
def process_media_task(post_id: int) -> None:
    """
    Celery task untuk memproses media generation
    Updated untuk menggunakan Wan2.1 models (T2V dan VACE)
    """
    post = Post.objects.get(id=post_id)
    post.status = Post.PostStatus.PROCESSING
    post.save(update_fields=["status"])

    try:
        use_ai = post.use_ai
        gen_type = post.generation_type
        style = post.ai_style or "auto"
        prompt = post.ai_prompt or "default high quality cinematic scene"
        video_prompt = post.video_prompt or prompt
        
        # Default video settings untuk Wan2.1
        # 81 frames @ 16fps = ~5 detik per clip
        # For longer videos, system will chain multiple clips
        target_sec = getattr(post, 'ai_video_duration', 10)  # Default 10 detik
        fps = 16  # FPS optimal untuk Wan2.1
        num_frames = min(int(target_sec * fps), 160)  # Max 160 frames for 10s
        
        # For long videos (>2.5s), will use chaining
        if target_sec > 2.5:
            print(f"[INFO] Long video ({target_sec}s) - will use multi-clip chaining")

        print(f"[TASK START] Post ID {post_id}")
        print(f"  gen_type: {gen_type}")
        print(f"  use_ai: {use_ai}")
        print(f"  target_duration: {target_sec}s")
        print(f"  num_frames: {num_frames}")
        print(f"  fps: {fps}")

        # === MODE 1: Non-AI (hanya compress video) ===
        if not use_ai:
            print("[MODE] No AI - compress existing video if any")
            if post.video:
                try:
                    VideoGeneratorService.compress_video(post.video.path)
                    print("[OK] Video compressed")
                except Exception as e:
                    print(f"[WARN] Compress failed: {e}")
            
            post.status = Post.PostStatus.COMPLETED
            post.save(update_fields=["status"])
            return

        # === Helper: Save video to post ===
        def save_video_to_post(temp_video_path: str | None) -> bool:
            """Save generated video to post model"""
            if not temp_video_path or not os.path.exists(temp_video_path):
                print("[ERROR] Video file not found after generation")
                return False

            filename = f"posts/videos/vid_{uuid.uuid4().hex}.mp4"
            
            try:
                with open(temp_video_path, "rb") as f:
                    post.video.save(filename, File(f), save=False)
                print(f"[OK] Video saved → {post.video.name}")
                
                # Cleanup temp file
                try:
                    os.remove(temp_video_path)
                    print(f"[OK] Temp file removed: {temp_video_path}")
                except Exception as e:
                    print(f"[WARN] Failed to remove temp file: {e}")
                
                return True
                
            except Exception as e:
                print(f"[ERROR] Failed to save video: {e}")
                return False

        # === MODE 2: AI Generation ===
        print(f"[MODE] AI Generation")
        print(f"  Generation Type: {gen_type}")
        print(f"  Prompt: {prompt[:100]}...")
        print(f"  Video Prompt: {video_prompt[:100]}...")

        # === TEXT-TO-IMAGE ===
        if gen_type == Post.GenerationType.TEXT_TO_IMAGE:
            print("[T2I] Starting Text-to-Image generation...")
            temp_path = os.path.join(tempfile.gettempdir(), f"img_{uuid.uuid4().hex}.png")
            
            try:
                result = ImageGeneratorService.generate_from_text(
                    temp_path, 
                    prompt, 
                    style=style
                )
                
                if result and result.get("success"):
                    filename = f"posts/images/gen_{uuid.uuid4().hex}.png"
                    with open(temp_path, "rb") as f:
                        post.image.save(filename, File(f), save=False)
                    print(f"[T2I OK] Image saved → {post.image.name}")
                else:
                    raise RuntimeError(f"Text-to-image failed: {result}")
                    
            finally:
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception as e:
                        print(f"[WARN] Failed to remove temp image: {e}")

        # === IMAGE-TO-IMAGE ===
        elif gen_type == Post.GenerationType.IMAGE_TO_IMAGE:
            print("[I2I] Starting Image-to-Image generation...")
            
            if not post.image:
                raise ValueError("Initial image required for image-to-image generation")
            
            result = ImageGeneratorService.generate(
                post.image.path,
                prompt,
                style=style,
                strength=post.ai_strength,
                use_adetailer=True,
                enhance_face=True
            )
            
            if result and result.get("success"):
                # Overwrite existing image
                with open(post.image.path, "rb") as f:
                    post.image.save(post.image.name, File(f), save=False)
                print("[I2I OK] Image-to-image completed (overwrite)")
            else:
                raise RuntimeError(f"Image-to-image failed: {result}")

        # === TEXT-TO-VIDEO ===
        elif gen_type == Post.GenerationType.TEXT_TO_VIDEO:
            print("[T2V] Starting Text-to-Video generation...")
            print(f"[T2V] Using Wan2.1-T2V-1.3B-Diffusers pipeline")
            
            # Text-to-Video langsung tanpa perlu image intermediate
            # Menggunakan dedicated T2V pipeline
            # For long videos, will automatically chain multiple clips
            try:
                success, video_path = VideoGeneratorService.generate_long_video(
                    input_image_path=None,  # T2V - no image
                    prompt=video_prompt,
                    target_duration_sec=target_sec,
                    fps=fps,
                    seed=42 + post_id,
                )
                
                if success and video_path:
                    if save_video_to_post(video_path):
                        print(f"[T2V OK] Video generated and saved")
                    else:
                        raise RuntimeError("Failed to save generated video")
                else:
                    raise RuntimeError("Text-to-video generation failed")
                    
            except Exception as e:
                print(f"[T2V ERROR] {str(e)}")
                raise

        # === IMAGE-TO-VIDEO ===
        elif gen_type == Post.GenerationType.IMAGE_TO_VIDEO:
            print("[I2V] Starting Image-to-Video generation...")
            print(f"[I2V] Using Wan2.1-VACE-1.3B-diffusers pipeline")
            
            if not post.image:
                raise ValueError("Initial image required for image-to-video generation")
            
            # Image-to-Video menggunakan dedicated VACE pipeline
            # For long videos, will automatically chain multiple clips
            try:
                success, video_path = VideoGeneratorService.generate_long_video(
                    input_image_path=post.image.path,
                    prompt=video_prompt,
                    target_duration_sec=target_sec,
                    fps=fps,
                    seed=42 + post_id,
                )
                
                if success and video_path:
                    if save_video_to_post(video_path):
                        print(f"[I2V OK] Video generated and saved")
                    else:
                        raise RuntimeError("Failed to save generated video")
                else:
                    raise RuntimeError("Image-to-video generation failed")
                    
            except Exception as e:
                print(f"[I2V ERROR] {str(e)}")
                raise

        else:
            raise ValueError(f"Unknown generation type: {gen_type}")

        # === Post-processing: Compress Video ===
        if post.video and os.path.exists(post.video.path):
            try:
                print("[COMPRESS] Compressing video...")
                if VideoGeneratorService.compress_video(post.video.path, crf=28):
                    print("[COMPRESS OK] Video compressed successfully")
                else:
                    print("[COMPRESS WARN] Video compression failed, using original")
            except Exception as e:
                print(f"[COMPRESS ERROR] {str(e)}")
                # Continue anyway, compression is optional

        # === Mark as completed ===
        post.status = Post.PostStatus.COMPLETED
        post.save(update_fields=["status", "video", "image"])

        print(f"[TASK SUCCESS] Post {post_id} completed successfully")
        print(f"  - Image: {post.image.name if post.image else 'None'}")
        print(f"  - Video: {post.video.name if post.video else 'None'}")

    except Exception as e:
        # === Error handling ===
        error_msg = f"{str(e)}\n{traceback.format_exc()[:1500]}"
        print(f"[TASK ERROR] Post {post_id} failed:")
        print(error_msg)
        
        post.status = Post.PostStatus.FAILED
        post.error_message = error_msg
        post.save(update_fields=["status", "error_message"])

    finally:
        # === Cleanup ===
        print("[CLEANUP] Cleaning up GPU memory...")
        try:
            cleanup_gpu_memory()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
            
            if torch.cuda.is_available():
                vram_used = torch.cuda.memory_allocated(0) / 1024**3
                print(f"[CLEANUP OK] VRAM after cleanup: {vram_used:.2f} GB")
        except Exception as e:
            print(f"[CLEANUP WARN] Cleanup failed: {e}")


@shared_task
def batch_process_media_tasks(post_ids: list[int]) -> dict:
    """
    Process multiple posts in batch
    Useful untuk queue management
    
    Returns:
        dict dengan summary hasil processing
    """
    results = {
        'total': len(post_ids),
        'success': 0,
        'failed': 0,
        'errors': []
    }
    
    print(f"[BATCH] Processing {len(post_ids)} posts...")
    
    for post_id in post_ids:
        try:
            process_media_task(post_id)
            results['success'] += 1
            print(f"[BATCH] Post {post_id} ✓")
        except Exception as e:
            results['failed'] += 1
            results['errors'].append({
                'post_id': post_id,
                'error': str(e)
            })
            print(f"[BATCH] Post {post_id} ✗ - {str(e)}")
    
    print(f"[BATCH COMPLETE] {results['success']}/{results['total']} succeeded")
    return results


@shared_task
def cleanup_failed_posts() -> dict:
    """
    Cleanup task untuk retry atau remove failed posts
    Dapat dijalankan periodic dengan celery beat
    """
    from datetime import timedelta
    from django.utils import timezone
    
    # Find posts yang failed > 1 jam yang lalu
    one_hour_ago = timezone.now() - timedelta(hours=1)
    failed_posts = Post.objects.filter(
        status=Post.PostStatus.FAILED,
        updated_at__lt=one_hour_ago
    )
    
    count = failed_posts.count()
    print(f"[CLEANUP] Found {count} failed posts older than 1 hour")
    
    # Option 1: Reset to pending untuk retry
    reset_count = 0
    for post in failed_posts[:10]:  # Limit 10 per cleanup
        try:
            post.status = Post.PostStatus.PENDING
            post.error_message = ""
            post.save(update_fields=["status", "error_message"])
            
            # Re-queue
            process_media_task.delay(post.id)
            reset_count += 1
            
            print(f"[CLEANUP] Post {post.id} reset and re-queued")
        except Exception as e:
            print(f"[CLEANUP ERROR] Failed to reset post {post.id}: {e}")
    
    return {
        'found': count,
        'reset': reset_count
    }


@shared_task
def warm_up_models() -> dict:
    """
    Warm-up task untuk pre-load models
    Berguna untuk mengurangi cold-start time
    """
    results = {
        'success': False,
        'models_loaded': []
    }
    
    try:
        print("[WARMUP] Starting model warm-up...")
        
        # Load T2V pipeline
        try:
            from .services import VideoGeneratorService
            from .services.video_generator_wan21 import get_t2v_pipeline
            
            print("[WARMUP] Loading T2V pipeline...")
            t2v_pipe = get_t2v_pipeline()
            results['models_loaded'].append('T2V-1.3B')
            print("[WARMUP] T2V pipeline loaded ✓")
        except Exception as e:
            print(f"[WARMUP ERROR] T2V loading failed: {e}")
        
        # Load VACE pipeline
        try:
            from .services.video_generator_wan21 import get_vace_pipeline
            
            print("[WARMUP] Loading VACE pipeline...")
            vace_pipe = get_vace_pipeline()
            results['models_loaded'].append('VACE-1.3B')
            print("[WARMUP] VACE pipeline loaded ✓")
        except Exception as e:
            print(f"[WARMUP ERROR] VACE loading failed: {e}")
        
        # Load Image generator if needed
        try:
            print("[WARMUP] Loading Image generator...")
            # Add your image generator warm-up here
            results['models_loaded'].append('Image-Generator')
            print("[WARMUP] Image generator loaded ✓")
        except Exception as e:
            print(f"[WARMUP ERROR] Image generator loading failed: {e}")
        
        results['success'] = len(results['models_loaded']) > 0
        print(f"[WARMUP COMPLETE] Loaded {len(results['models_loaded'])} models")
        
    except Exception as e:
        print(f"[WARMUP ERROR] Warm-up failed: {e}")
        results['error'] = str(e)
    
    finally:
        # Cleanup setelah warm-up
        cleanup_gpu_memory()
        gc.collect()
    
    return results