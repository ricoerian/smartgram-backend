from celery import shared_task

from .models import Post
from .services import compress_video, generate_ai_image, generate_ai_video


@shared_task
def process_media_task(post_id: int) -> None:
    try:
        post = Post.objects.get(id=post_id)
        post.status = 'processing'
        post.save()

        print(f"Processing Post ID: {post_id}")

        ai_strength = post.ai_strength if post.ai_strength else 0.35

        if post.image and post.use_ai:
            base_prompt = post.ai_prompt or ""
            style = post.ai_style or "auto"
            generate_ai_image(post.image.path, base_prompt, style=style, strength=ai_strength)

        if post.video:
            if post.use_ai:
                print(f"Running AI Video for Post {post_id}")
                base_prompt = post.ai_prompt or "high quality video"
                style = post.ai_style or "auto"
                generate_ai_video(post.video.path, base_prompt, style=style, strength=ai_strength)
            else:
                compress_video(post.video.path)

        post.status = 'completed'
        post.save()
        print("Task completed successfully")

    except Exception as e:
        print(f"Task error: {e}")
        try:
            p = Post.objects.get(id=post_id)
            p.status = 'failed'
            p.save()
        except Exception:
            pass