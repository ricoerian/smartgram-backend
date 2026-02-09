
import os
import django
from unittest.mock import patch, MagicMock

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from posts.models import Post
from posts.tasks import process_media_task
from posts.services import ImageGeneratorService

def verify_refactoring():
    print("🚀 Starting Verification...")
    
    # 1. Verify Post Status Enum
    print("\n1. Verifying Post Status Enum...")
    assert Post.PostStatus.PENDING == 'pending'
    assert Post.PostStatus.FAILED == 'failed'
    print("✅ PostStatus Enum works correctly.")

    # 2. Verify Success Flow
    print("\n2. Verifying Process Media Task (Success Flow)...")
    
    # Create dummy post
    user = django.contrib.auth.models.User.objects.first()
    if not user:
        user = django.contrib.auth.models.User.objects.create_user('testuser', 'test@example.com', 'password')
    
    post = Post.objects.create(
        user=user,
        caption="Test Post",
        use_ai=True,
        ai_style="auto",
        status=Post.PostStatus.PENDING
    )
    # Mock image field to have a path
    post.image = MagicMock()
    post.image.path = "/tmp/fake_image.jpg"
    post.save()

    with patch('posts.tasks.ImageGeneratorService.generate') as mock_generate, \
         patch('posts.tasks.cleanup_gpu_memory'):
        
        # Mock success response
        mock_generate.return_value = {
            "success": True,
            "was_auto_detected": True,
            "strength": 0.5
        }
        
        process_media_task(post.id)
        
        post.refresh_from_db()
        print(f"Post Status: {post.status}")
        assert post.status == Post.PostStatus.COMPLETED
        assert post.strength_auto_detected is True
        assert post.detected_strength_value == 0.5
        print("✅ Success flow verified.")

    # 3. Verify Failure Flow
    print("\n3. Verifying Process Media Task (Failure Flow)...")
    
    post_fail = Post.objects.create(
        user=user,
        caption="Fail Post",
        use_ai=True,
        ai_style="auto",
        status=Post.PostStatus.PENDING
    )
    post_fail.image = MagicMock()
    post_fail.image.path = "/tmp/fake_image_fail.jpg"
    post_fail.save()

    with patch('posts.tasks.ImageGeneratorService.generate') as mock_generate, \
         patch('posts.tasks.cleanup_gpu_memory'):
        
        # Mock failure by raising exception
        mock_generate.side_effect = Exception("Simulated AI Failure")
        
        process_media_task(post_fail.id)
        
        post_fail.refresh_from_db()
        print(f"Post Status: {post_fail.status}")
        print(f"Error Message: {post_fail.error_message}")
        
        assert post_fail.status == Post.PostStatus.FAILED
        assert "Simulated AI Failure" in post_fail.error_message
        print("✅ Failure flow verified (Exception caught, status failed, error logged).")

    print("\n🎉 All verifications passed!")

if __name__ == "__main__":
    verify_refactoring()
