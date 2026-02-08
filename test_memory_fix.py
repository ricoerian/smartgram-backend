#!/usr/bin/env python3
import os
import sys
import django

os.chdir('/home/ricoeri/Documents/smartgram-backend')
sys.path.insert(0, '/home/ricoeri/Documents/smartgram-backend')

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from posts.services import generate_ai_image
import torch
from PIL import Image

test_img = 'media/test_memory.jpg'

os.makedirs('media', exist_ok=True)
if not os.path.exists(test_img):
    img = Image.new('RGB', (512, 512), color=(100, 150, 200))
    img.save(test_img, quality=95)
    print(f"Created test image: {test_img}")

def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    else:
        print("CUDA not available")

print("=" * 60)
print("MEMORY LEAK TEST - Running AI process twice")
print("=" * 60)

print("\n[INITIAL STATE]")
print_gpu_memory()

print("\n" + "=" * 60)
print("FIRST RUN - Processing image...")
print("=" * 60)
result1 = generate_ai_image(
    test_img, 
    'a beautiful landscape with mountains and lake', 
    use_openpose=False, 
    use_hires=False, 
    inference_steps=10
)
print(f"\nFirst run result: {'✅ SUCCESS' if result1 else '❌ FAILED'}")
print_gpu_memory()

print("\n" + "=" * 60)
print("SECOND RUN - Critical test (this should not crash)")
print("=" * 60)
result2 = generate_ai_image(
    test_img, 
    'a futuristic cityscape at sunset', 
    use_openpose=False, 
    use_hires=False, 
    inference_steps=10
)
print(f"\nSecond run result: {'✅ SUCCESS' if result2 else '❌ FAILED'}")
print_gpu_memory()

print("\n" + "=" * 60)
print("TEST RESULTS")
print("=" * 60)
if result1 and result2:
    print("✅ PASS: Both runs completed successfully!")
    print("✅ Memory leak fix verified - no SIGKILL crash")
    sys.exit(0)
else:
    print("❌ FAIL: One or both runs failed")
    if result1 and not result2:
        print("❌ Second run failed - memory leak still present")
    sys.exit(1)
