# Smartgram Backend

AI-powered image and video processing API built with Django REST Framework, featuring Stable Diffusion XL with ControlNet for professional-grade image enhancement and style transfer.

## Features

- **AI Image Enhancement**: Transform images using Stable Diffusion XL with intelligent ControlNet guidance
- **16 Artistic Styles**: From hyperrealistic to anime, watercolor to cyberpunk
- **Intelligent Auto-Detection**: Automatic strength computation based on image complexity analysis
- **ControlNet Integration**: 
  - Canny edge detection for structure preservation
  - OpenPose detection for pose-aware generation
- **Memory Optimized**: Specifically tuned for 16GB RAM + RTX 3060 12GB GPU
- **Async Processing**: Celery-based task queue for non-blocking API responses
- **RESTful API**: Clean API design with Django REST Framework
- **Clean Architecture**: Modular design with separation of concerns

## Technology Stack

### Core Framework
- **Django 5.x**: Web framework and ORM
- **Django REST Framework**: RESTful API toolkit
- **Celery**: Distributed task queue

### AI/ML
- **PyTorch**: Deep learning framework
- **Diffusers**: Hugging Face Stable Diffusion pipeline
- **ControlNet**: Structure-guided image generation
- **Compel**: Advanced prompt weighting
- **OpenCV**: Computer vision operations
- **Pillow**: Image processing

### Infrastructure
- **Redis**: Message broker for Celery
- **SQLite**: Development database
- **FFmpeg**: Video processing (optional)

## Hardware Requirements

**Recommended Configuration** (Optimized for):
- **RAM**: 16GB DDR4 or higher
- **GPU**: NVIDIA RTX 3060 12GB (or equivalent with 12GB+ VRAM)
- **Storage**: 20GB+ free space for model cache
- **CUDA**: 11.8 or higher

**Minimum Requirements**:
- **RAM**: 8GB (CPU mode, slower)
- **GPU**: Any CUDA-capable GPU with 8GB+ VRAM
- **Storage**: 15GB free space

## Installation

### 1. Clone Repository

```bash
git clone <repository-url>
cd smartgram-backend
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 4. Install Redis

**Ubuntu/Debian**:
```bash
sudo apt update
sudo apt install redis-server
sudo systemctl start redis
```

**macOS**:
```bash
brew install redis
brew services start redis
```

**Windows**: Download from [redis.io](https://redis.io/download)

### 5. Configure Environment

Create `.env` file in project root:

```env
DJANGO_SECRET_KEY=your-secret-key-here
DJANGO_DEBUG=True
DJANGO_ALLOWED_HOSTS=localhost,127.0.0.1

CELERY_BROKER_URL=redis://127.0.0.1:6379/0
CELERY_RESULT_BACKEND=redis://127.0.0.1:6379/0
```

### 6. Run Migrations

```bash
python manage.py migrate
python manage.py createsuperuser
```

### 7. Start Services

**Terminal 1 - Django Server**:
```bash
python manage.py runserver
```

**Terminal 2 - Celery Worker**:
```bash
celery -A config worker -l info
```

**Terminal 3 - Redis** (if not running as service):
```bash
redis-server
```

## API Documentation

### Endpoints

#### List Posts
```http
GET /api/posts/
```

#### Create Post
```http
POST /api/posts/
Content-Type: multipart/form-data

{
  "caption": "Beautiful landscape",
  "image": <file>,
  "use_ai": true,
  "ai_style": "hdr",
  "ai_prompt": "dramatic sky, golden hour"
}
```

#### Retrieve Post
```http
GET /api/posts/{id}/
```

#### Update Post
```http
PUT /api/posts/{id}/
PATCH /api/posts/{id}/
```

#### Delete Post
```http
DELETE /api/posts/{id}/
```

### Request Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `caption` | string | No | Post caption text |
| `image` | file | No* | Image file to process |
| `video` | file | No* | Video file to process |
| `use_ai` | boolean | No | Enable AI processing (default: false) |
| `ai_style` | string | No | Style preset (default: "auto") |
| `ai_prompt` | string | No | Additional prompt guidance |
| `ai_strength` | float | No | ControlNet strength 0.20-0.70 (auto if empty) |

*Either `image` or `video` required

### Response Example

```json
{
  "id": 1,
  "username": "john",
  "caption": "Beautiful landscape",
  "image": "/media/posts/images/photo.jpg",
  "video": null,
  "use_ai": true,
  "ai_style": "hdr",
  "ai_prompt": "dramatic sky, golden hour",
  "ai_strength": null,
  "status": "completed",
  "created_at": "2026-02-08T21:30:00Z"
}
```

### Status Values

- `pending`: Queued for processing
- `processing`: Currently being processed
- `completed`: Successfully processed
- `failed`: Processing failed

## AI Style Options

| Style | Description |
|-------|-------------|
| `auto` | Automatic enhancement with quality boost |
| `realistic` | Hyperrealistic photography |
| `hdr` | High dynamic range photography |
| `noir` | Film noir black and white |
| `sepia` | Vintage sepia tone |
| `sketch` | Pencil sketch drawing |
| `cartoon` | Pixar/Disney 3D render |
| `anime` | High-quality anime style |
| `ghibli` | Studio Ghibli aesthetic |
| `oil_painting` | Classical oil painting |
| `watercolor` | Watercolor painting |
| `pop_art` | Andy Warhol pop art |
| `cyber` | Cyberpunk neon aesthetic |
| `fantasy` | Epic fantasy concept art |
| `steampunk` | Victorian steampunk |
| `minimalist` | Minimalist fine art |

## Architecture

The project follows clean architecture principles:

```
posts/
├── config/          # Configuration constants
├── domain/          # Business entities and value objects
├── use_cases/       # Application business logic
├── infrastructure/  # External dependencies (AI, image processing)
├── adapters/        # Data transformation
└── services.py      # Public API facade
```

### Layers

**Domain Layer**: Pure business logic with no external dependencies
- Entities: `ImageAnalysis`, `ProcessingConfig`
- Value Objects: `DeviceConfig`, `ImageDimensions`, `StrengthValue`

**Use Cases Layer**: Application workflows
- Image analysis and strength computation
- AI generation orchestration

**Infrastructure Layer**: External integrations
- AI model management and lifecycle
- Image processing utilities
- Memory management

**Adapters Layer**: Data transformation
- Prompt building and enhancement

## Development

### Running Tests

```bash
python manage.py test
```

### Code Quality

The codebase follows:
- Clean code principles
- Self-documenting code (no comments)
- Type hints for better IDE support
- Modular architecture for testability

### Memory Optimization

For systems with limited resources, environmental variables can be adjusted:

```python
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
```

The system automatically:
- Clears GPU cache between operations
- Uses memory-efficient attention mechanisms
- Enables VAE tiling and slicing
- Optimizes model loading/unloading

## Troubleshooting

### Out of Memory Errors

1. Reduce image resolution (default max: 1024px)
2. Disable hires fix
3. Close other GPU applications
4. Restart Celery worker to clear memory

### Celery Not Processing

```bash
redis-cli ping  # Should return PONG
celery -A config inspect active
```

### Model Download Issues

Models auto-download on first run. Ensure:
- Stable internet connection
- ~15GB free disk space
- Hugging Face not blocked by firewall

### Import Errors

```bash
pip install --upgrade -r requirements.txt
```

## Performance Tips

1. **Use GPU**: Always use CUDA-enabled GPU for production
2. **Batch Processing**: Process multiple images sequentially rather than parallel
3. **Model Caching**: Keep Celery worker running to avoid model reload
4. **Redis Configuration**: Tune Redis maxmemory for your system
5. **Image Size**: Smaller images process faster (resize before upload)

## Contributing

1. Follow the established clean architecture pattern
2. Write self-documenting code without comments
3. Use type hints for all function signatures
4. Test on the recommended hardware configuration
5. Update README for new features

## License

[Your License Here]

## Acknowledgments

- [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- [ControlNet](https://github.com/lllyasviel/ControlNet)
- [Diffusers](https://github.com/huggingface/diffusers)
- [RunDiffusion Juggernaut XL](https://huggingface.co/RunDiffusion/Juggernaut-XL-v9)
