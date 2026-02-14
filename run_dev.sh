#!/bin/bash

# Configuration
LOG_FILE="dev_server.log"

# Cleanup function to kill background processes
cleanup() {
    echo ""
    echo "Stopping all services..."
    
    # Kill all child processes
    pkill -P $$
    
    # Extra cleanup for GPU
    echo "Cleaning up GPU memory..."
    pkill -9 celery 2>/dev/null || true
    pkill -9 python 2>/dev/null || true
    
    # Try to reset GPU (may need sudo)
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --gpu-reset 2>/dev/null || true
    fi
    
    wait
    echo "Services stopped and GPU memory cleared."
    exit 0
}

# Trap SIGINT (Ctrl+C) and SIGTERM
trap cleanup SIGINT SIGTERM

# Check for virtual environment
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "Error: Virtual environment 'venv' not found."
    echo "Please create it using: python3 -m venv venv"
    exit 1
fi

# ===== GPU ENVIRONMENT SETUP =====
echo "Setting GPU environment variables..."
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:32"
export CUDA_LAUNCH_BLOCKING=0
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

# ===== PRE-START CLEANUP =====
echo "Cleaning up any existing processes..."
pkill -9 celery 2>/dev/null || true
pkill -9 python 2>/dev/null || true
sleep 2

# Try to reset GPU
if command -v nvidia-smi &> /dev/null; then
    echo "Resetting GPU memory..."
    nvidia-smi --gpu-reset 2>/dev/null || true
    sleep 1
    
    # Check VRAM
    VRAM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo "0")
    echo "GPU VRAM usage: ${VRAM_USED} MB"
    
    if [ "$VRAM_USED" -gt 1000 ]; then
        echo "WARNING: GPU memory still in use (${VRAM_USED} MB)"
        echo "Consider running: sudo nvidia-smi --gpu-reset"
    fi
fi

echo "Starting services... Detailed logs are being written to $LOG_FILE"

# Clear log file
> "$LOG_FILE"

# Start Django Development Server
echo "Starting Django Development Server..." | tee -a "$LOG_FILE"
python manage.py runserver >> "$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo "Django PID: $SERVER_PID"

# Wait for Django to initialize
sleep 3

# Start Celery Worker with GPU-optimized settings
echo "Starting Celery Worker (GPU Mode: concurrency=1)..." | tee -a "$LOG_FILE"
celery -A config worker \
    --pool=solo \
    --concurrency=1 \
    --max-tasks-per-child=1 \
    --autoscale=1,1 \
    --loglevel=info \
    --hostname=gpu-worker@%h \
    >> "$LOG_FILE" 2>&1 &
CELERY_PID=$!
echo "Celery PID: $CELERY_PID"

# Wait and verify Celery started correctly
sleep 5

if ! ps -p $CELERY_PID > /dev/null; then
    echo "ERROR: Celery failed to start!"
    echo "Last 20 lines of log:"
    tail -20 "$LOG_FILE"
    cleanup
fi

# Verify concurrency is actually 1
echo "Verifying Celery concurrency..."
sleep 2
if grep -q "concurrency: 1" "$LOG_FILE"; then
    echo "✓ Celery running with concurrency=1 (correct for GPU)"
elif grep -q "concurrency: 12" "$LOG_FILE"; then
    echo "✗ ERROR: Celery running with concurrency=12 (WRONG!)"
    echo "This will cause GPU OOM errors!"
    echo "Check your config/settings.py or config/celery.py"
    echo ""
    echo "Add to config/settings.py:"
    echo "  CELERY_WORKER_CONCURRENCY = 1"
    echo "  CELERY_WORKER_POOL = 'solo'"
    cleanup
else
    echo "⚠ WARNING: Could not verify Celery concurrency"
    echo "Check log file: $LOG_FILE"
fi

echo ""
echo "========================================="
echo "Services started successfully!"
echo "========================================="
echo "Django PID:  $SERVER_PID"
echo "Celery PID:  $CELERY_PID"
echo "Log file:    $LOG_FILE"
echo ""
echo "Monitoring:"
echo "  GPU:   watch -n 1 nvidia-smi"
echo "  Logs:  tail -f $LOG_FILE"
echo ""
echo "Press Ctrl+C to stop all services"
echo "========================================="
echo ""

# Spinner animation characters
sp="/-\|"
sc=0

# Loop to keep script running and show spinner + periodic GPU stats
COUNTER=0
while kill -0 $SERVER_PID 2>/dev/null && kill -0 $CELERY_PID 2>/dev/null; do
    printf "\rRunning services... [%s]" "${sp:sc++:1}"
    ((sc==4)) && sc=0
    
    # Show GPU stats every 30 seconds
    ((COUNTER++))
    if [ $((COUNTER % 300)) -eq 0 ] && command -v nvidia-smi &> /dev/null; then
        echo ""
        echo "[$(date +%H:%M:%S)] GPU Status:"
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | \
        awk '{printf "  GPU: %s%% | VRAM: %s/%s MB | Temp: %s°C\n", $1, $2, $3, $4}'
        printf "\rRunning services... [%s]" "${sp:sc:1}"
    fi
    
    sleep 0.1
done

# If we get here, one of the processes has exited unexpectedly
echo ""
echo "One of the services has exited unexpectedly. Check $LOG_FILE for details."
echo ""
echo "Last 30 lines of log:"
tail -30 "$LOG_FILE"
cleanup