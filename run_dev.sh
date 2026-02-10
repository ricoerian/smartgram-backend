#!/bin/bash

# Configuration
LOG_FILE="dev_server.log"

# Cleanup function to kill background processes
cleanup() {
    echo ""
    echo "Stopping all services..."
    # Kill all child processes of the current script's process group
    pkill -P $$
    wait
    echo "Services stopped."
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

echo "Starting services... Detailed logs are being written to $LOG_FILE"

# Clear log file
> "$LOG_FILE"

# Start Django Development Server
echo "Starting Django Development Server..." >> "$LOG_FILE"
python manage.py runserver >> "$LOG_FILE" 2>&1 &
SERVER_PID=$!

# Start Celery Worker
echo "Starting Celery Worker..." >> "$LOG_FILE"
celery -A config worker --pool=solo --loglevel=info >> "$LOG_FILE" 2>&1 &
CELERY_PID=$!

echo "Services started. (PID: Server=$SERVER_PID, Celery=$CELERY_PID)"
echo "Press Ctrl+C to stop."

# Spinner animation characters
sp="/-\|"
sc=0

# Loop to keep script running and show spinner
while kill -0 $SERVER_PID 2>/dev/null && kill -0 $CELERY_PID 2>/dev/null; do
    printf "\rRunning services... [%s]" "${sp:sc++:1}"
    ((sc==4)) && sc=0
    sleep 0.1
done

# If we get here, one of the processes has exited unexpectedly
echo ""
echo "One of the services has exited unexpectedly. Check $LOG_FILE for details."
cleanup
