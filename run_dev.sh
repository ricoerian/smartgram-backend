#!/bin/bash

cleanup() {
    echo "Stopping all services..."
    pkill -P $$
    wait
    echo "Services stopped."
}
trap cleanup SIGINT SIGTERM

if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "Error: Virtual environment 'venv' not found."
    echo "Please create it using: python3 -m venv venv"
    exit 1
fi

echo "Starting Django Development Server..."
python manage.py runserver &

echo "Starting Celery Worker..."
celery -A config worker --pool=solo --loglevel=info &
wait
