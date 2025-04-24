#!/bin/sh

echo "📦 Updating system packages"
apt-get update && apt-get install -y build-essential libgl1-mesa-glx libglib2.0-0

echo "🐍 Creating virtual environment"
if [ ! -d "antenv" ]; then
    python -m venv antenv
fi

echo "✅ Activating virtual environment"
. antenv/bin/activate

echo "⬇️ Installing Python dependencies"
pip install --upgrade pip
pip install -r requirements.txt

echo "🚀 Starting Flask app with Gunicorn"
exec gunicorn --bind=0.0.0.0:${PORT:-5000} app:app
