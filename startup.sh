#!/bin/sh
echo "🔁 Creating or activating virtualenv"
if [ ! -d "antenv" ]; then
    python -m venv antenv
fi
. antenv/bin/activate
pip install -r requirements.txt

echo "🚀 Launching app via Gunicorn"
exec gunicorn --bind=0.0.0.0:${PORT:-5000} app:app
