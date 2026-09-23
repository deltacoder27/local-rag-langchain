#!/bin/bash

set -e

if ! command -v ollama &> /dev/null; then
    echo "Error: Ollama is not installed."
    echo "Please install Ollama first, then run this script again."
    exit 1
fi

echo "Creating Python virtual environment..."
python3 -m venv .venv

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Pulling Ollama models..."
ollama pull qwen2.5:7b
ollama pull embeddinggemma:latest

echo "Setup complete!"
echo "Run the project with: source .venv/bin/activate && python main.py"