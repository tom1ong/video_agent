#!/bin/bash
# Production setup script

set -e

echo "🚀 Setting up Video Editing Agent..."

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Create directories
echo "📁 Creating directories..."
mkdir -p workspace logs

# Check for .env.local
if [ ! -f .env.local ]; then
    echo "⚠️  Creating .env.local template..."
    echo "GEMINI_API_KEY=your-api-key-here" > .env.local
    echo "❌ Please edit .env.local and add your GEMINI_API_KEY"
    exit 1
fi

echo "✅ Setup complete!"
echo ""
echo "To run: python3 main.py"

