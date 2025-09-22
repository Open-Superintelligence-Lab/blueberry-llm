#!/bin/bash
# One-click setup for Blueberry LLM

echo "🫐 Blueberry LLM Setup"
echo "======================"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8+"
    exit 1
fi

echo "✅ Python found: $(python3 --version)"

# Install PyTorch with CUDA support
# echo "📦 Installing PyTorch..."
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
echo "📦 Installing other dependencies..."
pip3 install -r requirements.txt

# Test installation
echo "🧪 Testing installation..."
python3 -c "import torch; print(f'✅ PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'✅ CUDA available: {torch.cuda.is_available()}')"

if command -v nvidia-smi &> /dev/null; then
    echo "🎯 GPU Status:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi

# Test auto-configuration
echo ""
echo "🔧 Testing auto-configuration..."
python3 core/auto_config.py

echo ""
echo "🚀 Setup complete! Ready to train:"
echo "   python3 train.py"
echo ""
echo "📊 To see configuration only:"
echo "   python3 core/auto_config.py"
echo ""
echo "🧪 To run tests:"
echo "   python3 test.py"
