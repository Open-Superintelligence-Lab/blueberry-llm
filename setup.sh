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

# Install core dependencies first
echo "📦 Installing core dependencies..."
pip3 install datasets transformers torchtune torchao

# Install profiler dependencies (optional)
echo "📦 Installing profiler dependencies (optional)..."
pip3 install psutil matplotlib pandas || echo "⚠️  Some profiler dependencies failed to install (optional)"
pip3 install nvidia-ml-py3 || echo "⚠️  nvidia-ml-py3 failed to install (optional)"

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
python3 auto_config.py

echo ""
echo "🚀 Setup complete! Ready to train:"
echo "   python3 train_auto.py"
echo ""
echo "📊 To see configuration only:"
echo "   python3 auto_config.py"
