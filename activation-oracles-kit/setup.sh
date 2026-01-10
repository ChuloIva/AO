#!/bin/bash

echo "🧠 Activation Oracles Research Kit - Setup Script"
echo "================================================="
echo ""

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Found Python $python_version"

# Check if virtual environment exists, if not create it
if [ ! -d ".venv" ]; then
    echo ""
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo ""
echo "🔌 Activating virtual environment..."
source .venv/bin/activate

# Detect platform
OS_TYPE=$(uname -s)
echo ""
echo "🖥️  Detected platform: $OS_TYPE"

# Install requirements
echo ""
echo "📥 Installing requirements..."
pip install -r requirements.txt --quiet

# Check if activation_oracles exists
echo ""
echo "📦 Checking for activation_oracles dependency..."
if [ -d "../activation_oracles" ]; then
    echo "   ✅ Found activation_oracles in parent directory"
    cd ../activation_oracles

    if [ "$OS_TYPE" = "Darwin" ]; then
        # macOS - skip CUDA/GPU-dependent packages, use MLX for Apple Silicon
        echo "   ⚠️  macOS detected - adjusting dependencies for Apple Silicon"
        echo "   Removing: flash-attn, vllm, liger-kernel (require CUDA/triton)"
        echo "   Installing MLX for Apple Silicon GPU acceleration..."

        # Install MLX and related packages for Apple Silicon
        # Use mlx-lm 0.29.1 for transformers 4.55.2 compatibility
        pip install mlx --quiet
        pip install "mlx-lm>=0.29,<0.30" --quiet

        # Create a temporary pyproject.toml with modified dependencies
        sed -e '/flash-attn/d' \
            -e '/vllm/d' \
            -e '/liger-kernel/d' \
            -e 's/bitsandbytes==0.48.1/bitsandbytes>=0.49.0/' \
            pyproject.toml > pyproject.toml.tmp
        mv pyproject.toml.tmp pyproject.toml

        echo "   Installing activation_oracles as editable package..."
        pip install -e . --quiet

        # Restore original pyproject.toml
        git checkout pyproject.toml 2>/dev/null || true
    else
        # Linux - install torch first, then flash-attn
        echo "   🐧 Linux detected - installing with CUDA support"
        echo "   Installing torch first..."
        pip install torch --quiet

        echo "   🔥 Installing flash-attn (this may take a few minutes)..."
        pip install flash-attn --no-build-isolation

        echo "   Installing activation_oracles as editable package..."
        pip install -e . --no-build-isolation
    fi

    cd ../activation-oracles-kit
else
    echo "   ⚠️  activation_oracles not found in parent directory"
    echo "   Please clone it from: https://github.com/adamkarvonen/activation_oracles"
    exit 1
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "🚀 To run the app:"
echo "   streamlit run app.py"
echo ""
echo "📖 For more information, see README.md"
