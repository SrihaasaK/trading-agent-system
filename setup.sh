#!/bin/bash
# setup.sh — One-time setup script for Mac Mini
# Run once after cloning: bash setup.sh

set -e

echo "================================================"
echo "  Trading Agent System — Mac Mini Setup"
echo "================================================"
echo ""

# 1. Check Python version
PYTHON=$(which python3)
PYVER=$($PYTHON --version 2>&1 | awk '{print $2}')
echo "✓ Python: $PYVER at $PYTHON"

# 2. Create virtual environment
if [ ! -d "venv" ]; then
    echo "→ Creating virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate
echo "✓ Virtual environment activated"

# 3. Install dependencies
echo "→ Installing dependencies (this takes 2-3 minutes)..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "✓ Dependencies installed"

# 4. Copy .env
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo ""
    echo "⚠️  Created .env from template."
    echo "   Open .env and add your API keys before running the bot."
    echo ""
fi

# 5. Pull Ollama model (optional but recommended)
if command -v ollama &> /dev/null; then
    echo "→ Pulling Ollama model (llama3.2:8b)..."
    ollama pull llama3.2:8b
    echo "✓ Ollama model ready"
else
    echo "ℹ️  Ollama not found. Install from https://ollama.ai for free local LLM inference."
    echo "   The system works without it but will use more Anthropic API credits."
fi

# 6. Create data directories
mkdir -p data/historical logs
touch data/.gitkeep logs/.gitkeep
echo "✓ Data directories created"

# 7. Install launchd agent (optional)
echo ""
read -p "→ Install launchd agent for auto-start on login? (y/n) " INSTALL_LAUNCHD
if [[ $INSTALL_LAUNCHD == "y" ]]; then
    USERNAME=$(whoami)
    SCRIPT_DIR=$(pwd)
    HOME_DIR="$HOME"
    cat > "$HOME_DIR/start_tradingbot.sh" <<EOF
#!/bin/bash
cd "$SCRIPT_DIR" || exit 1
exec "$SCRIPT_DIR/venv/bin/python" "$SCRIPT_DIR/main.py"
EOF
    chmod +x "$HOME_DIR/start_tradingbot.sh"

    # Replace placeholders in plist
    sed "s|YOUR_USERNAME|$USERNAME|g; s|__REPO_DIR__|$SCRIPT_DIR|g; s|__HOME_DIR__|$HOME_DIR|g" \
        com.tradingbot.plist > ~/Library/LaunchAgents/com.tradingbot.plist

    launchctl load ~/Library/LaunchAgents/com.tradingbot.plist 2>/dev/null || true
    echo "✓ launchd agent installed. Bot will start automatically at login."
    echo "  To stop:  launchctl unload ~/Library/LaunchAgents/com.tradingbot.plist"
    echo "  To start: launchctl load ~/Library/LaunchAgents/com.tradingbot.plist"
fi

echo ""
echo "================================================"
echo "  Setup complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "  1. Edit .env with your API keys"
echo "  2. Run backtest:  python backtesting/strategy.py --ticker AAPL"
echo "  3. Start bot:     python main.py"
echo "  4. Dashboard:     streamlit run dashboard/app.py"
echo ""
echo "Get free API keys:"
echo "  Alpaca:       https://alpaca.markets (paper trading, no signup cost)"
echo "  Finnhub:      https://finnhub.io/register"
echo "  Anthropic:    https://console.anthropic.com"
