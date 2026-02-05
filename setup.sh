#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# Lamochka LLM Council — Setup Script
# ═══════════════════════════════════════════════════════════════════

set -e

echo "🏛️  Lamochka LLM Council Setup"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 not found. Please install Python 3.10+${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo -e "${GREEN}✓${NC} Python $PYTHON_VERSION found"

# Create .env if not exists
if [ ! -f .env ]; then
    echo ""
    echo -e "${YELLOW}📝 Creating .env file...${NC}"
    cp .env.example .env
    echo -e "${GREEN}✓${NC} .env created from .env.example"
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo -e "${YELLOW}⚠️  IMPORTANT: Add your API keys to .env${NC}"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    echo "You need AT LEAST ONE of these:"
    echo ""
    echo "  Option 1 (Recommended): OpenRouter — one key for all models"
    echo "    → Get key: https://openrouter.ai/keys"
    echo "    → Add to .env: OPENROUTER_API_KEY=sk-or-v1-xxx"
    echo ""
    echo "  Option 2: Direct API keys (any combination)"
    echo "    → OpenAI:    https://platform.openai.com/api-keys"
    echo "    → Anthropic: https://console.anthropic.com/"
    echo "    → Google:    https://aistudio.google.com/app/apikey"
    echo "    → xAI Grok:  https://console.x.ai/"
    echo ""
else
    echo -e "${GREEN}✓${NC} .env already exists"
fi

# Install dependencies
echo ""
echo -e "${YELLOW}📦 Installing dependencies...${NC}"

if command -v poetry &> /dev/null; then
    poetry install
    echo -e "${GREEN}✓${NC} Dependencies installed via Poetry"
else
    pip3 install -r requirements.txt 2>/dev/null || pip3 install typer rich httpx pydantic pydantic-settings python-dotenv openai anthropic google-generativeai tenacity duckduckgo-search aiohttp ollama
    echo -e "${GREEN}✓${NC} Dependencies installed via pip"
fi

# Verify setup
echo ""
echo -e "${YELLOW}🔍 Checking provider status...${NC}"
echo ""
python3 main.py status 2>/dev/null || echo -e "${YELLOW}Run 'python3 main.py status' after adding API keys${NC}"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo -e "${GREEN}✅ Setup complete!${NC}"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo "  1. Edit .env and add your API keys"
echo "  2. Run: python3 main.py status"
echo "  3. Try:  python3 main.py run \"Your question here\""
echo ""
echo "Examples:"
echo "  python3 main.py run \"What's the best database for my project?\""
echo "  python3 main.py run --hitl \"Should we use microservices?\""
echo "  python3 main.py example  # See more examples"
echo ""
