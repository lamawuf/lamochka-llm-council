# 🏛️ Lamochka LLM Council

Multi-LLM Council for collaborative AI decision-making with role-based debates.

A sophisticated system that orchestrates multiple LLMs (Claude, GPT-4, Gemini, Grok) in a structured debate format to produce well-reasoned, thoroughly-vetted answers.

## ✨ Features

- **Three-Stage Methodology**: First Opinions → Review & Debate → Final Synthesis
- **Role-Based Agents**: Chairman (arbitrator), Critic (devil's advocate), Researcher (fact-checker), Optimizer (practical improver)
- **Multi-Round Debates**: Automatic additional rounds when disagreement is high
- **Web Search Integration**: Researcher role can verify facts with live web search
- **Human-in-the-Loop**: Optional human intervention between stages
- **Manual Mode**: Input responses yourself (no API required)
- **Multi-Provider Support**: OpenRouter, OpenAI, Anthropic, Google, xAI, Ollama
- **CLI & API**: Command-line interface and REST API for programmatic access

## 🚀 Quick Start

### Option 1: One-Line Setup (Recommended)

```bash
git clone https://github.com/lamawuf/lamochka-llm-council.git
cd lamochka-llm-council
chmod +x setup.sh && ./setup.sh
```

The setup script will:
- Create `.env` from template
- Install dependencies
- Show you what API keys to add

### Option 2: Manual Setup

```bash
# Clone
git clone https://github.com/lamawuf/lamochka-llm-council.git
cd lamochka-llm-council

# Create .env
cp .env.example .env

# Install dependencies
pip install -r requirements.txt
# or: poetry install

# Add your API keys to .env (see below)
```

### API Keys

You need **at least one** provider. Edit `.env`:

```bash
# Option A: OpenRouter (ONE key for ALL models) — recommended
OPENROUTER_API_KEY=sk-or-v1-xxx   # Get at https://openrouter.ai/keys

# Option B: Direct API keys (any combination)
OPENAI_API_KEY=sk-xxx             # https://platform.openai.com/api-keys
ANTHROPIC_API_KEY=sk-ant-xxx      # https://console.anthropic.com/
GOOGLE_API_KEY=xxx                # https://aistudio.google.com/app/apikey
XAI_API_KEY=xai-xxx               # https://console.x.ai/
```

### Verify Setup

```bash
python main.py status   # Check which providers are available
```

### Run Your First Council

```bash
# Basic question
python main.py run "What's the best database for a real-time chat app?"

# With human-in-the-loop (you can intervene between debate rounds)
python main.py run --hitl "Should we use microservices or monolith?"

# Controversial topic (triggers multi-round debates)
python main.py run --threshold 3 "Is remote work better than office work?"

# Manual mode (no API keys needed — you input responses yourself)
python main.py run --manual "Your question here"

# Save output to file
python main.py run -o markdown -s result.md "Your question"
```

## 📋 Council Methodology

### Stage 1: First Opinions
Each council member provides their initial response to the prompt independently.

### Stage 2: Review & Debate
Responses are anonymized (A, B, C...) and each member reviews others based on their role:
- **Critic**: Identifies weaknesses, biases, logical fallacies
- **Researcher**: Fact-checks claims, provides sources
- **Optimizer**: Suggests practical improvements

If disagreement score > threshold (default 7/10), additional debate rounds occur.

### Stage 3: Final Synthesis
The Chairman (default: Claude) synthesizes all input into a comprehensive final answer, citing which inputs influenced the conclusion.

## 🎭 Roles

| Role | Default Model | Purpose |
|------|---------------|---------|
| **Chairman** | Claude | Arbitrator, synthesizes final answer |
| **Critic** | Grok | Devil's advocate, finds weaknesses |
| **Researcher** | Gemini | Fact-checker with web search |
| **Optimizer** | GPT-4 | Practical improver, action plans |

### Custom Roles

Create a `roles.json` file:

```json
[
  {"role": "chairman", "model": "claude"},
  {"role": "critic", "model": "grok"},
  {"role": "researcher", "model": "gemini"},
  {"role": "optimizer", "model": "gpt4"}
]
```

```bash
python main.py run -r roles.json "Your question"
```

## 🔧 CLI Commands

```bash
# Run council session
python main.py run "Your prompt"

# Options:
#   --file, -f        Read prompt from file
#   --roles, -r       Custom roles JSON file
#   --max-rounds, -m  Max debate rounds (1-5, default: 3)
#   --threshold, -t   Disagreement threshold (1-10, default: 7)
#   --hitl            Enable human-in-the-loop
#   --manual          Manual input mode (no API)
#   --no-search       Disable web search
#   --output, -o      Output format: markdown|json
#   --save, -s        Save result to file

# Check provider status
python main.py status

# List available roles
python main.py roles

# Show examples
python main.py example

# Start API server
python main.py serve --port 8000
```

## 🌐 API Usage

Start the server:

```bash
python main.py serve
```

### Endpoints

```bash
# Health check
GET /health

# List providers
GET /providers

# List roles
GET /roles

# Run council (synchronous)
POST /council/run
{
  "prompt": "Your question",
  "max_debate_rounds": 3,
  "disagreement_threshold": 7.0
}

# Start async council
POST /council/start
{
  "prompt": "Your question"
}
# Returns: {"session_id": "uuid", "status": "pending"}

# Check status
GET /council/status/{session_id}

# Get result
GET /council/result/{session_id}
```

## 🔑 Provider Configuration

### OpenRouter (Recommended)
Single API key for all models:

```env
OPENROUTER_API_KEY=sk-or-v1-xxx
```

### Direct APIs

```env
OPENAI_API_KEY=sk-xxx
ANTHROPIC_API_KEY=sk-ant-xxx
GOOGLE_API_KEY=xxx
XAI_API_KEY=xai-xxx
```

### Local (Ollama)

```bash
# Install Ollama: https://ollama.ai
ollama pull llama3.1:70b

# In .env:
OLLAMA_HOST=http://localhost:11434
```

### Web Search

```env
# Optional: Serper API (faster, more reliable)
SERPER_API_KEY=xxx

# Falls back to DuckDuckGo (free, no key needed)
```

## 🚂 Railway Deployment

1. Create a new Railway project
2. Connect your GitHub repo
3. Add environment variables:
   - `OPENROUTER_API_KEY`
   - Any other API keys needed
4. Deploy!

The `railway.toml` configures:
- Nixpacks build
- Health checks at `/health`
- Automatic restarts on failure

## 📁 Project Structure

```
lamochka-llm-council/
├── main.py              # CLI entry point
├── council.py           # Core council logic
├── config.py            # Settings and defaults
├── api.py               # FastAPI server
├── search.py            # Web search integration
├── providers/           # LLM provider implementations
│   ├── base.py          # Base provider interface
│   ├── openrouter.py    # OpenRouter (unified)
│   ├── openai_provider.py
│   ├── anthropic_provider.py
│   ├── google_provider.py
│   ├── xai_provider.py  # Grok
│   ├── ollama_provider.py
│   └── factory.py       # Provider factory
├── schemas/             # Pydantic models
│   └── council.py       # Data structures
├── prompts/             # YAML prompt templates
│   ├── roles.yaml       # Role definitions
│   └── stages.yaml      # Stage prompts
├── tests/               # Unit tests
├── .env.example         # Environment template
├── pyproject.toml       # Dependencies (Poetry)
├── railway.toml         # Railway deployment
├── Dockerfile           # Container build
└── README.md
```

## 🧪 Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=. --cov-report=html

# Specific test file
pytest tests/test_council.py -v
```

## 🔒 Security Scanner

Before running code from unknown repos, scan for malicious content:

```bash
# Scan any GitHub repository
python main.py check-repo https://github.com/user/repo

# Exit code: 0 = safe, 1 = issues found
```

**What it checks:**
- 🔴 **HIGH**: Obfuscated code (drainers), hardcoded wallet keys, curl|bash, data exfil
- 🟡 **MEDIUM**: Hardcoded API keys, base64 decode, shell=True subprocess
- 🟢 **LOW**: exec/eval usage, dangerous rm -rf patterns

**Also checks:**
- Typosquatting in dependencies (fake packages like `reqeusts` instead of `requests`)
- `.env` files committed or not gitignored

See [SECURITY_GUIDE.md](./SECURITY_GUIDE.md) for manual security audit steps.

## 📊 Best Practices

1. **Clear Prompts**: Be specific about what you're asking
2. **Right Roles**: Customize roles for your domain
3. **HITL for Critical Decisions**: Use `--hitl` for important decisions
4. **Review Debate Scores**: High disagreement (>7) suggests complex tradeoffs
5. **Save Results**: Use `-s` to keep records of decisions

## 🔮 Roadmap

- [ ] **V2**: Full web UI with session history
- [ ] Streaming responses
- [ ] Persistent session storage
- [ ] Custom prompt templates via UI
- [ ] Integration with knowledge bases
- [ ] Multi-language support

## 📜 License

MIT License - see LICENSE file

## 🙏 Credits

Based on [llm-council-plus](https://github.com/jacob-bd/llm-council-plus) with significant enhancements for role-based methodology and practical deployment.

---

Built with ❤️ by [Ruslan Lama](https://github.com/llamich)
