# gepa-ai-example

Multi-agent prompt optimization lab using [GEPA](https://pypi.org/project/gepa/).

## Getting Started

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- OpenAI API key

### Installation

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd gepa-ai-example
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   uv venv
   uv pip install -e .
   ```

3. Activate the virtual environment (required before running commands):
   ```bash
   source .venv/bin/activate  # macOS/Linux
   # Or use: .venv\Scripts\activate  # Windows
   ```

   **Alternative:** Skip activation and prefix commands with `uv run`:
   ```bash
   uv run python3 main.py --task math_aime
   ```

4. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

### Running the Optimizer

Run GEPA prompt optimization for a specific task:

```bash
python3 main.py --task <task_name> [options]
```

**Available tasks:**
- `math_aime` - AIME-style math problem solving
- `qa_multistep` - Multi-step question answering

**Options:**
- `--task` (required): Task name to optimize
- `--max-metric-calls`: Maximum metric evaluations (default: 150)
- `--reflection-lm`: Model for reflection/optimization (default: `openai/gpt-4o`)

**Example:**
```bash
python3 main.py --task math_aime --max-metric-calls 50
```

Results are saved to `artifacts/<task_name>/run-<timestamp>/`

### Custom Executors

Each task defines its own executor (the agent/LLM being tested). This enables:
- Direct API integrations (Anthropic, OpenAI, etc.)
- LangChain agents or chains
- Custom retry/caching logic
- Mock executors for testing

**Creating a custom executor:**

An executor is a callable that takes a list of messages and returns a string response:

```python
# In your task.py
from collections.abc import Sequence
from src.core.executor import ChatMessage

def my_executor(messages: Sequence[ChatMessage]) -> str:
    # messages is a list of {"role": str, "content": str}
    # Return the model's response as a string
    ...

TaskImpl = SimpleTask(
    name="my_task",
    ...
    executor=my_executor,
)
```

See `examples/custom_executor.py` for complete examples including Anthropic and mock implementations.
