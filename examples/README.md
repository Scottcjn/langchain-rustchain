# Examples

Runnable demos for `langchain-rustchain`.

## `how_big_is_rustchain.py`

A LangChain **ReAct agent** that answers *"How big is RustChain right now?"* by
deciding which read-only RustChain tools to call (network activity, current
epoch, attesting miners, payouts) and summarising what it reads.

### Run it with an LLM (real ReAct loop)

```bash
pip install -e ".[langchain]" langchain-openai langgraph
export OPENAI_API_KEY=sk-...
python examples/how_big_is_rustchain.py
```

The agent reasons about which tools to call and writes the answer itself. Any
chat model with tool/function calling works — pass `--model` to choose one
(default `gpt-4o-mini`).

### Run it offline (no API key)

```bash
python examples/how_big_is_rustchain.py --offline
```

This skips the LLM and calls the same read tools directly, composing the answer
deterministically. It still queries the live, public, read-only RustChain
endpoints, so you get real numbers without an LLM. If no API key is found the
script falls back to this mode automatically.

### Options

| Flag         | Default                  | Meaning                                      |
|--------------|--------------------------|----------------------------------------------|
| `--offline`  | off                      | Skip the LLM; compose from the tools directly |
| `--base-url` | `https://rustchain.org`  | Point at a different RustChain node           |
| `--model`    | `gpt-4o-mini`            | Chat model for the ReAct agent                |

Everything here is **read-only and keyless** — the same public surfaces a human
can hit in a browser. No wallet writes, no secrets.
