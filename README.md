# langchain-rustchain

[![CI](https://github.com/Scottcjn/langchain-rustchain/actions/workflows/ci.yml/badge.svg)](https://github.com/Scottcjn/langchain-rustchain/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/langchain-rustchain-tools)](https://pypi.org/project/langchain-rustchain-tools/)
[![License: MIT](https://img.shields.io/badge/License-MIT-DAA520)](LICENSE)

Read-only **LangChain tools** for [RustChain](https://rustchain.org) — the DePIN
blockchain where mining rewards go to *verified physical hardware* (Proof-of-Antiquity).
Drop these tools into any LangChain agent so it can answer questions about the
RustChain agent economy from **live, self-verifying** public data.

> Part of the RustChain agent stack. Already have CrewAI, Agno, AutoGen, and
> smolagents tools — this fills the LangChain gap.

## Why

RustChain's thesis is *agents need crypto, and crypto needs agents*. For an agent
to reason about the network — how much RTC has been paid, who's mining, is the
node up — it needs tools, not a docs page. These four read-only tools give it
exactly that. **No keys, no writes, no wallet operations** — same public surfaces
you can open in a browser.

## Install

```bash
pip install langchain-rustchain-tools        # + langchain-core for the tools
```

## Use

```python
from rustchain_langchain import get_rustchain_tools

tools = get_rustchain_tools()          # list[BaseTool]

# bind to any LangChain chat model / agent:
# llm.bind_tools(tools)  |  AgentExecutor(...tools=tools)  |  langgraph create_react_agent(llm, tools)
```

Each tool returns a compact, agent-friendly summary (agents reason better on a
3-line summary than a 5 KB JSON blob):

| Tool | Answers |
|------|---------|
| `rustchain_network_stats` | on-chain activity: transfers, RTC moved, distinct wallets |
| `rustchain_payouts` | total RTC paid + distinct recipients (chain-computed) |
| `rustchain_miners` | who's attesting, broken down by hardware architecture |
| `rustchain_node_health` | is the node up (ok / db_rw / version / backup age) |

The framework-free `RustChainClient` and `summarize_*` helpers are also exported,
so you can use the data without LangChain.

## Point it at your own node

```python
get_rustchain_tools(base_url="https://50.28.86.131", verify=False)  # self-signed dev node
```

## Develop / test

```bash
pip install -e ".[test]"
pytest -q          # no network — HTTP is mocked
```

## License

MIT © Elyan Labs. Built for the RustChain ecosystem.
