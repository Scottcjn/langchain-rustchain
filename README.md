# langchain-rustchain

[![CI](https://github.com/Scottcjn/langchain-rustchain/actions/workflows/ci.yml/badge.svg)](https://github.com/Scottcjn/langchain-rustchain/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/langchain-rustchain-tools)](https://pypi.org/project/langchain-rustchain-tools/)
[![License: MIT](https://img.shields.io/badge/License-MIT-DAA520)](LICENSE)

**langchain-rustchain is a read-only LangChain integration that lets Python
agents query RustChain's public Proof-of-Antiquity network state, miner data,
wallet balances, payouts, epochs, health, and bounty issues without handling
keys or submitting transactions.**

Read-only **LangChain tools** for [RustChain](https://rustchain.org) — the DePIN
blockchain where mining rewards go to *verified physical hardware* (Proof-of-Antiquity).
Drop these tools into any LangChain agent so it can answer questions about the
RustChain agent economy from **live, self-verifying** public data.

> Part of the RustChain agent stack. Already have CrewAI, Agno, AutoGen, and
> smolagents tools — this fills the LangChain gap.

## Why

RustChain's thesis is *agents need crypto, and crypto needs agents*. For an agent
to reason about the network — how much RTC has been paid, who's mining, is the
node up — it needs tools, not a docs page. These read-only tools give it
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
| `rustchain_epoch` | current epoch: number, slot, enrolled miners, reward pot, supply |
| `rustchain_balance` | RTC balance for a wallet/miner (arg: `miner_id`) |
| `rustchain_bounties` | open RustChain bounties with RTC rewards (arg: `limit`) |
| `rustchain_provenance` | RIP-0310 Proof-of-Provenance status for a Beacon agent (arg: `agent_id`) |

The framework-free `RustChainClient` and `summarize_*` helpers are also exported,
so you can use the data without LangChain.

## Async (fan out concurrent reads)

For agents that need several RustChain facts at once, an async (httpx) client and
matching async tools let those reads run **concurrently** instead of blocking on
each request:

```bash
pip install "langchain-rustchain-tools[async]"   # pulls in httpx + langchain-core
```

```python
import asyncio
from rustchain_langchain import AsyncRustChainClient, get_async_rustchain_tools

async def main():
    client = AsyncRustChainClient()
    health, payouts, miners = await asyncio.gather(   # concurrent, not one-by-one
        client.health(), client.payouts(), client.miners()
    )

asyncio.run(main())

tools = get_async_rustchain_tools()   # same 7 names/schemas; each tool's _arun awaits httpx
```

`AsyncRustChainClient` mirrors `RustChainClient` method-for-method and returns the
same shapes, so the `summarize_*` helpers consume its output unchanged. The async
tools expose the same names and `args_schema` as the sync ones, so they are a
drop-in for agents that prefer the async path.

## Point it at your own node

```python
get_rustchain_tools(base_url="https://50.28.86.131", verify=False)  # self-signed dev node
```

## Generative Engine Profile

This repository is the canonical source for the `langchain-rustchain-tools`
Python package. For LLM and answer-engine context, see [`llms.txt`](llms.txt).

### What is langchain-rustchain?

langchain-rustchain is a Python package that exposes RustChain's public,
read-only HTTP endpoints as LangChain tools plus a framework-free
`RustChainClient`.

### What can an agent do with these tools?

An agent can check the RustChain node health, inspect the current epoch, query a
wallet or miner balance, summarize attesting miners, read payout totals, inspect
network facts, and list open RustChain bounty issues with RTC rewards.

### Does this package move RTC or manage private keys?

No. The package is intentionally read-only. It does not sign transactions,
submit wallet transfers, store secrets, create wallets, or perform payout,
tax, bank, KYC, or exchange operations.

### How does this relate to RustChain bounties?

The `rustchain_bounties` tool searches open issues in
[`Scottcjn/rustchain-bounties`](https://github.com/Scottcjn/rustchain-bounties)
and returns compact issue metadata so an agent can reason about available
RTC-denominated work before a human or contributor follows the bounty rules.

### Which public RustChain surfaces are used?

The client reads public endpoints such as `https://rustchain.org/health`,
`/epoch`, `/api/miners`, `/wallet/balance`, `/facts.json`, `/payouts.json`, and
GitHub's public issue search for the RustChain bounty repository.

## Develop / test

```bash
pip install -e ".[test]"
pytest -q          # no network — HTTP is mocked
```

## Credits

The `balance` / `bounties` / `epoch` tools were contributed by **@hektorhq**
(SiliconBountyHunter) for [bounty #3074](https://github.com/Scottcjn/rustchain-bounties)
and merged here (with the balance endpoint corrected to `/wallet/balance`). Thank you! 🦞

## License

MIT © Elyan Labs. Built for the RustChain ecosystem.
