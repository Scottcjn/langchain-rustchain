# SPDX-License-Identifier: MIT
"""LangChain tools that let an agent read RustChain's live, public state.

Read-only and keyless. The ``summarize_*`` functions are plain and unit-tested
on their own; the LangChain ``BaseTool`` wrappers below call the client and feed
the JSON through them, returning a compact, agent-friendly string (agents do
better with a 3-line summary than a 5KB JSON blob).

Usage:

    from rustchain_langchain import get_rustchain_tools
    tools = get_rustchain_tools()              # list[BaseTool], drop into any agent
    # e.g. with langgraph / AgentExecutor / bind_tools(...)
"""
from __future__ import annotations

from typing import List

from .client import RustChainClient
from .async_client import AsyncRustChainClient


# --- pure summarizers (framework-free, unit-tested) ---------------------
def summarize_network(data: dict) -> str:
    facts = {f.get("id"): f for f in data.get("facts", [])}
    a = (facts.get("onchain_activity") or {}).get("value", {})
    parts = [f"RustChain on-chain activity (as of {data.get('as_of', '?')}):"]
    if a:
        parts.append(
            f"- {a.get('wallet_transfers', '?')} wallet transfers, "
            f"{a.get('rtc_moved_in_transfers', '?')} RTC moved across "
            f"{a.get('distinct_wallets', '?')} distinct wallets"
        )
        parts.append(f"- {a.get('ledger_entries_total', '?')} total ledger entries")
    if not a:
        parts.append(f"- {data.get('tagline', 'verifiable agent economy')}")
    return "\n".join(parts)


def summarize_payouts(data: dict) -> str:
    return (
        f"RustChain payouts: {data.get('total_paid_rtc', '?')} RTC paid to "
        f"{data.get('unique_recipients', '?')} distinct recipients across "
        f"{data.get('transactions', '?')} transactions "
        f"(updated {data.get('updated_at', '?')})."
    )


def summarize_miners(data) -> str:
    # /api/miners returns {"miners": [...]}, but tolerate a bare list too.
    if isinstance(data, list):
        miners = data
    elif isinstance(data, dict):
        miners = data.get("miners", [])
    else:
        miners = []
    if not isinstance(miners, list):
        miners = []
    n = len(miners)
    from collections import Counter

    arches = Counter(m.get("device_arch", "unknown") for m in miners)
    top = ", ".join(f"{a}×{c}" for a, c in arches.most_common(6))
    return (
        f"RustChain has {n} attesting miner(s). By architecture: {top or 'n/a'}. "
        "Older/exotic hardware earns higher antiquity multipliers."
    )


def summarize_health(data: dict) -> str:
    return (
        f"RustChain node health: ok={data.get('ok', '?')}, "
        f"db_rw={data.get('db_rw', '?')}, version={data.get('version', '?')}, "
        f"backup_age_hours={round(data.get('backup_age_hours', 0), 1)}."
    )


def summarize_balance(data: dict) -> str:
    return (
        f"Wallet '{data.get('miner_id', '?')}' holds "
        f"{data.get('amount_rtc', '?')} RTC."
    )


def summarize_epoch(data: dict) -> str:
    return (
        f"RustChain epoch {data.get('epoch', '?')} (slot {data.get('slot', '?')}): "
        f"{data.get('enrolled_miners', '?')} enrolled miners, "
        f"epoch reward pot {data.get('epoch_pot', '?')} RTC, "
        f"{data.get('blocks_per_epoch', '?')} blocks/epoch, "
        f"total supply {data.get('total_supply_rtc', '?')} RTC."
    )


def summarize_bounties(items) -> str:
    if not isinstance(items, list) or not items:
        return "No open RustChain bounties found."
    lines = [f"{len(items)} open RustChain bounties (newest first):"]
    for b in items:
        lines.append(
            f"- #{b.get('number')} [{b.get('reward')}] {b.get('title')} — {b.get('url')}"
        )
    return "\n".join(lines)


# --- LangChain tool wrappers --------------------------------------------
def get_rustchain_tools(
    base_url: str = "https://rustchain.org",
    timeout: int = 15,
    verify: bool = True,
) -> List["object"]:
    """Return a list of LangChain ``BaseTool`` objects for RustChain.

    Imported lazily so the client/summarizers stay usable without LangChain.
    Requires ``langchain-core``.
    """
    from langchain_core.tools import BaseTool  # lazy import
    from pydantic import BaseModel, Field
    from typing import Type

    client = RustChainClient(base_url=base_url, timeout=timeout, verify=verify)

    def _make(name: str, description: str, fetch, summarize):
        class _Tool(BaseTool):
            def _run(self, *args, **kwargs) -> str:
                try:
                    return summarize(fetch())
                except Exception as e:  # never raise inside an agent loop
                    return f"RustChain query failed ({type(e).__name__}): {e}"

            async def _arun(self, *args, **kwargs) -> str:
                return self._run(*args, **kwargs)

        return _Tool(name=name, description=description)

    # --- argument-taking tools (balance, bounties) ----------------------
    class _WalletInput(BaseModel):
        miner_id: str = Field(description="Wallet address or miner id to query, e.g. 'dual-g4-125'")

    class _BalanceTool(BaseTool):
        name: str = "rustchain_balance"
        description: str = (
            "Check the RTC balance of a RustChain wallet/miner. "
            "Input: miner_id (a wallet address or miner id)."
        )
        args_schema: Type[BaseModel] = _WalletInput

        def _run(self, miner_id: str) -> str:
            try:
                return summarize_balance(client.balance(miner_id))
            except Exception as e:
                return f"RustChain query failed ({type(e).__name__}): {e}"

        async def _arun(self, miner_id: str) -> str:
            return self._run(miner_id)

    class _BountyInput(BaseModel):
        limit: int = Field(default=10, description="Max bounties to return (1-50)")

    class _BountiesTool(BaseTool):
        name: str = "rustchain_bounties"
        description: str = (
            "List open RustChain bounties (GitHub issues with RTC rewards). "
            "Input: limit (default 10). Returns number, reward, title, URL."
        )
        args_schema: Type[BaseModel] = _BountyInput

        def _run(self, limit: int = 10) -> str:
            try:
                return summarize_bounties(client.bounties(limit))
            except Exception as e:
                return f"RustChain query failed ({type(e).__name__}): {e}"

        async def _arun(self, limit: int = 10) -> str:
            return self._run(limit)

    return [
        _make(
            "rustchain_network_stats",
            "Get RustChain's live on-chain activity (wallet transfers, RTC moved, "
            "distinct wallets). Use when asked about RustChain network activity or size.",
            client.network_stats,
            summarize_network,
        ),
        _make(
            "rustchain_payouts",
            "Get total RTC paid out and the number of distinct recipients on RustChain. "
            "Use for questions about how much RustChain has paid contributors/miners.",
            client.payouts,
            summarize_payouts,
        ),
        _make(
            "rustchain_miners",
            "Get the current attesting miners and a breakdown by hardware architecture "
            "(PowerPC G4/G5, POWER8, x86, Apple Silicon, etc.). Use for questions about "
            "who is mining or what hardware is on the network.",
            client.miners,
            summarize_miners,
        ),
        _make(
            "rustchain_node_health",
            "Check whether the RustChain node is healthy (ok, db read-write, version, "
            "backup age). Use to verify the network is up before relying on it.",
            client.health,
            summarize_health,
        ),
        _make(
            "rustchain_epoch",
            "Get the current RustChain epoch: number, slot, enrolled miners, epoch "
            "reward pot, and total supply. Use for questions about the current mining round.",
            client.epoch,
            summarize_epoch,
        ),
        _BalanceTool(),
        _BountiesTool(),
    ]


def get_async_rustchain_tools(
    base_url: str = "https://rustchain.org",
    timeout: int = 15,
    verify: bool = True,
) -> List["object"]:
    """Return the same RustChain tools backed by an async (httpx) client.

    Identical names, descriptions, schemas and summaries as
    :func:`get_rustchain_tools`, but each tool's ``_arun`` awaits
    :class:`AsyncRustChainClient`, so an agent can issue several RustChain reads
    concurrently (e.g. under ``asyncio.gather``) instead of one blocking request
    at a time. ``_run`` bridges to the coroutine for sync callers when no event
    loop is already running. Requires ``langchain-core`` (and ``httpx`` at call
    time).
    """
    import asyncio

    from langchain_core.tools import BaseTool  # lazy import
    from pydantic import BaseModel, Field
    from typing import Type

    client = AsyncRustChainClient(base_url=base_url, timeout=timeout, verify=verify)

    def _bridge(coro_factory, *args, **kwargs):
        """Run an async tool body from a sync ``_run``: execute the coroutine
        when no loop is spinning, otherwise tell the caller to use the async
        path. Never raises into an agent loop."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro_factory(*args, **kwargs))
        return (
            "call this tool asynchronously (await ainvoke/_arun) — an event "
            "loop is already running on this thread."
        )

    def _make(name: str, description: str, afetch, summarize):
        class _AsyncTool(BaseTool):
            async def _arun(self, *args, **kwargs) -> str:
                try:
                    return summarize(await afetch())
                except Exception as e:  # never raise inside an agent loop
                    return f"RustChain query failed ({type(e).__name__}): {e}"

            def _run(self, *args, **kwargs) -> str:
                return _bridge(self._arun, *args, **kwargs)

        return _AsyncTool(name=name, description=description)

    # --- argument-taking async tools (balance, bounties) ----------------
    class _WalletInput(BaseModel):
        miner_id: str = Field(description="Wallet address or miner id to query, e.g. 'dual-g4-125'")

    class _AsyncBalanceTool(BaseTool):
        name: str = "rustchain_balance"
        description: str = (
            "Check the RTC balance of a RustChain wallet/miner. "
            "Input: miner_id (a wallet address or miner id)."
        )
        args_schema: Type[BaseModel] = _WalletInput

        async def _arun(self, miner_id: str) -> str:
            try:
                return summarize_balance(await client.balance(miner_id))
            except Exception as e:
                return f"RustChain query failed ({type(e).__name__}): {e}"

        def _run(self, miner_id: str) -> str:
            return _bridge(self._arun, miner_id)

    class _BountyInput(BaseModel):
        limit: int = Field(default=10, description="Max bounties to return (1-50)")

    class _AsyncBountiesTool(BaseTool):
        name: str = "rustchain_bounties"
        description: str = (
            "List open RustChain bounties (GitHub issues with RTC rewards). "
            "Input: limit (default 10). Returns number, reward, title, URL."
        )
        args_schema: Type[BaseModel] = _BountyInput

        async def _arun(self, limit: int = 10) -> str:
            try:
                return summarize_bounties(await client.bounties(limit))
            except Exception as e:
                return f"RustChain query failed ({type(e).__name__}): {e}"

        def _run(self, limit: int = 10) -> str:
            return _bridge(self._arun, limit)

    return [
        _make(
            "rustchain_network_stats",
            "Get RustChain's live on-chain activity (wallet transfers, RTC moved, "
            "distinct wallets). Use when asked about RustChain network activity or size.",
            client.network_stats,
            summarize_network,
        ),
        _make(
            "rustchain_payouts",
            "Get total RTC paid out and the number of distinct recipients on RustChain. "
            "Use for questions about how much RustChain has paid contributors/miners.",
            client.payouts,
            summarize_payouts,
        ),
        _make(
            "rustchain_miners",
            "Get the current attesting miners and a breakdown by hardware architecture "
            "(PowerPC G4/G5, POWER8, x86, Apple Silicon, etc.). Use for questions about "
            "who is mining or what hardware is on the network.",
            client.miners,
            summarize_miners,
        ),
        _make(
            "rustchain_node_health",
            "Check whether the RustChain node is healthy (ok, db read-write, version, "
            "backup age). Use to verify the network is up before relying on it.",
            client.health,
            summarize_health,
        ),
        _make(
            "rustchain_epoch",
            "Get the current RustChain epoch: number, slot, enrolled miners, epoch "
            "reward pot, and total supply. Use for questions about the current mining round.",
            client.epoch,
            summarize_epoch,
        ),
        _AsyncBalanceTool(),
        _AsyncBountiesTool(),
    ]
