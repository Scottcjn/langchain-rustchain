# SPDX-License-Identifier: MIT
"""Async, read-only HTTP client for RustChain's public endpoints.

The ``async`` twin of :class:`rustchain_langchain.client.RustChainClient`. It
exposes the same read-only, keyless surfaces (network stats, payouts, miners,
health, wallet balance, epoch, bounties) but every call is a coroutine backed by
``httpx.AsyncClient`` — so an agent can fan out several RustChain reads
concurrently with ``asyncio.gather(...)`` instead of blocking on each one.

``httpx`` is imported lazily inside the request path so the rest of the package
(the sync client, the framework-free ``summarize_*`` helpers) keeps working
without it installed — the same contract as the lazy ``langchain-core`` import
in ``tools.py``. Install the async extra with
``pip install langchain-rustchain-tools[async]``.
"""
from __future__ import annotations

from .client import (
    DEFAULT_BASE_URL,
    DEFAULT_TIMEOUT,
    _bounties_search_url,
    _reshape_bounty,
)


class AsyncRustChainClient:
    """Async read-only client for the RustChain public API.

    Mirrors :class:`~rustchain_langchain.client.RustChainClient` method-for-method,
    but every call is a coroutine. Returned shapes are identical, so the same
    framework-free ``summarize_*`` helpers consume the output unchanged.

    Args:
        base_url: RustChain node/site base URL (default https://rustchain.org).
        timeout: per-request timeout in seconds.
        verify: TLS verification (default True; set False only for self-signed
            dev nodes).
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
        verify: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.verify = verify

    async def _get_json(self, path: str):
        import httpx  # lazy: keep the rest of the package httpx-free

        url = f"{self.base_url}/{path.lstrip('/')}"
        async with httpx.AsyncClient(timeout=self.timeout, verify=self.verify) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.json()

    # --- public, read-only surfaces (async twins of RustChainClient) ----
    async def network_stats(self) -> dict:
        """Self-verifying on-chain activity facts (facts.json)."""
        return await self._get_json("/facts.json")

    async def payouts(self) -> dict:
        """Chain-computed payout totals + recipient counts (payouts.json)."""
        return await self._get_json("/payouts.json")

    async def metrics(self) -> dict:
        """Repo/ecosystem metrics snapshot (metrics.json)."""
        return await self._get_json("/metrics.json")

    async def miners(self) -> dict:
        """Currently attesting miners, with device arch + antiquity multipliers."""
        return await self._get_json("/api/miners")

    async def health(self) -> dict:
        """Node health (ok, db_rw, version, backup age)."""
        return await self._get_json("/health")

    async def balance(self, miner_id: str) -> dict:
        """RTC balance for a wallet / miner id.

        Uses the live /wallet/balance endpoint (the bare /balance path 404s).
        """
        import httpx  # lazy

        url = f"{self.base_url}/wallet/balance"
        async with httpx.AsyncClient(timeout=self.timeout, verify=self.verify) as client:
            resp = await client.get(url, params={"miner_id": miner_id})
            resp.raise_for_status()
            return resp.json()

    async def epoch(self) -> dict:
        """Current epoch: number, slot, enrolled miners, reward pot, total supply."""
        return await self._get_json("/epoch")

    async def bounties(self, limit: int = 10) -> list:
        """Open RustChain bounties (GitHub issues on Scottcjn/rustchain-bounties).

        Read-only GitHub search; returns a list of {number, title, reward, url,
        created}. Async twin of :meth:`RustChainClient.bounties` — same query,
        reward parser and output shape via the shared canonical helpers.
        """
        import httpx  # lazy

        limit = max(1, min(int(limit), 50))
        async with httpx.AsyncClient(timeout=self.timeout, verify=self.verify) as client:
            resp = await client.get(
                _bounties_search_url(limit),
                headers={"Accept": "application/vnd.github.v3+json"},
            )
            resp.raise_for_status()
            items = resp.json().get("items", [])[:limit]

        return [_reshape_bounty(it) for it in items]
