# SPDX-License-Identifier: MIT
"""Thin, read-only HTTP client for RustChain's public endpoints.

Everything here is unauthenticated and read-only — these are the same public
surfaces a human can hit in a browser (rustchain.org/facts.json, /payouts.json,
/api/miners, /health). No keys, no writes, no wallet operations. The LangChain
tools in ``tools.py`` wrap these methods; this module is deliberately
framework-free so it can be tested (and reused) without LangChain installed.
"""
from __future__ import annotations

import re

import requests

DEFAULT_BASE_URL = "https://rustchain.org"
DEFAULT_TIMEOUT = 15

# --- canonical bounty contract (shared by sync + async clients) ---------
# One query, one reward parser, one output shape so the sync client, the async
# client and ``summarize_bounties`` can never drift apart. The search is scoped
# to issues actually carrying the ``bounty`` label, and the reward is read from
# the title *and* the body (titles like ``[BOUNTY: 50 RTC]`` are common) with
# decimal amounts preserved.
BOUNTIES_SEARCH_QUERY = (
    "repo:Scottcjn/rustchain-bounties+state:open+is:issue+label:bounty"
)
_REWARD_RE = re.compile(r"(\d+(?:\.\d+)?)\s*RTC", re.IGNORECASE)


def _parse_reward(title: str, body: str) -> str:
    """Extract an ``"<amount> RTC"`` reward from a bounty issue.

    Looks at the title first (the canonical place for ``[BOUNTY: 50 RTC]``),
    then the body, and keeps decimal amounts (``2.5 RTC``). Falls back to
    ``"see issue"`` when no amount is stated.
    """
    for text in (title or "", body or ""):
        m = _REWARD_RE.search(text)
        if m:
            return f"{m.group(1)} RTC"
    return "see issue"


def _reshape_bounty(item: dict) -> dict:
    """Map a raw GitHub issue dict to the canonical bounty shape.

    The single source of truth for ``{number, title, reward, url, created}`` —
    both :class:`RustChainClient` and the async client funnel through this so
    their output is byte-for-byte identical and ``summarize_bounties`` consumes
    either unchanged.
    """
    title = item.get("title") or ""
    body = item.get("body", "") or ""
    return {
        "number": item.get("number"),
        "title": title[:100],
        "reward": _parse_reward(title, body),
        "url": item.get("html_url"),
        "created": (item.get("created_at") or "")[:10],
    }


def _bounties_search_url(limit: int) -> str:
    """Build the canonical GitHub search URL for open RustChain bounties."""
    limit = max(1, min(int(limit), 50))
    return (
        "https://api.github.com/search/issues?"
        f"q={BOUNTIES_SEARCH_QUERY}&"
        f"per_page={limit}&sort=created&order=desc"
    )


class RustChainClient:
    """Read-only client for the RustChain public API.

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

    def _get_json(self, path: str) -> dict:
        url = f"{self.base_url}/{path.lstrip('/')}"
        resp = requests.get(url, timeout=self.timeout, verify=self.verify)
        resp.raise_for_status()
        return resp.json()

    # --- public, read-only surfaces -------------------------------------
    def network_stats(self) -> dict:
        """Self-verifying on-chain activity facts (facts.json)."""
        return self._get_json("/facts.json")

    def payouts(self) -> dict:
        """Chain-computed payout totals + recipient counts (payouts.json)."""
        return self._get_json("/payouts.json")

    def metrics(self) -> dict:
        """Repo/ecosystem metrics snapshot (metrics.json)."""
        return self._get_json("/metrics.json")

    def miners(self) -> dict:
        """Currently attesting miners, with device arch + antiquity multipliers."""
        return self._get_json("/api/miners")

    def health(self) -> dict:
        """Node health (ok, db_rw, version, backup age)."""
        return self._get_json("/health")

    def balance(self, miner_id: str) -> dict:
        """RTC balance for a wallet / miner id.

        Uses the live /wallet/balance endpoint (the bare /balance path 404s).
        """
        url = f"{self.base_url}/wallet/balance"
        resp = requests.get(
            url, params={"miner_id": miner_id}, timeout=self.timeout, verify=self.verify
        )
        resp.raise_for_status()
        return resp.json()

    def epoch(self) -> dict:
        """Current epoch: number, slot, enrolled miners, reward pot, total supply."""
        return self._get_json("/epoch")

    def bounties(self, limit: int = 10) -> list:
        """Open RustChain bounties (GitHub issues on Scottcjn/rustchain-bounties).

        Read-only GitHub search; returns a list of {number, title, reward, url,
        created}. Query, reward parsing and output shape are the shared canonical
        helpers (:func:`_bounties_search_url`, :func:`_reshape_bounty`) so the
        async client returns the identical contract.
        """
        limit = max(1, min(int(limit), 50))
        resp = requests.get(
            _bounties_search_url(limit), timeout=self.timeout,
            headers={"Accept": "application/vnd.github.v3+json"},
        )
        resp.raise_for_status()
        items = resp.json().get("items", [])[:limit]
        return [_reshape_bounty(it) for it in items]
