# SPDX-License-Identifier: MIT
"""Thin, read-only HTTP client for RustChain's public endpoints.

Everything here is unauthenticated and read-only — these are the same public
surfaces a human can hit in a browser (rustchain.org/facts.json, /payouts.json,
/api/miners, /health). No keys, no writes, no wallet operations. The LangChain
tools in ``tools.py`` wrap these methods; this module is deliberately
framework-free so it can be tested (and reused) without LangChain installed.
"""
from __future__ import annotations

import datetime
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


# --- canonical hall-of-fame contract (shared by sync + async clients) ----
# RustChain rewards antiquity: the older / rarer the attesting hardware, the
# higher its multiplier. The hall of fame is the leaderboard of that — and it
# is derived from the *keyless* /api/miners surface (the /hall/leaderboard
# endpoint is auth-gated, which would break the package's read-only / keyless
# contract), so both clients funnel the same miner list through one reshape and
# return byte-identical output.
def _as_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _as_int(v) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _reshape_hall(miners, limit: int = 10) -> list:
    """Rank attesting miners into a hall of fame — oldest / most-prized first.

    Accepts the raw ``/api/miners`` payload (``{"miners": [...]}`` or a bare
    list), ranks by antiquity multiplier (the chain's own reward for age/rarity)
    with the earliest first-attestation breaking ties, and returns the canonical
    ``{rank, miner, device, hardware, antiquity_multiplier, first_attest,
    days_attesting}`` shape. ``first_attest`` is a ``YYYY-MM-DD`` day (or
    ``None``); ``days_attesting`` is the whole-day span between first and last
    attestation (or ``None`` when unknown).
    """
    if isinstance(miners, dict):
        miners = miners.get("miners", [])
    if not isinstance(miners, list):
        miners = []
    limit = max(1, min(int(limit), 50))

    # highest antiquity first; older machine (smaller first_attest) breaks ties
    ranked = sorted(
        miners,
        key=lambda m: (-_as_float(m.get("antiquity_multiplier")),
                       _as_int(m.get("first_attest")) or 1 << 62),
    )[:limit]

    out = []
    for i, m in enumerate(ranked, 1):
        first = _as_int(m.get("first_attest"))
        last = _as_int(m.get("last_attest"))
        days = max(0, (last - first) // 86400) if first and last else None
        first_day = (
            datetime.datetime.fromtimestamp(
                first, datetime.timezone.utc).strftime("%Y-%m-%d")
            if first else None
        )
        out.append({
            "rank": i,
            "miner": m.get("miner"),
            "device": m.get("device_family") or m.get("device_arch") or "unknown",
            "hardware": m.get("hardware_type") or "Unknown/Other",
            "antiquity_multiplier": m.get("antiquity_multiplier"),
            "first_attest": first_day,
            "days_attesting": days,
        })
    return out


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

    def hall_of_fame(self, limit: int = 10) -> list:
        """RustChain hall of fame — the oldest / most-prized attesting hardware.

        Keyless: derived from the public ``/api/miners`` surface and ranked by
        antiquity multiplier (the ``/hall/leaderboard`` endpoint is auth-gated).
        Returns the shared :func:`_reshape_hall` contract, so the async client
        returns the identical list.
        """
        return _reshape_hall(self.miners(), limit)

    def beacon_agents(self) -> list:
        """Registered Beacon agent-identity cards (id ``bcn_<hex>``, name, status)."""
        data = self._get_json("/beacon/api/agents")
        return data if isinstance(data, list) else []

    def beacon_contracts(self) -> list:
        """Open Beacon economic contracts (leases/offers between agents)."""
        data = self._get_json("/beacon/api/contracts")
        return data if isinstance(data, list) else []

    def provenance(self, agent_id: str) -> dict:
        """RIP-0310 Proof-of-Provenance status for a Beacon agent id.

        Read-only / keyless. Composes the deployed provenance signals for a
        ``bcn_<id>`` identity: its Beacon agent card (Agent layer) and any
        Beacon contracts it is party to (Economic layer). The Content-binding
        layer (a live ``BindingCert``) is specified in RIP-0310 but not yet
        deployed, so it is reported as such rather than fabricated. Returns a
        structured dict; ``summarize_provenance`` turns it into an agent-friendly
        string.
        """
        agent_id = (agent_id or "").strip()
        agents = self.beacon_agents()
        card = next((a for a in agents if a.get("agent_id") == agent_id), None)
        if card is None:
            # tolerate being given a display name instead of a bcn_ id
            card = next(
                (a for a in agents
                 if (a.get("name") or "").lower() == agent_id.lower()),
                None,
            )

        contracts = []
        if card is not None:
            aid = card.get("agent_id")
            for c in self.beacon_contracts():
                if c.get("from") == aid or c.get("to") == aid:
                    role = "payer" if c.get("from") == aid else "payee"
                    other = c.get("to") if role == "payer" else c.get("from")
                    contracts.append({
                        "id": c.get("id"),
                        "type": c.get("type"),
                        "amount": c.get("amount"),
                        "currency": c.get("currency"),
                        "state": c.get("state"),
                        "role": role,
                        "counterparty": other,
                    })

        return {
            "agent_id": agent_id,
            "found": card is not None,
            "registered_agents": len(agents),
            "identity": card,
            "contracts": contracts,
        }
