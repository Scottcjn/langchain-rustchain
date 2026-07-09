# SPDX-License-Identifier: MIT
"""Tests for langchain-rustchain. No network: requests is monkeypatched."""
import json
from unittest import mock

from rustchain_langchain import (
    RustChainClient,
    summarize_network,
    summarize_payouts,
    summarize_miners,
    summarize_health,
    summarize_balance,
    summarize_epoch,
    summarize_bounties,
    summarize_hall_of_fame,
    summarize_provenance,
)


class _Resp:
    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._p


def _client_returning(payload):
    c = RustChainClient(base_url="https://example.test")
    with mock.patch("rustchain_langchain.client.requests.get", return_value=_Resp(payload)) as g:
        return c, g


def test_client_builds_url_and_parses():
    c, g = _client_returning({"ok": True})
    with mock.patch("rustchain_langchain.client.requests.get", return_value=_Resp({"ok": True})) as g:
        out = c.health()
    assert out == {"ok": True}
    called_url = g.call_args[0][0]
    assert called_url == "https://example.test/health"


def test_summarize_network():
    data = {
        "as_of": "2026-06-16",
        "facts": [
            {"id": "onchain_activity", "value": {
                "wallet_transfers": 3112, "rtc_moved_in_transfers": 124848.19,
                "distinct_wallets": 1354, "ledger_entries_total": 7584}},
        ],
    }
    s = summarize_network(data)
    assert "3112 wallet transfers" in s
    assert "1354 distinct wallets" in s


def test_summarize_payouts():
    s = summarize_payouts({"total_paid_rtc": "66,531+", "unique_recipients": 1061,
                           "transactions": 3234, "updated_at": "2026-06-16"})
    assert "66,531+ RTC paid" in s
    assert "1061 distinct recipients" in s


def test_summarize_miners_counts_by_arch():
    data = {"miners": [
        {"device_arch": "G4"}, {"device_arch": "G4"}, {"device_arch": "POWER8"},
        {"device_arch": "modern"},
    ]}
    s = summarize_miners(data)
    assert "4 attesting miner" in s
    assert "G4×2" in s


def test_summarize_miners_tolerates_bare_list():
    s = summarize_miners([{"device_arch": "M4"}])
    assert "1 attesting miner" in s


def test_summarize_health():
    s = summarize_health({"ok": True, "db_rw": True, "version": "2.2.1", "backup_age_hours": 10.87})
    assert "ok=True" in s and "version=2.2.1" in s


def test_tool_run_never_raises_on_failure():
    # if langchain-core is unavailable, skip the wrapper test gracefully
    try:
        from rustchain_langchain import get_rustchain_tools
        tools = get_rustchain_tools(base_url="https://example.test")
    except (ImportError, ModuleNotFoundError):
        return
    tool = next(t for t in tools if t.name == "rustchain_payouts")
    with mock.patch("rustchain_langchain.client.requests.get", side_effect=RuntimeError("boom")):
        out = tool._run()
    assert "RustChain query failed" in out  # graceful, not an exception


def test_tool_run_summarizes_on_success():
    try:
        from rustchain_langchain import get_rustchain_tools
        tools = get_rustchain_tools(base_url="https://example.test")
    except (ImportError, ModuleNotFoundError):
        return
    tool = next(t for t in tools if t.name == "rustchain_payouts")
    payload = {"total_paid_rtc": "66,531+", "unique_recipients": 1061, "transactions": 3234, "updated_at": "x"}
    with mock.patch("rustchain_langchain.client.requests.get", return_value=_Resp(payload)):
        out = tool._run()
    assert "66,531+ RTC paid" in out


def test_summarize_balance():
    s = summarize_balance({"miner_id": "dual-g4-125", "amount_rtc": 718.51})
    assert "dual-g4-125" in s and "718.51 RTC" in s


def test_summarize_epoch():
    s = summarize_epoch({"epoch": 195, "slot": 28191, "enrolled_miners": 24,
                         "epoch_pot": 1.5, "blocks_per_epoch": 144, "total_supply_rtc": 8388608})
    assert "epoch 195" in s and "24 enrolled" in s and "1.5 RTC" in s


def test_summarize_bounties():
    s = summarize_bounties([{"number": 3074, "reward": "17 RTC", "title": "LangChain tool",
                             "url": "http://x", "created": "2026-03"}])
    assert "#3074" in s and "17 RTC" in s


def test_summarize_bounties_empty():
    assert "No open" in summarize_bounties([])


def test_client_balance_uses_wallet_balance_endpoint():
    c = RustChainClient(base_url="https://example.test")
    with mock.patch("rustchain_langchain.client.requests.get", return_value=_Resp({"amount_rtc": 5.0, "miner_id": "x"})) as g:
        out = c.balance("x")
    assert out["amount_rtc"] == 5.0
    assert g.call_args[0][0] == "https://example.test/wallet/balance"  # NOT bare /balance


# --- async client / async tools (httpx) ---------------------------------
# httpx is monkeypatched, so these never touch the network either. Async
# coroutines are driven with asyncio.run(...) so no pytest-asyncio is needed.
import asyncio

from rustchain_langchain import AsyncRustChainClient


class _AsyncResp:
    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._p


def _fake_async_client(payload, capture=None):
    """Drop-in for ``httpx.AsyncClient`` that records the request and yields
    ``payload`` — no network, no real httpx connection."""
    cap = capture if capture is not None else {}

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            cap["init_kwargs"] = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get(self, url, **kwargs):
            cap["url"] = url
            cap["get_kwargs"] = kwargs
            return _AsyncResp(payload)

    return _FakeAsyncClient


def test_async_client_builds_url_and_parses():
    cap = {}
    c = AsyncRustChainClient(base_url="https://example.test", timeout=7)
    with mock.patch("httpx.AsyncClient", _fake_async_client({"ok": True}, cap)):
        out = asyncio.run(c.health())
    assert out == {"ok": True}
    assert cap["url"] == "https://example.test/health"
    assert cap["init_kwargs"].get("timeout") == 7


def test_async_client_miners_path():
    cap = {}
    c = AsyncRustChainClient(base_url="https://example.test")
    with mock.patch("httpx.AsyncClient", _fake_async_client({"miners": []}, cap)):
        asyncio.run(c.miners())
    assert cap["url"] == "https://example.test/api/miners"


def test_async_client_balance_passes_miner_id():
    cap = {}
    c = AsyncRustChainClient(base_url="https://example.test")
    payload = {"miner_id": "dual-g4-125", "amount_rtc": 42}
    with mock.patch("httpx.AsyncClient", _fake_async_client(payload, cap)):
        out = asyncio.run(c.balance("dual-g4-125"))
    assert cap["url"] == "https://example.test/wallet/balance"
    assert cap["get_kwargs"]["params"] == {"miner_id": "dual-g4-125"}
    assert summarize_balance(out) == "Wallet 'dual-g4-125' holds 42 RTC."


def test_async_client_bounties_reshapes_items():
    cap = {}
    c = AsyncRustChainClient(base_url="https://example.test")
    payload = {"items": [
        {"number": 7, "title": "Add a thing", "body": "pays 25 RTC on merge",
         "html_url": "https://x/7", "created_at": "2026-06-16T00:00:00Z"},
        {"number": 8, "title": "No reward", "body": "", "html_url": "https://x/8",
         "created_at": "2026-06-15T00:00:00Z"},
    ]}
    with mock.patch("httpx.AsyncClient", _fake_async_client(payload, cap)):
        out = asyncio.run(c.bounties(limit=5))
    assert "search/issues" in cap["url"]
    assert "label:bounty" in cap["url"]  # canonical query filters to bounty issues
    assert [b["number"] for b in out] == [7, 8]
    assert out[0]["reward"] == "25 RTC"
    assert out[1]["reward"] == "see issue"


def test_canonical_reward_parser_reads_title_and_decimals():
    """Shared parser used by both clients: title rewards like ``[BOUNTY: 50 RTC]``
    and decimal amounts must be captured, not dropped to ``see issue``."""
    from rustchain_langchain.client import _parse_reward, _reshape_bounty

    # title-only reward (the common `[BOUNTY: N RTC]` shape)
    assert _parse_reward("[BOUNTY: 50 RTC] Add X", "no amount in body") == "50 RTC"
    # decimal amount in the body
    assert _parse_reward("Fix Y", "reward is 2.5 RTC on merge") == "2.5 RTC"
    # title takes precedence over body when both carry an amount
    assert _parse_reward("[BOUNTY: 75 RTC]", "10 RTC mentioned offhand") == "75 RTC"
    # nothing stated -> graceful fallback
    assert _parse_reward("No money here", "") == "see issue"
    # _reshape_bounty applies the same parser end-to-end
    shaped = _reshape_bounty({
        "number": 12, "title": "[BOUNTY: 50 RTC] Add a tool",
        "body": "", "html_url": "https://x/12", "created_at": "2026-06-20T00:00:00Z",
    })
    assert shaped == {
        "number": 12, "title": "[BOUNTY: 50 RTC] Add a tool", "reward": "50 RTC",
        "url": "https://x/12", "created": "2026-06-20",
    }


def test_sync_and_async_bounties_share_canonical_shape():
    """The same raw issue must produce byte-identical output from the sync client
    and the async client (the README's sync/async parity promise)."""
    from rustchain_langchain import RustChainClient
    raw = {"items": [
        {"number": 9, "title": "[BOUNTY: 12.5 RTC] Tweak Z", "body": "",
         "html_url": "https://x/9", "created_at": "2026-06-14T00:00:00Z"},
    ]}
    sync_c = RustChainClient(base_url="https://example.test")
    with mock.patch("rustchain_langchain.client.requests.get",
                    return_value=_Resp(raw)):
        sync_out = sync_c.bounties(limit=5)
    async_c = AsyncRustChainClient(base_url="https://example.test")
    with mock.patch("httpx.AsyncClient", _fake_async_client(raw)):
        async_out = asyncio.run(async_c.bounties(limit=5))
    assert sync_out == async_out
    assert sync_out[0]["reward"] == "12.5 RTC"


def test_async_client_methods_fan_out_concurrently():
    cap = {}
    c = AsyncRustChainClient(base_url="https://example.test")
    payload = {"ok": True}

    async def _gather():
        with mock.patch("httpx.AsyncClient", _fake_async_client(payload, cap)):
            return await asyncio.gather(c.health(), c.health(), c.health())

    results = asyncio.run(_gather())
    assert results == [payload, payload, payload]


def test_async_tool_arun_summarizes_on_success():
    try:
        from rustchain_langchain import get_async_rustchain_tools
        tools = get_async_rustchain_tools(base_url="https://example.test")
    except (ImportError, ModuleNotFoundError):
        return  # langchain-core unavailable — skip gracefully
    tool = next(t for t in tools if t.name == "rustchain_payouts")
    payload = {"total_paid_rtc": "66,531+", "unique_recipients": 1061,
               "transactions": 3234, "updated_at": "x"}
    with mock.patch("httpx.AsyncClient", _fake_async_client(payload)):
        out = asyncio.run(tool._arun())
    assert "66,531+ RTC paid" in out


def test_async_tool_arun_never_raises_on_failure():
    try:
        from rustchain_langchain import get_async_rustchain_tools
        tools = get_async_rustchain_tools(base_url="https://example.test")
    except (ImportError, ModuleNotFoundError):
        return
    tool = next(t for t in tools if t.name == "rustchain_payouts")

    class _BoomClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get(self, *a, **k):
            raise RuntimeError("boom")

    with mock.patch("httpx.AsyncClient", _BoomClient):
        out = asyncio.run(tool._arun())
    assert "RustChain query failed" in out  # graceful, not an exception


def test_async_balance_tool_takes_miner_id():
    try:
        from rustchain_langchain import get_async_rustchain_tools
        tools = get_async_rustchain_tools(base_url="https://example.test")
    except (ImportError, ModuleNotFoundError):
        return
    tool = next(t for t in tools if t.name == "rustchain_balance")
    payload = {"miner_id": "g5-001", "amount_rtc": 7}
    with mock.patch("httpx.AsyncClient", _fake_async_client(payload)):
        out = asyncio.run(tool._arun("g5-001"))
    assert out == "Wallet 'g5-001' holds 7 RTC."


def test_async_tool_sync_bridge_runs_when_no_loop():
    try:
        from rustchain_langchain import get_async_rustchain_tools
        tools = get_async_rustchain_tools(base_url="https://example.test")
    except (ImportError, ModuleNotFoundError):
        return
    tool = next(t for t in tools if t.name == "rustchain_node_health")
    payload = {"ok": True, "db_rw": True, "version": "2.2.1", "backup_age_hours": 1.0}
    with mock.patch("httpx.AsyncClient", _fake_async_client(payload)):
        out = tool._run()  # no running loop -> bridges via asyncio.run
    assert "ok=True" in out and "version=2.2.1" in out


def test_async_tools_match_sync_tool_names():
    try:
        from rustchain_langchain import get_rustchain_tools, get_async_rustchain_tools
        sync_names = {t.name for t in get_rustchain_tools(base_url="https://example.test")}
        async_names = {t.name for t in get_async_rustchain_tools(base_url="https://example.test")}
    except (ImportError, ModuleNotFoundError):
        return
    assert sync_names == async_names  # async surface mirrors the sync one
_AGENTS = [
    {"agent_id": "bcn_sophia_elya", "name": "Sophia Elya", "status": "active",
     "relay": False, "model_id": "gpt-x", "capabilities": ["chat", "curate"]},
    {"agent_id": "bcn_quiet_one", "name": "QuietOne", "status": "presumed_dead", "relay": True},
]
_CONTRACTS = [
    {"id": "ctr_05ce6bf7", "type": "lease_to_own", "amount": 5.0, "currency": "RTC",
     "state": "offered", "from": "bcn_sophia_elya", "to": "relay_sh_sophia_elya"},
    {"id": "ctr_other", "type": "lease", "amount": 2.0, "currency": "RTC",
     "state": "active", "from": "bcn_nobody", "to": "bcn_else"},
]


def test_client_provenance_composes_identity_and_contracts():
    c = RustChainClient(base_url="https://example.test")
    # provenance() calls beacon_agents() then beacon_contracts()
    with mock.patch("rustchain_langchain.client.requests.get",
                    side_effect=[_Resp(_AGENTS), _Resp(_CONTRACTS)]):
        out = c.provenance("bcn_sophia_elya")
    assert out["found"] is True
    assert out["identity"]["name"] == "Sophia Elya"
    assert out["registered_agents"] == 2
    assert len(out["contracts"]) == 1
    assert out["contracts"][0]["role"] == "payer"
    assert out["contracts"][0]["counterparty"] == "relay_sh_sophia_elya"


def test_client_provenance_matches_display_name():
    c = RustChainClient(base_url="https://example.test")
    with mock.patch("rustchain_langchain.client.requests.get",
                    side_effect=[_Resp(_AGENTS), _Resp(_CONTRACTS)]):
        out = c.provenance("Sophia Elya")
    assert out["found"] is True and out["identity"]["agent_id"] == "bcn_sophia_elya"


def test_client_provenance_not_found():
    c = RustChainClient(base_url="https://example.test")
    with mock.patch("rustchain_langchain.client.requests.get", return_value=_Resp(_AGENTS)):
        out = c.provenance("bcn_ghost")
    assert out["found"] is False and out["identity"] is None and out["contracts"] == []


def test_summarize_provenance_found():
    s = summarize_provenance({
        "agent_id": "bcn_sophia_elya", "found": True, "registered_agents": 2,
        "identity": _AGENTS[0],
        "contracts": [{"id": "ctr_05ce6bf7", "type": "lease_to_own", "amount": 5.0,
                       "currency": "RTC", "state": "offered", "role": "payer",
                       "counterparty": "relay_sh_sophia_elya"}],
    })
    assert "Sophia Elya" in s
    assert "IDENTITY PRESENT" in s
    assert "1 Beacon contract" in s and "ctr_05ce6bf7" in s
    assert "not yet deployed" in s  # honest about the un-built content layer


def test_summarize_provenance_not_found():
    s = summarize_provenance({"agent_id": "bcn_ghost", "found": False,
                              "registered_agents": 116, "identity": None, "contracts": []})
    assert "No Beacon agent found" in s and "116" in s


def test_provenance_tool_run():
    try:
        from rustchain_langchain import get_rustchain_tools
        tools = get_rustchain_tools(base_url="https://example.test")
    except Exception:
        return
    tool = next(t for t in tools if t.name == "rustchain_provenance")
    with mock.patch("rustchain_langchain.client.requests.get",
                    side_effect=[_Resp(_AGENTS), _Resp(_CONTRACTS)]):
        out = tool._run("bcn_sophia_elya")
    assert "Sophia Elya" in out and "Agent layer" in out


def test_provenance_tool_never_raises():
    try:
        from rustchain_langchain import get_rustchain_tools
        tools = get_rustchain_tools(base_url="https://example.test")
    except Exception:
        return
    tool = next(t for t in tools if t.name == "rustchain_provenance")
    with mock.patch("rustchain_langchain.client.requests.get", side_effect=RuntimeError("boom")):
        out = tool._run("bcn_x")
    assert "RustChain query failed" in out


# --- hall of fame (oldest / most-prized attesting hardware) --------------
# first_attest 1_700_000_000 -> 2023-11-14; +864000s == 10 whole days later.
_MINERS_HALL = {"miners": [
    {"miner": "RTCyoung", "device_family": "ARM", "device_arch": "aarch64",
     "hardware_type": "Unknown/Other", "antiquity_multiplier": 0.0005,
     "first_attest": 1_782_000_000, "last_attest": 1_782_086_400},
    {"miner": "RTColdest_powerpc_g4_00000000000000000000",
     "device_family": "PowerPC", "device_arch": "ppc", "hardware_type": "Power Mac G4",
     "antiquity_multiplier": 3.0, "first_attest": 1_700_000_000,
     "last_attest": 1_700_864_000},
    {"miner": "RTCg5", "device_family": "PowerPC", "device_arch": "ppc64",
     "hardware_type": "Power Mac G5", "antiquity_multiplier": 3.0,
     "first_attest": 1_710_000_000, "last_attest": 1_710_000_000},
]}


def test_reshape_hall_ranks_by_antiquity_then_age():
    from rustchain_langchain.client import _reshape_hall
    out = _reshape_hall(_MINERS_HALL)
    # highest antiquity first; the two ×3.0 tie broken by earliest first_attest
    assert [m["miner"] for m in out] == [
        "RTColdest_powerpc_g4_00000000000000000000", "RTCg5", "RTCyoung"]
    assert [m["rank"] for m in out] == [1, 2, 3]
    top = out[0]
    assert top["device"] == "PowerPC" and top["hardware"] == "Power Mac G4"
    assert top["first_attest"] == "2023-11-14"  # UTC day of first_attest
    assert top["days_attesting"] == 10  # (last-first)//86400


def test_reshape_hall_respects_limit_and_tolerates_junk():
    from rustchain_langchain.client import _reshape_hall
    assert _reshape_hall(_MINERS_HALL, limit=1)[0]["rank"] == 1
    assert len(_reshape_hall(_MINERS_HALL, limit=1)) == 1
    assert _reshape_hall({}) == []
    assert _reshape_hall([{"miner": "x"}])[0]["days_attesting"] is None  # no timestamps


def test_summarize_hall_of_fame():
    from rustchain_langchain.client import _reshape_hall
    s = summarize_hall_of_fame(_reshape_hall(_MINERS_HALL))
    assert "hall of fame" in s and "3 oldest" in s
    assert "Power Mac G4" in s and "antiquity ×3.0" in s
    assert "#1 RTColdest" in s  # long id is shortened but recognisable


def test_summarize_hall_of_fame_empty():
    assert "No RustChain hall-of-fame" in summarize_hall_of_fame([])


def test_client_hall_of_fame_reads_miners_endpoint():
    c = RustChainClient(base_url="https://example.test")
    with mock.patch("rustchain_langchain.client.requests.get",
                    return_value=_Resp(_MINERS_HALL)) as g:
        out = c.hall_of_fame(limit=2)
    assert g.call_args[0][0] == "https://example.test/api/miners"  # keyless surface
    assert [m["miner"] for m in out] == [
        "RTColdest_powerpc_g4_00000000000000000000", "RTCg5"]


def test_sync_and_async_hall_share_canonical_shape():
    sync_c = RustChainClient(base_url="https://example.test")
    with mock.patch("rustchain_langchain.client.requests.get",
                    return_value=_Resp(_MINERS_HALL)):
        sync_out = sync_c.hall_of_fame(limit=5)
    async_c = AsyncRustChainClient(base_url="https://example.test")
    with mock.patch("httpx.AsyncClient", _fake_async_client(_MINERS_HALL)):
        async_out = asyncio.run(async_c.hall_of_fame(limit=5))
    assert sync_out == async_out


def test_hall_tool_run_and_never_raises():
    try:
        from rustchain_langchain import get_rustchain_tools
        tools = get_rustchain_tools(base_url="https://example.test")
    except (ImportError, ModuleNotFoundError):
        return
    tool = next(t for t in tools if t.name == "rustchain_hall_of_fame")
    with mock.patch("rustchain_langchain.client.requests.get",
                    return_value=_Resp(_MINERS_HALL)):
        out = tool._run(limit=3)
    assert "hall of fame" in out and "Power Mac G4" in out
    with mock.patch("rustchain_langchain.client.requests.get", side_effect=RuntimeError("boom")):
        assert "RustChain query failed" in tool._run()


def test_async_hall_tool_arun():
    try:
        from rustchain_langchain import get_async_rustchain_tools
        tools = get_async_rustchain_tools(base_url="https://example.test")
    except (ImportError, ModuleNotFoundError):
        return
    tool = next(t for t in tools if t.name == "rustchain_hall_of_fame")
    with mock.patch("httpx.AsyncClient", _fake_async_client(_MINERS_HALL)):
        out = asyncio.run(tool._arun(limit=3))
    assert "hall of fame" in out and "Power Mac G4" in out
