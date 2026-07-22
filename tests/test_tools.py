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
    except Exception:
        return
    tool = next(t for t in tools if t.name == "rustchain_payouts")
    with mock.patch("rustchain_langchain.client.requests.get", side_effect=RuntimeError("boom")):
        out = tool._run()
    assert "RustChain query failed" in out  # graceful, not an exception


def test_tool_run_summarizes_on_success():
    try:
        from rustchain_langchain import get_rustchain_tools
        tools = get_rustchain_tools(base_url="https://example.test")
    except Exception:
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


def test_bounties_reward_parse_decimal_and_thousands():
    # Real open bounty bodies carry decimals ("2.5 RTC") and large grouped
    # amounts ("3,333 RTC"). The bare \d+ parser dropped the decimal ("2.5" ->
    # "5") and the leading groups ("3,333" -> "333"). Guard the real amount.
    payload = {"items": [
        {"number": 1, "title": "small", "body": "**Rate:** 2.5 RTC (micro)",
         "html_url": "u1", "created_at": "2026-07-22T00:00:00Z"},
        {"number": 2, "title": "big", "body": "Founding 100 miners — 3,333 RTC pool",
         "html_url": "u2", "created_at": "2026-07-22T00:00:00Z"},
        {"number": 3, "title": "plain", "body": "Rate: 12 RTC",
         "html_url": "u3", "created_at": "2026-07-22T00:00:00Z"},
        {"number": 4, "title": "none", "body": "no amount here",
         "html_url": "u4", "created_at": "2026-07-22T00:00:00Z"},
    ]}
    c = RustChainClient(base_url="https://example.test")
    with mock.patch("rustchain_langchain.client.requests.get", return_value=_Resp(payload)):
        out = c.bounties(limit=10)
    rewards = {b["number"]: b["reward"] for b in out}
    assert rewards[1] == "2.5 RTC"   # was "5 RTC"
    assert rewards[2] == "3333 RTC"  # was "333 RTC"
    assert rewards[3] == "12 RTC"    # unchanged
    assert rewards[4] == "see issue"
