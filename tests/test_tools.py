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
    summarize_epoch,
    summarize_hall_of_fame,
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


def test_summarize_epoch():
    s = summarize_epoch({
        "blocks_per_epoch": 144, "enrolled_miners": 24, "epoch": 195,
        "epoch_pot": 1.5, "slot": 28190, "total_supply_rtc": 8388608,
    })
    assert "epoch 195" in s
    assert "slot 28190" in s
    assert "24 miner(s) enrolled" in s
    assert "8388608 RTC" in s


def test_summarize_hall_of_fame_ranks_machines():
    data = {"leaderboard": [
        {"rank": 1, "device_model": "PowerPC 7450 (G4) @ 733MHz", "manufacture_year": 2001,
         "rust_score": 1178.45, "total_attestations": 868457, "badge": "Oxidized Legend"},
        {"rank": 2, "device_model": "Mac", "rust_score": 900.0, "total_attestations": 5000},
    ]}
    s = summarize_hall_of_fame(data, top=1)
    assert "top 1" in s
    assert "#1 PowerPC 7450 (G4) @ 733MHz, 2001" in s
    assert "Oxidized Legend" in s
    # only the requested top slice is rendered
    assert "#2 Mac" not in s


def test_summarize_hall_of_fame_empty():
    assert "no machines" in summarize_hall_of_fame({"leaderboard": []})
    assert "no machines" in summarize_hall_of_fame("garbage")


def test_summarize_bounties_sorts_and_totals():
    issues = [
        {"number": 1, "title": "[BOUNTY: 5 RTC] Small fix"},
        {"number": 2, "title": "[BOUNTY: 50-200 RTC] Big video"},
        {"number": 3, "title": "[BOUNTY] feverdream client", "body": "pays 10 RTC on merge"},
        {"number": 4, "title": "PR for wallet RTC5f3aabbcc"},  # wallet addr, no reward
    ]
    s = summarize_bounties(issues, top=2)
    assert "4 open RustChain bounties" in s
    # 200 (range upper) + 5 + 10 + 0 = 215
    assert "~215 RTC" in s
    # sorted by reward desc: #2 (200) then #3 (10); top=2 excludes #1/#4
    assert s.index("#2") < s.index("#3")
    assert "#1" not in s and "#4" not in s
    # the wallet-only issue must not be mistaken for a reward
    assert "RTC5f3aabbcc" not in s


def test_summarize_bounties_filters_pull_requests_and_empty():
    assert "no open bounties" in summarize_bounties([])
    only_pr = [{"number": 9, "title": "[BOUNTY: 5 RTC] x", "pull_request": {"url": "y"}}]
    assert "no open bounties" in summarize_bounties(only_pr)


def test_bounties_client_builds_github_url_and_drops_prs():
    c = RustChainClient()
    payload = [
        {"number": 1, "title": "a"},
        {"number": 2, "title": "b", "pull_request": {"url": "x"}},
    ]
    with mock.patch("rustchain_langchain.client.requests.get", return_value=_Resp(payload)) as g:
        out = c.bounties()
    called_url = g.call_args[0][0]
    assert called_url == "https://api.github.com/repos/Scottcjn/rustchain-bounties/issues"
    assert g.call_args.kwargs["params"]["labels"] == "bounty"
    assert g.call_args.kwargs["params"]["state"] == "open"
    assert [i["number"] for i in out] == [1]  # PR filtered out
