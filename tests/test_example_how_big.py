# SPDX-License-Identifier: MIT
"""Tests for the examples/how_big_is_rustchain.py demo. No network, no LLM."""
import importlib.util
import os
from unittest import mock

from rustchain_langchain import RustChainClient

# Load the example module by path (examples/ is not an importable package).
_EXAMPLE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "examples", "how_big_is_rustchain.py"
)
_spec = importlib.util.spec_from_file_location("how_big_is_rustchain", _EXAMPLE_PATH)
how_big = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(how_big)


_NETWORK = {
    "as_of": "2026-06-16",
    "facts": [
        {"id": "onchain_activity", "value": {
            "wallet_transfers": 3112, "rtc_moved_in_transfers": 124848.19,
            "distinct_wallets": 1354, "ledger_entries_total": 7584}},
    ],
}
_EPOCH = {
    "epoch": 42, "slot": 7, "enrolled_miners": 116, "epoch_pot": 500,
    "blocks_per_epoch": 100, "total_supply_rtc": 1000000,
}
_MINERS = {"miners": [
    {"device_arch": "ppc-g4"}, {"device_arch": "ppc-g4"}, {"device_arch": "power8"},
]}
_PAYOUTS = {"total_paid_rtc": "66,531+", "unique_recipients": 1061,
            "transactions": 3234, "updated_at": "2026-06-16"}


def _mocked_client():
    c = RustChainClient(base_url="https://example.test")
    c.network_stats = mock.Mock(return_value=_NETWORK)
    c.epoch = mock.Mock(return_value=_EPOCH)
    c.miners = mock.Mock(return_value=_MINERS)
    c.payouts = mock.Mock(return_value=_PAYOUTS)
    return c


def test_answer_how_big_composes_all_sections():
    out = how_big.answer_how_big(_mocked_client())
    assert out.startswith("How big is RustChain right now?")
    # one fact from each of the four tools is present
    assert "3112 wallet transfers" in out      # network
    assert "epoch 42" in out                    # epoch
    assert "3 attesting miner(s)" in out        # miners
    assert "66,531+ RTC paid" in out            # payouts


def test_answer_how_big_never_raises_on_bad_endpoint():
    c = _mocked_client()
    c.epoch = mock.Mock(side_effect=RuntimeError("boom"))
    out = how_big.answer_how_big(c)
    # the failing section is reported, the others still render
    assert "Current epoch: unavailable (RuntimeError: boom)" in out
    assert "3112 wallet transfers" in out
    assert "66,531+ RTC paid" in out


def test_main_offline_prints_answer(capsys):
    c = _mocked_client()
    with mock.patch.object(how_big, "RustChainClient", return_value=c):
        rc = how_big.main(["--offline"])
    assert rc == 0
    assert "How big is RustChain right now?" in capsys.readouterr().out


def test_main_without_key_falls_back_to_offline(capsys):
    c = _mocked_client()
    with mock.patch.dict(os.environ, {}, clear=True), \
            mock.patch.object(how_big, "RustChainClient", return_value=c):
        rc = how_big.main([])
    assert rc == 0
    assert "How big is RustChain right now?" in capsys.readouterr().out
