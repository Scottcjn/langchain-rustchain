# SPDX-License-Identifier: MIT
"""Tests for runnable examples. No live network required."""

from examples.how_big_is_rustchain import (
    QUESTION,
    build_offline_tools,
    run_react_trace,
)


def test_offline_react_demo_answers_how_big_question():
    steps, answer = run_react_trace(QUESTION, build_offline_tools())

    assert [step.action for step in steps] == [
        "rustchain_node_health",
        "rustchain_epoch",
        "rustchain_miners",
        "rustchain_payouts",
        "rustchain_network_stats",
    ]
    assert "RustChain is live" in answer
    assert "epoch 277" in answer
    assert "4 attesting miner" in answer
    assert "66,531+ RTC paid" in answer
    assert "3112 wallet transfers" in answer
