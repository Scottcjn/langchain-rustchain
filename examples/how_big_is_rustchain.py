# SPDX-License-Identifier: MIT
"""Ask a small ReAct-style agent: "how big is RustChain right now?"

This example is read-only and keyless. By default it calls the live RustChain
public endpoints through the package's LangChain tools. Use ``--offline`` for a
deterministic fixture-backed run that is useful in CI or on airplanes.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Protocol

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rustchain_langchain.tools import (
    get_rustchain_tools,
    summarize_epoch,
    summarize_health,
    summarize_miners,
    summarize_network,
    summarize_payouts,
)

QUESTION = "how big is RustChain right now?"
PLAN = (
    "rustchain_node_health",
    "rustchain_epoch",
    "rustchain_miners",
    "rustchain_payouts",
    "rustchain_network_stats",
)


class RunnableTool(Protocol):
    name: str
    description: str

    def invoke(self, input: object | None = None) -> str:
        ...


@dataclass(frozen=True)
class ReActStep:
    thought: str
    action: str
    observation: str


class FixtureTool:
    """Tiny tool adapter used only for the offline demo and tests."""

    def __init__(self, name: str, description: str, observation: str) -> None:
        self.name = name
        self.description = description
        self._observation = observation

    def invoke(self, input: object | None = None) -> str:
        return self._observation


def _run_tool(tool: object) -> str:
    if hasattr(tool, "invoke"):
        return str(tool.invoke({}))
    if hasattr(tool, "_run"):
        return str(tool._run())
    raise TypeError(f"Tool {tool!r} is not runnable")


def run_react_trace(
    question: str,
    tools: Iterable[RunnableTool],
    plan: Iterable[str] = PLAN,
) -> tuple[list[ReActStep], str]:
    """Run a deterministic ReAct trace over the supplied RustChain tools."""
    tool_map: Mapping[str, RunnableTool] = {tool.name: tool for tool in tools}
    steps: list[ReActStep] = []
    for name in plan:
        tool = tool_map[name]
        thought = _thought_for(name)
        observation = _run_tool(tool)
        steps.append(ReActStep(thought=thought, action=name, observation=observation))
    return steps, synthesize_answer(question, steps)


def synthesize_answer(question: str, steps: Iterable[ReActStep]) -> str:
    observations = {step.action: step.observation for step in steps}
    failures = [
        observation
        for observation in observations.values()
        if observation.startswith("RustChain query failed")
    ]
    if len(failures) == len(observations):
        status = "The agent ran, but every live RustChain read failed in this run."
    elif failures:
        status = "RustChain is measurable, but this run had partial live-read failures."
    else:
        status = "RustChain is live and measurable from its public read-only APIs."
    return "\n".join(
        [
            f"Question: {question}",
            "Final Answer:",
            status,
            observations.get("rustchain_epoch", ""),
            observations.get("rustchain_miners", ""),
            observations.get("rustchain_payouts", ""),
            observations.get("rustchain_network_stats", ""),
        ]
    )


def build_offline_tools() -> list[FixtureTool]:
    return [
        FixtureTool(
            "rustchain_node_health",
            "Fixture node health.",
            summarize_health(
                {"ok": True, "db_rw": True, "version": "demo", "backup_age_hours": 1.2}
            ),
        ),
        FixtureTool(
            "rustchain_epoch",
            "Fixture epoch details.",
            summarize_epoch(
                {
                    "epoch": 277,
                    "slot": 39888,
                    "enrolled_miners": 24,
                    "epoch_pot": 1.5,
                    "blocks_per_epoch": 144,
                    "total_supply_rtc": 8388608,
                }
            ),
        ),
        FixtureTool(
            "rustchain_miners",
            "Fixture attesting miners.",
            summarize_miners(
                {
                    "miners": [
                        {"device_arch": "PowerPC G4"},
                        {"device_arch": "PowerPC G4"},
                        {"device_arch": "POWER8"},
                        {"device_arch": "x86_64"},
                    ]
                }
            ),
        ),
        FixtureTool(
            "rustchain_payouts",
            "Fixture payout totals.",
            summarize_payouts(
                {
                    "total_paid_rtc": "66,531+",
                    "unique_recipients": 1061,
                    "transactions": 3234,
                    "updated_at": "2026-06-16",
                }
            ),
        ),
        FixtureTool(
            "rustchain_network_stats",
            "Fixture network activity.",
            summarize_network(
                {
                    "as_of": "2026-06-16",
                    "facts": [
                        {
                            "id": "onchain_activity",
                            "value": {
                                "wallet_transfers": 3112,
                                "rtc_moved_in_transfers": 124848.19,
                                "distinct_wallets": 1354,
                                "ledger_entries_total": 7584,
                            },
                        }
                    ],
                }
            ),
        ),
    ]


def _thought_for(tool_name: str) -> str:
    thoughts = {
        "rustchain_node_health": "First confirm the node is healthy enough to trust.",
        "rustchain_epoch": "Measure the current consensus/mining round.",
        "rustchain_miners": "Count the attesting hardware behind the network.",
        "rustchain_payouts": "Check the size of the contributor/miner economy.",
        "rustchain_network_stats": "Finish with on-chain wallet and ledger activity.",
    }
    return thoughts[tool_name]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="https://rustchain.org")
    parser.add_argument("--timeout", type=int, default=15)
    parser.add_argument("--insecure", action="store_true", help="Disable TLS verification for dev nodes.")
    parser.add_argument("--offline", action="store_true", help="Use fixture-backed tools instead of live HTTP.")
    args = parser.parse_args()

    tools = (
        build_offline_tools()
        if args.offline
        else get_rustchain_tools(
            base_url=args.base_url,
            timeout=args.timeout,
            verify=not args.insecure,
        )
    )
    steps, answer = run_react_trace(QUESTION, tools)
    for i, step in enumerate(steps, start=1):
        print(f"Thought {i}: {step.thought}")
        print(f"Action {i}: {step.action}")
        print(f"Observation {i}: {step.observation}\n")
    print(answer)


if __name__ == "__main__":
    main()
