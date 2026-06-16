# SPDX-License-Identifier: MIT
"""Demo: a LangChain agent answering "How big is RustChain right now?"

This is the canonical end-to-end example for ``langchain-rustchain``: it wires
the read-only RustChain tools into a ReAct agent and asks it a single, concrete
question about the live network. The agent decides which tools to call
(network activity, current epoch, attesting miners, payouts) and writes a short
answer from what it reads.

Two ways to run it
------------------

1. **With an LLM (real ReAct loop).** Set an API key and run::

       export OPENAI_API_KEY=sk-...
       python examples/how_big_is_rustchain.py

   The agent reasons about which RustChain tools to call and summarises the
   answer itself. Any chat model that supports tool/function calling works —
   pass ``--model`` to pick one (default ``gpt-4o-mini``).

2. **Offline (no key, no LLM).** Run::

       python examples/how_big_is_rustchain.py --offline

   This skips the LLM and calls the same read tools directly, composing the
   "how big" answer deterministically. It still hits the live, public,
   read-only RustChain endpoints (or any ``--base-url`` you point it at), so
   you can see real output without paying for an LLM.

If no LLM API key is found, the script automatically falls back to ``--offline``
so it always produces output.

Everything here is read-only and keyless — the same public surfaces a human can
hit in a browser. No wallet writes, no secrets.
"""
from __future__ import annotations

import argparse
import os
import sys

# Make the example runnable from a source checkout without installing first.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rustchain_langchain import (  # noqa: E402
    RustChainClient,
    summarize_epoch,
    summarize_miners,
    summarize_network,
    summarize_payouts,
)

QUESTION = "How big is RustChain right now?"


# --- offline path: compose the answer directly from the read tools ----------
def answer_how_big(client: RustChainClient) -> str:
    """Answer "how big is RustChain right now?" by reading live state.

    Pure and framework-free (no LangChain, no LLM): it queries the same
    read-only surfaces the agent tools wrap and stitches their summaries into a
    short report. Each lookup is independent — if one endpoint is unavailable we
    still report the rest rather than failing the whole answer.
    """
    sections: list[str] = []
    for label, fetch, summarize in (
        ("Network activity", client.network_stats, summarize_network),
        ("Current epoch", client.epoch, summarize_epoch),
        ("Attesting miners", client.miners, summarize_miners),
        ("Payouts to date", client.payouts, summarize_payouts),
    ):
        try:
            sections.append(summarize(fetch()))
        except Exception as e:  # never let one bad endpoint sink the report
            sections.append(f"{label}: unavailable ({type(e).__name__}: {e}).")
    return "How big is RustChain right now?\n\n" + "\n\n".join(sections)


# --- LLM path: a real ReAct agent that chooses which tools to call ----------
def build_react_agent(llm, base_url: str = "https://rustchain.org"):
    """Build a ReAct agent wired with the RustChain tools.

    Uses ``langgraph.prebuilt.create_react_agent`` (the modern LangChain ReAct
    runtime). ``langgraph`` and ``langchain-core`` are required for this path.
    """
    from langgraph.prebuilt import create_react_agent  # lazy import

    from rustchain_langchain import get_rustchain_tools

    tools = get_rustchain_tools(base_url=base_url)
    return create_react_agent(llm, tools)


def _make_llm(model: str):
    """Construct a tool-calling chat model from the environment, or return None.

    Kept tiny and provider-agnostic: we use OpenAI if ``OPENAI_API_KEY`` is set
    because it is the most common, but any ``langchain`` chat model with tool
    calling can be dropped in here.
    """
    if not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        print(
            "OPENAI_API_KEY is set but `langchain-openai` is not installed.\n"
            "Install it (`pip install langchain-openai langgraph`) or run with "
            "--offline.",
            file=sys.stderr,
        )
        return None
    return ChatOpenAI(model=model, temperature=0)


def run_with_agent(llm, base_url: str) -> str:
    agent = build_react_agent(llm, base_url=base_url)
    result = agent.invoke({"messages": [("user", QUESTION)]})
    return result["messages"][-1].content


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip the LLM and compose the answer directly from the read tools.",
    )
    parser.add_argument(
        "--base-url",
        default="https://rustchain.org",
        help="RustChain base URL (default: https://rustchain.org).",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Chat model for the ReAct agent (default: gpt-4o-mini).",
    )
    args = parser.parse_args(argv)

    llm = None if args.offline else _make_llm(args.model)

    if llm is None:
        if not args.offline:
            print("(no LLM available — falling back to --offline)\n", file=sys.stderr)
        client = RustChainClient(base_url=args.base_url)
        print(answer_how_big(client))
        return 0

    print(f"Q: {QUESTION}\n")
    print(run_with_agent(llm, args.base_url))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
