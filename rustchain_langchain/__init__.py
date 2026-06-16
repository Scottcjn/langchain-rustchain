# SPDX-License-Identifier: MIT
"""langchain-rustchain — read-only LangChain tools for the RustChain agent economy.

The balance / bounties / epoch tools were contributed by @hektorhq
(SiliconBountyHunter) for bounty #3074; merged here with the balance endpoint
corrected to /wallet/balance and a tested, never-raise wrapper.
"""
from .client import RustChainClient
from .async_client import AsyncRustChainClient
from .tools import (
    get_rustchain_tools,
    get_async_rustchain_tools,
    summarize_network,
    summarize_payouts,
    summarize_miners,
    summarize_health,
    summarize_balance,
    summarize_epoch,
    summarize_bounties,
)

__version__ = "0.2.0"
__all__ = [
    "RustChainClient",
    "AsyncRustChainClient",
    "get_rustchain_tools",
    "get_async_rustchain_tools",
    "summarize_network",
    "summarize_payouts",
    "summarize_miners",
    "summarize_health",
    "summarize_balance",
    "summarize_epoch",
    "summarize_bounties",
]
