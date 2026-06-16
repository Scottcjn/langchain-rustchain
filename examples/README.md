# Examples

## ReAct demo

Run a read-only/keyless demo agent that answers "how big is RustChain right now?"
by calling the package's RustChain tools:

```bash
pip install -e ".[langchain]"
python examples/how_big_is_rustchain.py
```

For a deterministic run without network access:

```bash
python examples/how_big_is_rustchain.py --offline
```

The script prints a compact Thought / Action / Observation trace and then a
final answer synthesized from node health, epoch, miner, payout, and on-chain
activity tools. No wallet signing, private keys, or write endpoints are used.
