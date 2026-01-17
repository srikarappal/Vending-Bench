# Engram-Vend2: Vending-Bench 2 Implementation

A faithful reproduction of Andon Labs' Vending-Bench 2 benchmark for testing AI agents on long-running business tasks.

## Overview

This benchmark simulates a vending machine business over a configurable time period (1-365 days). The agent must:
- Manage inventory and pricing
- Handle supplier relationships
- Respond to competition and events
- Maintain profitability

**Key Innovation:** Tests whether Engram's memory architecture maintains coherence over extended simulations where pure LLMs experience "meltdown loops."

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run baseline (pure LLM, no Engram)
python experiments/run_experiment.py --mode baseline --days 3

# Run with Engram
python experiments/run_experiment.py --mode engram --days 3

# Run full 365-day simulation
python experiments/run_experiment.py --mode engram --days 365 --events full
```

## Architecture

```
Customer LLM (Claude Sonnet 4.5)
    ↓ analyzes events
    ↓ makes decisions
memLLM-R (Claude Sonnet 4.5)
    ↓ retrieves relevant memories
    ↓ ingests outcomes
Storage (ChromaDB + SQLite + Neo4j)
```

## Configuration

Edit `config/simulation_config.py`:

```python
simulation_days = 30        # 1, 3, 30, 365
event_complexity = "simple" # simple, medium, full
starting_cash = 500
daily_fee = 2
max_messages = 2000
```

## Experiments

**Baseline:** Pure Claude Sonnet 4.5 with only context window
**Engram:** Claude Sonnet 4.5 + memLLM-R for persistent memory

### Metrics
- Net worth (primary)
- Days to profitability
- Coherence score (detects meltdown loops)
- Inventory efficiency
- Decision consistency

## Project Structure

```
engram-vend2/
├── config/              # Simulation parameters
├── src/                 # Core simulation engine
├── agents/              # Agent implementations
├── tasks/               # inspect_ai tasks
├── experiments/         # Experiment runners
├── tests/               # Unit tests
└── storage/             # Database storage
```

## References

- [Vending-Bench 2 Paper](https://arxiv.org/abs/2502.15840)
- [Andon Labs Website](https://andonlabs.com/evals/vending-bench-2)
- [inspect_ai Framework](https://github.com/UKGovernmentBEIS/inspect_ai)
