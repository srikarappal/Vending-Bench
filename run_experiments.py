"""
Experiment runner for Vending-Bench 2.

Runs baseline and Engram agents across different simulation configurations.
"""

import argparse
import os
import json
import logging
import warnings
from datetime import datetime

# Suppress Google API key warning (must be done before any imports that trigger it)
class DuplicateFilter(logging.Filter):
    """Filter that suppresses duplicate log messages."""
    def __init__(self):
        super().__init__()
        self.seen = set()

    def filter(self, record):
        msg = record.getMessage()
        if "GOOGLE_API_KEY" in msg and "GEMINI_API_KEY" in msg:
            if msg in self.seen:
                return False
            self.seen.add(msg)
        return True

# Apply filter to root logger and specific loggers
for logger_name in ["", "inspect_ai", "inspect_ai._util", "inspect_ai._util._api_client"]:
    logger = logging.getLogger(logger_name)
    logger.addFilter(DuplicateFilter())

warnings.filterwarnings("ignore", message=".*GOOGLE_API_KEY.*GEMINI_API_KEY.*")

import asyncio
from inspect_ai import eval
from inspect_ai.log import read_eval_log

from tasks.baseline_task import vending_baseline, vending_subagent
from tasks.engram_task import vending_engram


def run_experiment(
    agent_type: str = "engram",
    simulation_days: int = 3,
    starting_cash: float = 500.0,
    event_complexity: str = "simple",
    memory_llm_model: str = "claude-sonnet-4-5",
    customer_llm_model: str = "claude-sonnet-4-5",
    subagent_llm_model: str = None,
    allowed_search_types: list = None,
    log_dir: str = "./experiments/logs",
    debug: bool = False,
    prefix: str = None,
    email_system_enabled: bool = False
):
    """
    Run a single vending machine experiment.

    Args:
        agent_type: "baseline", "subagent", or "engram"
        simulation_days: Number of days to simulate
        starting_cash: Starting cash balance
        event_complexity: Event complexity level
        memory_llm_model: Model for memLLM-R (Engram only)
        customer_llm_model: Model for customer-facing LLM
        subagent_llm_model: Model for sub-agent (subagent mode only)
        allowed_search_types: Search types for retrieval (Engram only)
        log_dir: Directory for experiment logs
        debug: Enable debug mode
        prefix: Optional prefix for log filename identification
        email_system_enabled: Enable VendingBench 2 style email-based supplier negotiation

    Returns:
        Evaluation results
    """
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)

    # Generate log filename with model name and days
    # Sanitize model name (replace / with _)
    model_short = customer_llm_model.replace("/", "_").replace(":", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_suffix = "_email" if email_system_enabled else ""
    if prefix:
        log_name = f"{prefix}_{agent_type}_{model_short}_{simulation_days}d{mode_suffix}_{timestamp}"
    else:
        log_name = f"{agent_type}_{model_short}_{simulation_days}d{mode_suffix}_{timestamp}"

    print(f"\n{'='*70}")
    print(f"Running Vending-Bench 2 Experiment")
    print(f"{'='*70}")
    print(f"Agent Type: {agent_type}")
    print(f"Simulation Days: {simulation_days}")
    print(f"Starting Cash: ${starting_cash:.2f}")
    print(f"Event Complexity: {event_complexity}")
    print(f"Ordering Mode: {'EMAIL (supplier negotiation)' if email_system_enabled else 'DIRECT (fixed prices)'}")
    if agent_type == "engram":
        print(f"Memory LLM: {memory_llm_model}")
        print(f"Customer LLM: {customer_llm_model}")
        print(f"Search Types: {allowed_search_types or ['semantic', 'fulltext', 'graph']}")
    elif agent_type == "subagent":
        print(f"Main Agent LLM: {customer_llm_model}")
        print(f"Sub-Agent LLM: {subagent_llm_model or customer_llm_model}")
    print(f"Log Name: {log_name}")
    print(f"{'='*70}\n")

    # Select task based on agent type
    if agent_type == "baseline":
        task = vending_baseline(
            simulation_days=simulation_days,
            starting_cash=starting_cash,
            event_complexity=event_complexity,
            customer_model=customer_llm_model,
            email_system_enabled=email_system_enabled,
            debug=debug
        )
    elif agent_type == "subagent":
        task = vending_subagent(
            simulation_days=simulation_days,
            starting_cash=starting_cash,
            event_complexity=event_complexity,
            customer_model=customer_llm_model,
            subagent_model=subagent_llm_model
            # Note: email_system_enabled not yet supported for subagent mode
        )
    elif agent_type == "engram":
        task = vending_engram(
            simulation_days=simulation_days,
            starting_cash=starting_cash,
            event_complexity=event_complexity,
            memory_llm_model=memory_llm_model,
            customer_llm_model=customer_llm_model,
            allowed_search_types=allowed_search_types,
            debug=debug
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}. Use 'baseline', 'subagent', or 'engram'")

    # Run evaluation
    results = eval(
        task,
        log_dir=log_dir,
        log_file_name=log_name
    )

    # Print results summary
    print(f"\n{'='*70}")
    print(f"Experiment Complete")
    print(f"{'='*70}")

    # Try to extract results from eval logs
    final_metrics = None

    # Method 1: Read from log file
    log_path = os.path.join(log_dir, f"{log_name}.json")
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            log_data = json.load(f)

        # Extract metrics
        samples = log_data.get("samples", [])
        if samples:
            sample = samples[0]
            sim_results = sample.get("metadata", {}).get("simulation_results", {})
            final_metrics = sim_results.get("final_metrics", {})

    # Method 2: Extract from eval results object
    if final_metrics is None and hasattr(results, 'samples') and results.samples:
        sample = results.samples[0]
        if hasattr(sample, 'metadata') and sample.metadata:
            sim_results = sample.metadata.get("simulation_results", {})
            final_metrics = sim_results.get("final_metrics", {})

    # Print metrics if found
    if final_metrics:
        starting_cash = 500.0  # from config
        final_cash = final_metrics.get('final_net_worth', 0) - (final_metrics.get('final_net_worth', 0) - final_metrics.get('profit_loss', 0) - starting_cash)
        # Simpler: final_cash = starting_cash + profit_loss (minus inventory value locked up)

        print(f"\nFinal Metrics:")
        print(f"  Starting Cash: ${starting_cash:.2f}")
        print(f"  Final Net Worth: ${final_metrics.get('final_net_worth', 0):.2f}")
        print(f"  Profit/Loss: ${final_metrics.get('profit_loss', 0):.2f}")
        print(f"  Total Revenue: ${final_metrics.get('total_revenue', 0):.2f}")
        print(f"  Total Costs: ${final_metrics.get('total_costs', 0):.2f}")
        print(f"  Days Profitable: {final_metrics.get('days_profitable', 0)}/{final_metrics.get('days_simulated', 0)}")
        print(f"  Messages Used: {final_metrics.get('messages_used', 0)}")

        # Memory stats for Engram
        if agent_type == "engram" and 'sim_results' in locals():
            memory_stats = sim_results.get("memory_stats", {})
            print(f"\nMemory Stats:")
            print(f"  Total Ingests: {memory_stats.get('total_ingests', 0)}")
            print(f"  Total Retrievals: {memory_stats.get('total_retrievals', 0)}")
            print(f"  Memories Stored: {memory_stats.get('total_memories_stored', 0)}")
            print(f"  Memories Retrieved: {memory_stats.get('total_memories_retrieved', 0)}")

    # Extract scores (only if log file exists)
    if os.path.exists(log_path):
        if 'log_data' in locals():
            scores = log_data.get("scores", {})
            if scores:
                print(f"\nScores:")
                for score_name, score_data in scores.items():
                    if isinstance(score_data, dict):
                        value = score_data.get("value", 0)
                        print(f"  {score_name}: {value:.4f}")

        print(f"\nLog saved to: {log_path}")
    print(f"{'='*70}\n")

    return results


def run_comparison_suite(
    simulation_days: int = 3,
    event_complexity: str = "simple",
    num_runs: int = 1
):
    """
    Run comparison suite: baseline vs Engram.

    Args:
        simulation_days: Number of days to simulate
        event_complexity: Event complexity level
        num_runs: Number of runs per configuration
    """
    print(f"\n{'='*70}")
    print(f"Vending-Bench 2 Comparison Suite")
    print(f"{'='*70}")
    print(f"Simulation Days: {simulation_days}")
    print(f"Event Complexity: {event_complexity}")
    print(f"Runs per configuration: {num_runs}")
    print(f"{'='*70}\n")

    all_results = []

    for run_idx in range(num_runs):
        print(f"\n{'='*70}")
        print(f"Run {run_idx + 1}/{num_runs}")
        print(f"{'='*70}\n")

        # Run baseline
        print(f"\n[1/2] Running BASELINE agent...")
        baseline_results = run_experiment(
            agent_type="baseline",
            simulation_days=simulation_days,
            event_complexity=event_complexity
        )
        all_results.append({
            "run": run_idx + 1,
            "agent": "baseline",
            "results": baseline_results
        })

        # Run Engram
        print(f"\n[2/2] Running ENGRAM agent...")
        engram_results = run_experiment(
            agent_type="engram",
            simulation_days=simulation_days,
            event_complexity=event_complexity
        )
        all_results.append({
            "run": run_idx + 1,
            "agent": "engram",
            "results": engram_results
        })

    # Save comparison summary
    summary_path = f"./experiments/comparison_summary_{simulation_days}d_{event_complexity}.json"
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"Comparison Suite Complete")
    print(f"{'='*70}")
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run Vending-Bench 2 experiments"
    )

    # Experiment type
    parser.add_argument(
        "--agent",
        type=str,
        choices=["baseline", "subagent", "engram", "compare"],
        default="subagent",
        help="Agent type to run (default: subagent)"
    )

    # Simulation parameters
    parser.add_argument(
        "--days",
        type=int,
        default=3,
        help="Number of simulation days (default: 3)"
    )
    parser.add_argument(
        "--starting-cash",
        type=float,
        default=500.0,
        help="Starting cash balance (default: 500.0)"
    )
    parser.add_argument(
        "--complexity",
        type=str,
        choices=["simple", "medium", "full"],
        default="simple",
        help="Event complexity level (default: simple)"
    )

    # Model parameters (Engram only)
    parser.add_argument(
        "--memory-model",
        type=str,
        default="claude-sonnet-4-5",
        help="Model for memLLM-R (default: claude-sonnet-4-5)"
    )
    parser.add_argument(
        "--customer-model",
        type=str,
        default="claude-sonnet-4-5",
        help="Model for customer LLM (default: claude-sonnet-4-5)"
    )
    parser.add_argument(
        "--search-types",
        type=str,
        nargs="+",
        default=None,
        help="Allowed search types (default: all)"
    )
    parser.add_argument(
        "--subagent-model",
        type=str,
        default=None,
        help="Model for sub-agent (subagent mode, defaults to customer-model)"
    )

    # Comparison parameters
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Number of runs per configuration (for compare mode)"
    )

    # Output parameters
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./experiments/logs",
        help="Directory for experiment logs (default: ./experiments/logs)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Prefix for log filename identification (e.g., 'exp1' -> 'exp1_baseline_model_200d_...')"
    )
    parser.add_argument(
        "--email-system",
        action="store_true",
        default=False,
        help="Enable VendingBench 2 style email-based supplier negotiation (default: direct ordering)"
    )

    args = parser.parse_args()

    # Create experiment directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs("./experiments/storage", exist_ok=True)

    # Run experiment
    if args.agent == "compare":
        run_comparison_suite(
            simulation_days=args.days,
            event_complexity=args.complexity,
            num_runs=args.num_runs
        )
    else:
        run_experiment(
            agent_type=args.agent,
            simulation_days=args.days,
            starting_cash=args.starting_cash,
            event_complexity=args.complexity,
            memory_llm_model=args.memory_model,
            customer_llm_model=args.customer_model,
            subagent_llm_model=args.subagent_model,
            allowed_search_types=args.search_types,
            log_dir=args.log_dir,
            debug=args.debug,
            prefix=args.prefix,
            email_system_enabled=args.email_system
        )


if __name__ == "__main__":
    main()
