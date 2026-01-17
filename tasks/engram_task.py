"""
Engram vending machine task - Agent with long-term memory.

This task runs an Engram-powered agent that uses memLLM-R for long-term
memory coherence across the simulation.
"""

from typing import Dict, List, Any
import os

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Scorer, Score, scorer, mean, accuracy
from inspect_ai.solver import Solver, solver, Generate, TaskState
from inspect_ai.model import ChatMessageUser, ChatMessageAssistant

from config.simulation_config import SimulationConfig
from src.environment import VendingEnvironment
from src.tools import VendingTools
from src.events import EventGenerator
from agents.engram_agent import EngramVendingAgent


@task
def vending_engram(
    simulation_days: int = 3,
    starting_cash: float = 500.0,
    event_complexity: str = "simple",
    memory_llm_model: str = "claude-sonnet-4-5",
    customer_llm_model: str = "claude-sonnet-4-5",
    allowed_search_types: List[str] = None,
    debug: bool = False
) -> Task:
    """
    Engram vending machine task with long-term memory.

    Args:
        simulation_days: Number of days to simulate (1, 3, 30, or 365)
        starting_cash: Starting cash balance
        event_complexity: Event complexity level ("simple", "medium", "full")
        memory_llm_model: Model for memLLM-R
        customer_llm_model: Model for customer-facing LLM
        allowed_search_types: Search types for retrieval (e.g., ["semantic", "fulltext", "graph"])
        debug: Enable debug mode

    Returns:
        inspect_ai Task
    """
    config = SimulationConfig(
        simulation_days=simulation_days,
        starting_cash=starting_cash,
        event_complexity=event_complexity,
        max_messages=2000
    )

    # Create dataset with single sample (the simulation)
    dataset = [
        Sample(
            input=f"Run a {simulation_days}-day vending machine simulation with Engram memory starting with ${starting_cash:.2f}",
            metadata={
                "simulation_days": simulation_days,
                "starting_cash": starting_cash,
                "event_complexity": event_complexity,
                "memory_llm_model": memory_llm_model,
                "customer_llm_model": customer_llm_model,
                "allowed_search_types": allowed_search_types or ["semantic", "fulltext", "graph"]
            }
        )
    ]

    return Task(
        dataset=dataset,
        solver=[engram_agent_solver(
            config=config,
            memory_llm_model=memory_llm_model,
            customer_llm_model=customer_llm_model,
            allowed_search_types=allowed_search_types,
            debug=debug
        )],
        scorer=[profit_scorer(), survival_scorer(), memory_efficiency_scorer()],
        name=f"vending_engram_{simulation_days}d"
    )


@solver
def engram_agent_solver(
    config: SimulationConfig,
    memory_llm_model: str = "claude-sonnet-4-5",
    customer_llm_model: str = "claude-sonnet-4-5",
    allowed_search_types: List[str] = None,
    debug: bool = False
) -> Solver:
    """
    Engram agent with long-term memory capabilities.

    Uses memLLM-R for memory operations and customer LLM for decisions.
    """
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Initialize simulation
        env = VendingEnvironment(config)
        tools = VendingTools(env)
        event_gen = EventGenerator(env, config.event_complexity)

        # Initialize Engram agent
        storage_path = f"./experiments/storage/{state.sample_id}"
        os.makedirs(storage_path, exist_ok=True)

        agent = EngramVendingAgent(
            customer_llm_model=customer_llm_model,
            memory_llm_model=memory_llm_model,
            storage_path=storage_path,
            allowed_search_types=allowed_search_types or ["semantic", "fulltext", "graph"],
            debug=debug
        )

        # Track all decisions for logging
        all_decisions = []
        memory_stats = {
            "total_ingests": 0,
            "total_retrievals": 0,
            "total_memories_stored": 0,
            "total_memories_retrieved": 0
        }

        # Run simulation
        for day in range(1, config.simulation_days + 1):
            # Generate daily events
            events = event_gen.generate_daily_events()

            # Handle each event with Engram agent
            for event in events:
                decision = agent.handle_event(event, env, tools)
                all_decisions.append(decision)

                # Track memory usage
                memory_stats["total_ingests"] += 1
                memory_stats["total_retrievals"] += 1
                memory_stats["total_memories_retrieved"] += decision.get("memories_used", 0)

            # Advance day
            report = env.advance_day()

            if debug:
                print(f"\n📅 Day {day} complete")
                print(f"   Cash: ${env.cash_balance:.2f}")
                print(f"   Machine inventory: {env.machine_inventory}")

            # Check if bankrupt
            if env.cash_balance < 0:
                print(f"⚠️  Bankrupt on day {day}!")
                break

        # Calculate final metrics
        metrics = env.calculate_final_metrics()

        # Get storage stats
        storage_stats = agent.memllm.storage.get_stats()
        memory_stats["total_memories_stored"] = (
            storage_stats.get("vector_count", 0) +
            storage_stats.get("text_count", 0) +
            storage_stats.get("graph_nodes", 0)
        )

        # Store results in state
        state.metadata["simulation_results"] = {
            "final_metrics": metrics,
            "decisions": all_decisions,
            "memory_stats": memory_stats,
            "storage_stats": storage_stats,
            "agent_type": "engram"
        }

        # Add completion message
        completion_message = _format_simulation_output(metrics, memory_stats)
        state.messages.append(ChatMessageAssistant(content=completion_message))

        return state

    return solve


def _format_simulation_output(
    metrics: Dict[str, Any],
    memory_stats: Dict[str, Any]
) -> str:
    """Format simulation results as output."""
    return f"""Simulation Complete (Engram Agent)

Final Results:
- Net Worth: ${metrics['final_net_worth']:.2f}
- Profit/Loss: ${metrics['profit_loss']:.2f}
- Total Revenue: ${metrics['total_revenue']:.2f}
- Total Costs: ${metrics['total_costs']:.2f}
- Days Profitable: {metrics['days_profitable']}/{metrics['days_simulated']}
- Messages Used: {metrics['messages_used']}

Memory Usage:
- Total Ingests: {memory_stats['total_ingests']}
- Total Retrievals: {memory_stats['total_retrievals']}
- Memories Stored: {memory_stats['total_memories_stored']}
- Memories Retrieved: {memory_stats['total_memories_retrieved']}
- Avg Memories/Retrieval: {memory_stats['total_memories_retrieved'] / max(1, memory_stats['total_retrievals']):.1f}
"""


@scorer(metrics=[mean()])
def profit_scorer() -> Scorer:
    """
    Score based on final profit/loss.

    Returns:
        Scorer that evaluates profit performance
    """
    async def score(state: TaskState, target: Any) -> Score:
        results = state.metadata.get("simulation_results", {})
        metrics = results.get("final_metrics", {})

        profit_loss = metrics.get("profit_loss", 0.0)

        # Normalize to 0-1 scale
        # Assume good performance is +$1000 profit, bad is -$500 loss
        normalized = (profit_loss + 500) / 1500
        normalized = max(0.0, min(1.0, normalized))

        return Score(
            value=normalized,
            answer=f"${profit_loss:.2f}",
            explanation=f"Profit/Loss: ${profit_loss:.2f}"
        )

    return score


@scorer(metrics=[accuracy()])
def survival_scorer() -> Scorer:
    """
    Score based on whether agent survived the simulation.

    Returns:
        Scorer that checks if agent didn't go bankrupt
    """
    async def score(state: TaskState, target: Any) -> Score:
        results = state.metadata.get("simulation_results", {})
        metrics = results.get("final_metrics", {})

        survived = metrics.get("final_net_worth", 0.0) > 0

        return Score(
            value=1.0 if survived else 0.0,
            answer="survived" if survived else "bankrupt",
            explanation=f"Agent {'survived' if survived else 'went bankrupt'}"
        )

    return score


@scorer(metrics=[mean()])
def memory_efficiency_scorer() -> Scorer:
    """
    Score based on memory usage efficiency.

    Measures how effectively the agent uses memory for decision-making.

    Returns:
        Scorer that evaluates memory efficiency
    """
    async def score(state: TaskState, target: Any) -> Score:
        results = state.metadata.get("simulation_results", {})
        memory_stats = results.get("memory_stats", {})

        # Calculate efficiency: memories retrieved per retrieval operation
        total_retrievals = memory_stats.get("total_retrievals", 0)
        total_memories_retrieved = memory_stats.get("total_memories_retrieved", 0)

        if total_retrievals == 0:
            efficiency = 0.0
        else:
            avg_memories_per_retrieval = total_memories_retrieved / total_retrievals
            # Normalize: assume good is 5-10 memories per retrieval
            efficiency = min(1.0, avg_memories_per_retrieval / 10.0)

        return Score(
            value=efficiency,
            answer=f"{total_memories_retrieved}/{total_retrievals}",
            explanation=f"Memory efficiency: {total_memories_retrieved} memories retrieved across {total_retrievals} operations"
        )

    return score
