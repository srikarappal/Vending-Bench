"""
Baseline vending machine task - Agent without memory.

This task runs a baseline agent that makes decisions based only on
current state without any long-term memory.
"""

import json
from typing import Dict, List, Any
from anthropic import Anthropic

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Scorer, Score, scorer, mean, accuracy
from inspect_ai.solver import Solver, solver, Generate, TaskState
from inspect_ai.model import ChatMessageUser, ChatMessageAssistant

from config.simulation_config import SimulationConfig
from src.environment import VendingEnvironment
from src.tools import VendingTools
from src.events import EventGenerator
from src.prompts import build_system_prompt


@task
def vending_baseline(
    simulation_days: int = 3,
    starting_cash: float = 500.0,
    event_complexity: str = "simple"
) -> Task:
    """
    Baseline vending machine task without memory.

    Args:
        simulation_days: Number of days to simulate (1, 3, 30, or 365)
        starting_cash: Starting cash balance
        event_complexity: Event complexity level ("simple", "medium", "full")

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
            input=f"Run a {simulation_days}-day vending machine simulation starting with ${starting_cash:.2f}",
            metadata={
                "simulation_days": simulation_days,
                "starting_cash": starting_cash,
                "event_complexity": event_complexity
            }
        )
    ]

    return Task(
        dataset=dataset,
        solver=[baseline_agent(config)],
        scorer=[profit_scorer(), survival_scorer()],
        name=f"vending_baseline_{simulation_days}d"
    )


@solver
def baseline_agent(config: SimulationConfig) -> Solver:
    """
    Baseline agent that operates without long-term memory.

    Makes decisions based only on current state and immediate events.
    """
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Initialize simulation
        env = VendingEnvironment(config)
        tools = VendingTools(env)
        event_gen = EventGenerator(env, config.event_complexity)

        # Initialize customer LLM
        client = Anthropic()
        customer_model = "claude-sonnet-4-5"

        # Build system prompt once (used for all interactions)
        system_prompt = build_system_prompt(
            tools=tools,
            starting_cash=config.starting_cash,
            daily_fee=config.daily_fee,
            simulation_days=config.simulation_days
        )

        # Track all decisions for logging
        all_decisions = []

        # Run simulation
        for day in range(1, config.simulation_days + 1):
            # Add day start message to transcript
            day_state = env.get_state()
            day_start_msg = f"""
═══════════════════════════════════════════════════════════
DAY {day} START
═══════════════════════════════════════════════════════════
Cash Balance: ${day_state['cash_balance']:.2f}
Storage Inventory: {', '.join(f'{k}={v}' for k, v in day_state['storage_inventory'].items())}
Machine Inventory: {', '.join(f'{k}={v}' for k, v in day_state['machine_inventory'].items())}
Current Prices: {', '.join(f'{k}=${v:.2f}' for k, v in day_state['prices'].items())}
"""
            state.messages.append(ChatMessageUser(content=day_start_msg))

            # Generate daily events
            events = event_gen.generate_daily_events()

            # Handle each event
            for event in events:
                # Make decision using customer LLM and capture EVERYTHING
                decision_result = _make_baseline_decision_with_transcript(
                    event, env, tools, client, customer_model, system_prompt
                )
                all_decisions.append(decision_result["decision"])

                # Add user message (event + state)
                state.messages.append(ChatMessageUser(content=decision_result["user_message"]))

                # Add assistant reasoning
                if decision_result["reasoning"]:
                    state.messages.append(ChatMessageAssistant(content=decision_result["reasoning"]))

                # Add tool uses and results
                for tool_interaction in decision_result["tool_interactions"]:
                    state.messages.append(ChatMessageAssistant(
                        content=f"[TOOL USE] {tool_interaction['tool_name']}\nInput: {tool_interaction['tool_input']}"
                    ))
                    state.messages.append(ChatMessageUser(
                        content=f"[TOOL RESULT]\n{tool_interaction['tool_result']}"
                    ))

            # Advance day
            day_report = env.advance_day()

            # Add day end summary
            day_end_msg = f"""
─────────────────────────────────────────────────────────
DAY {day} END
─────────────────────────────────────────────────────────
Cash Balance: ${env.cash_balance:.2f}
Daily Fee Charged: ${config.daily_fee:.2f}
"""
            if day_report.get("spoiled_items"):
                day_end_msg += f"\nSpoiled Items: {day_report['spoiled_items']}"

            state.messages.append(ChatMessageUser(content=day_end_msg))

            # Check if bankrupt
            if env.cash_balance < 0:
                bankruptcy_msg = f"⚠️  BANKRUPT on day {day}! Cash balance: ${env.cash_balance:.2f}"
                state.messages.append(ChatMessageUser(content=bankruptcy_msg))
                print(bankruptcy_msg)
                break

        # Calculate final metrics
        metrics = env.calculate_final_metrics()

        # Store results in state
        state.metadata["simulation_results"] = {
            "final_metrics": metrics,
            "decisions": all_decisions,
            "agent_type": "baseline"
        }

        # Add completion message
        completion_message = _format_simulation_output(metrics)
        state.messages.append(ChatMessageAssistant(content=completion_message))

        return state

    return solve


def _make_baseline_decision_with_transcript(
    event: Dict[str, Any],
    env: VendingEnvironment,
    tools: VendingTools,
    client: Anthropic,
    model: str,
    system_prompt: str
) -> Dict[str, Any]:
    """
    Make decision for baseline agent and capture EVERYTHING for transcript.

    Args:
        event: Current event
        env: Simulation environment
        tools: Available tools
        client: Anthropic client
        model: Model name
        system_prompt: Pre-built system prompt

    Returns:
        Dict with decision, transcript messages, and all interactions
    """
    # Build user prompt
    state = env.get_state()
    event_desc = event.get("description", "Unknown event")

    user_prompt = f"""Day {state['day']} - Business Event:

{event_desc}

Current State:
- Cash Balance: ${state['cash_balance']:.2f}
- Machine Inventory: {', '.join(f'{k}={v}' for k, v in state['machine_inventory'].items())}
- Storage Inventory: {', '.join(f'{k}={v}' for k, v in state['storage_inventory'].items())}
- Current Prices: {', '.join(f'{k}=${v:.2f}' for k, v in state['prices'].items())}

What actions should you take? Use tools to gather information and take actions."""

    # Get tool definitions
    tool_definitions = _get_tool_definitions(tools)

    # Call customer LLM
    response = client.messages.create(
        model=model,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        max_tokens=4096,
        temperature=0.1,
        tools=tool_definitions
    )

    # Process response and execute tools - capture EVERYTHING
    actions_taken = []
    reasoning = ""
    tool_interactions = []

    for block in response.content:
        if block.type == "text":
            reasoning = block.text

        elif block.type == "tool_use":
            # Execute tool
            tool_result = _execute_tool(block, tools)
            actions_taken.append({
                "tool": block.name,
                "input": block.input,
                "result": tool_result
            })

            # Capture for transcript
            tool_interactions.append({
                "tool_name": block.name,
                "tool_input": json.dumps(block.input, indent=2),
                "tool_result": json.dumps(tool_result, indent=2)
            })

    return {
        "decision": {
            "event": event,
            "reasoning": reasoning,
            "actions": actions_taken,
            "day": state['day']
        },
        "user_message": user_prompt,
        "reasoning": reasoning,
        "tool_interactions": tool_interactions,
        "response_metadata": {
            "model": response.model,
            "stop_reason": response.stop_reason,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens
            }
        }
    }


def _get_tool_definitions(tools: VendingTools) -> List[Dict[str, Any]]:
    """Get tool definitions for Claude API."""
    tool_list = tools.get_tool_list()
    definitions = []

    for tool in tool_list:
        input_schema = {
            "type": "object",
            "properties": {},
            "required": []
        }

        # Add parameters to schema
        for param_name, param_desc in tool.get("parameters", {}).items():
            # Infer type from description
            if "price" in param_name.lower() or "amount" in param_name.lower():
                param_type = "number"
            elif "quantity" in param_name.lower() or "count" in param_name.lower():
                param_type = "integer"
            else:
                param_type = "string"

            input_schema["properties"][param_name] = {
                "type": param_type,
                "description": param_desc
            }

            if tool.get("parameters"):  # If tool has parameters, they're required
                input_schema["required"].append(param_name)

        definitions.append({
            "name": tool["name"],
            "description": tool["description"],
            "input_schema": input_schema
        })

    return definitions


def _execute_tool(tool_call, tools: VendingTools) -> Dict[str, Any]:
    """Execute a tool call."""
    tool_name = tool_call.name
    tool_input = tool_call.input

    if hasattr(tools, tool_name):
        tool_method = getattr(tools, tool_name)
        if tool_input:
            return tool_method(**tool_input)
        else:
            return tool_method()
    else:
        return {"success": False, "error": f"Unknown tool: {tool_name}"}


def _format_simulation_output(metrics: Dict[str, Any]) -> str:
    """Format simulation results as output."""
    return f"""Simulation Complete

Final Results:
- Net Worth: ${metrics['final_net_worth']:.2f}
- Profit/Loss: ${metrics['profit_loss']:.2f}
- Total Revenue: ${metrics['total_revenue']:.2f}
- Total Costs: ${metrics['total_costs']:.2f}
- Days Profitable: {metrics['days_profitable']}/{metrics['days_simulated']}
- Messages Used: {metrics['messages_used']}
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
