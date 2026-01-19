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
from inspect_ai.log import transcript
from inspect_ai.util import display_counter

from config.simulation_config import SimulationConfig
from src.environment import VendingEnvironment
from src.tools import VendingTools
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
    Baseline agent using AGENT-DRIVEN simulation loop.

    Key architecture change: The agent controls time progression by calling
    wait_for_next_day. The agent can take multiple actions per day.
    """
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Initialize simulation
        env = VendingEnvironment(config)
        tools = VendingTools(env)

        # Initialize customer LLM
        client = Anthropic()
        customer_model = "claude-sonnet-4-5"

        # Build system prompt (updated for agent-driven architecture)
        system_prompt = build_system_prompt(
            tools=tools,
            starting_cash=config.starting_cash,
            daily_fee=config.daily_fee,
            simulation_days=config.simulation_days
        )

        # Get tool definitions for Claude API
        tool_definitions = _get_tool_definitions(tools)

        # Track all tool calls for logging
        all_tool_calls = []

        # Build initial morning briefing (Day 0 start)
        morning_briefing = _build_morning_briefing(env, is_first_day=True)

        # Add morning briefing to transcript
        state.messages.append(ChatMessageUser(content=morning_briefing))

        # Conversation history for the LLM (maintains context across tool calls)
        conversation_history = [{"role": "user", "content": morning_briefing}]

        # Track consecutive no-tool responses for continuation prompting
        consecutive_no_tool_responses = 0
        max_continuation_prompts = 3  # Prevent infinite loops

        # Progress logging - start
        print(f"\n{'='*60}")
        print(f"VENDING SIMULATION STARTED")
        print(f"  Model: {customer_model}")
        print(f"  Days: {config.simulation_days} | Starting Cash: ${config.starting_cash:.2f}")
        print(f"  Machine Capacity: 12 slots (6 small + 6 large)")
        print(f"{'='*60}")

        # Log to inspect transcript for live view
        transcript().info({
            "event": "simulation_start",
            "model": customer_model,
            "simulation_days": config.simulation_days,
            "starting_cash": config.starting_cash,
            "machine_capacity": 12
        })

        # Initial display counters
        display_counter("Day", f"0/{config.simulation_days}")
        display_counter("Cash Balance", f"${config.starting_cash:.2f}")
        display_counter("Cash +/-", "$0.00")
        display_counter("Daily Revenue", "$0.00")
        display_counter("Units Sold", "0")
        display_counter("Total Calls", "0")
        display_counter("Avg Calls/Day", "0.0")

        # Main agent-driven loop
        while not env.is_complete:
            # Call LLM with current conversation
            response = client.messages.create(
                model=customer_model,
                system=system_prompt,
                messages=conversation_history,
                max_tokens=4096,
                temperature=0.1,
                tools=tool_definitions
            )

            # Extract text reasoning (if any)
            reasoning_text = ""
            tool_uses = []

            for block in response.content:
                if block.type == "text":
                    reasoning_text = block.text
                elif block.type == "tool_use":
                    tool_uses.append(block)

            # Log assistant response to transcript
            if reasoning_text:
                state.messages.append(ChatMessageAssistant(content=reasoning_text))
                conversation_history.append({"role": "assistant", "content": reasoning_text})

            # Check if agent used any tools
            if not tool_uses:
                # No tools called - apply continuation prompting
                consecutive_no_tool_responses += 1

                if consecutive_no_tool_responses >= max_continuation_prompts:
                    # Agent is stuck, force end of day
                    state.messages.append(ChatMessageUser(
                        content="[SYSTEM] You haven't used any tools. Automatically advancing to next day..."
                    ))
                    # Manually call wait_for_next_day
                    tool_result = tools.wait_for_next_day()
                    if tool_result["is_simulation_complete"]:
                        break
                    # Reset and continue with new morning briefing
                    morning_briefing = tool_result["morning_briefing"]
                    state.messages.append(ChatMessageUser(content=morning_briefing))
                    conversation_history = [{"role": "user", "content": morning_briefing}]
                    consecutive_no_tool_responses = 0
                else:
                    # Send continuation prompt
                    continuation_msg = "Continue on your mission by using your tools. Remember: stock_machine puts items in the vending machine, and wait_for_next_day advances to the next day."
                    state.messages.append(ChatMessageUser(content=f"[CONTINUATION PROMPT] {continuation_msg}"))
                    conversation_history.append({"role": "user", "content": continuation_msg})
                continue

            # Reset no-tool counter since we got tool calls
            consecutive_no_tool_responses = 0

            # Process tool calls
            tool_results_for_llm = []

            for tool_use in tool_uses:
                tool_name = tool_use.name
                tool_input = tool_use.input
                tool_id = tool_use.id

                # Log tool use to transcript
                state.messages.append(ChatMessageAssistant(
                    content=f"[TOOL CALL] {tool_name}\nInput: {json.dumps(tool_input, indent=2)}"
                ))

                # Execute tool
                tool_result = _execute_tool(tool_use, tools)

                # Log result to transcript
                state.messages.append(ChatMessageUser(
                    content=f"[TOOL RESULT] {tool_name}\n{json.dumps(tool_result, indent=2)}"
                ))

                # Track for logging
                all_tool_calls.append({
                    "day": env.current_day,
                    "tool": tool_name,
                    "input": tool_input,
                    "result": tool_result
                })

                # Prepare result for LLM continuation
                tool_results_for_llm.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": json.dumps(tool_result)
                })

                # Special handling for wait_for_next_day
                if tool_name == "wait_for_next_day":
                    # Progress logging
                    sales = tool_result.get("overnight_sales", {})
                    new_day = tool_result.get("new_day", "?")
                    cash = tool_result.get("cash_balance", 0)
                    revenue = sales.get("total_revenue", 0)
                    units = sales.get("total_units_sold", 0)
                    print(f"  Day {new_day}: ${cash:.2f} cash | ${revenue:.2f} revenue | {units} units sold | {len(all_tool_calls)} tools")

                    # Update display counters for inspect-ai UI
                    cash_change = cash - config.starting_cash
                    cash_change_str = f"+${cash_change:.2f}" if cash_change >= 0 else f"-${abs(cash_change):.2f}"
                    total_calls = len(all_tool_calls)
                    avg_calls = total_calls / new_day if new_day > 0 else 0
                    display_counter("Day", f"{new_day}/{config.simulation_days}")
                    display_counter("Cash Balance", f"${cash:.2f}")
                    display_counter("Cash +/-", cash_change_str)
                    display_counter("Daily Revenue", f"${revenue:.2f}")
                    display_counter("Units Sold", str(units))
                    display_counter("Total Calls", str(total_calls))
                    display_counter("Avg Calls/Day", f"{avg_calls:.1f}")

                    # Log to inspect transcript for live view
                    transcript().info({
                        "event": "day_complete",
                        "day": new_day,
                        "cash_balance": cash,
                        "revenue": revenue,
                        "units_sold": units,
                        "total_tool_calls": len(all_tool_calls)
                    })

                    if tool_result.get("is_simulation_complete"):
                        # Simulation ended
                        env.is_complete = True
                        print(f"  Simulation complete at Day {new_day}")
                        break

                    # New day started - add morning briefing
                    morning_briefing = tool_result.get("morning_briefing", "")
                    state.messages.append(ChatMessageUser(content=f"\n{morning_briefing}"))

                    # Reset conversation for new day (fresh context)
                    conversation_history = [{"role": "user", "content": morning_briefing}]
                    tool_results_for_llm = []  # Clear since we're starting fresh
                    break  # Exit tool processing loop, continue main loop

            # If we didn't call wait_for_next_day, continue conversation with tool results
            if tool_results_for_llm and not env.is_complete:
                # Add assistant message with tool uses
                conversation_history.append({
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": tu.id, "name": tu.name, "input": tu.input} for tu in tool_uses]
                })
                # Add tool results
                conversation_history.append({
                    "role": "user",
                    "content": tool_results_for_llm
                })

            # Check for bankruptcy (handled by environment - 10 consecutive days of not paying fee)
            if env.is_complete and env.consecutive_bankrupt_days >= env.bankruptcy_threshold:
                bankruptcy_msg = f"⚠️  BANKRUPT! Could not pay daily fee for {env.consecutive_bankrupt_days} consecutive days. Cash balance: ${env.cash_balance:.2f}"
                state.messages.append(ChatMessageUser(content=bankruptcy_msg))
                print(bankruptcy_msg)
                break

            # Safety check: prevent infinite loops (max 1000 iterations)
            if len(all_tool_calls) > 1000:
                state.messages.append(ChatMessageUser(
                    content="[SYSTEM] Maximum tool calls reached. Ending simulation."
                ))
                break

        # Calculate final metrics
        metrics = env.calculate_final_metrics()

        # Get memory usage stats
        memory_stats = tools.get_memory_stats()

        # Progress logging - final summary
        print(f"\n{'='*60}")
        print(f"SIMULATION COMPLETE")
        print(f"  Final Net Worth: ${metrics['final_net_worth']:.2f}")
        print(f"  Profit/Loss: ${metrics['profit_loss']:.2f}")
        print(f"  Total Revenue: ${metrics['total_revenue']:.2f}")
        print(f"  Days Simulated: {metrics['days_simulated']}")
        print(f"  Total Tool Calls: {len(all_tool_calls)}")
        print(f"{'='*60}\n")

        # Log to inspect transcript for live view
        transcript().info({
            "event": "simulation_complete",
            "final_net_worth": metrics['final_net_worth'],
            "profit_loss": metrics['profit_loss'],
            "total_revenue": metrics['total_revenue'],
            "days_simulated": metrics['days_simulated'],
            "total_tool_calls": len(all_tool_calls)
        })

        # Store results in state
        state.metadata["simulation_results"] = {
            "final_metrics": metrics,
            "tool_calls": all_tool_calls,
            "memory_stats": memory_stats,
            "agent_type": "baseline"
        }

        # Add completion message
        completion_message = _format_simulation_output(metrics, memory_stats)
        state.messages.append(ChatMessageAssistant(content=completion_message))

        return state

    return solve


def _build_morning_briefing(env: VendingEnvironment, is_first_day: bool = False) -> str:
    """
    Build the morning briefing message for the agent.

    This is what the agent sees when they "wake up" each day.
    """
    state = env.get_state()

    if is_first_day:
        intro = f"""
════════════════════════════════════════════════════════════════════════════════
WELCOME TO YOUR VENDING MACHINE BUSINESS!
════════════════════════════════════════════════════════════════════════════════

You are starting Day {state['day']} with ${state['cash_balance']:.2f} in cash.

YOUR GOAL: Maximize your bank account balance over {env.config.simulation_days} days.

IMPORTANT - HOW THIS WORKS:
1. You have inventory in STORAGE that needs to be moved to the MACHINE
2. Customers can ONLY buy from the vending machine (not storage!)
3. Use stock_machine() to move items from storage to the vending machine
4. When you're done with today's activities, use wait_for_next_day()
5. Overnight, customers will buy from your machine based on your prices
6. You'll receive a sales report each morning

STARTING INVENTORY (in storage, NOT in machine yet!):
{chr(10).join(f'  - {product}: {qty} units' for product, qty in state['storage_inventory'].items())}

MACHINE INVENTORY (what customers can buy):
{chr(10).join(f'  - {product}: {qty} units' for product, qty in state['machine_inventory'].items())}

CURRENT PRICES:
{chr(10).join(f'  - {product}: ${price:.2f}' for product, price in state['prices'].items())}

DAILY OPERATING FEE: ${env.config.daily_fee:.2f} (charged each night)

What would you like to do? Start by stocking your vending machine!
"""
    else:
        intro = f"""
════════════════════════════════════════════════════════════════════════════════
DAY {state['day']} - MORNING BRIEFING
════════════════════════════════════════════════════════════════════════════════

CURRENT STATUS:
- Cash Balance: ${state['cash_balance']:.2f}
- Days Remaining: {state['days_remaining']}

MACHINE INVENTORY (what customers can buy):
{chr(10).join(f'  - {product}: {qty} units' for product, qty in state['machine_inventory'].items())}

STORAGE INVENTORY:
{chr(10).join(f'  - {product}: {qty} units' for product, qty in state['storage_inventory'].items())}

CURRENT PRICES:
{chr(10).join(f'  - {product}: ${price:.2f}' for product, price in state['prices'].items())}

What would you like to do today?
"""

    return intro


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


def _format_simulation_output(metrics: Dict[str, Any], memory_stats: Dict[str, Any] = None) -> str:
    """Format simulation results as output."""
    output = f"""Simulation Complete

Final Results:
- Net Worth: ${metrics['final_net_worth']:.2f}
- Profit/Loss: ${metrics['profit_loss']:.2f}
- Total Revenue: ${metrics['total_revenue']:.2f}
- Total Costs: ${metrics['total_costs']:.2f}
- Days Profitable: {metrics['days_profitable']}/{metrics['days_simulated']}
- Messages Used: {metrics['messages_used']}
"""
    if memory_stats:
        output += f"""
Memory Usage:
- Scratchpad entries: {memory_stats['scratchpad']['num_entries']}
- Key-Value store entries: {memory_stats['kv_store']['num_entries']}
"""
    return output


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
