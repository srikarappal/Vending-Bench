"""
Baseline vending machine task - Agent without memory.

This task runs a baseline agent that makes decisions based only on
current state without any long-term memory.

Uses inspect_ai's native model abstraction for multi-provider support.
Supports both direct tool access and sub-agent architecture (matching VendingBench).
"""

import json
from typing import Dict, List, Any, Tuple

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Scorer, Score, scorer, mean, accuracy
from inspect_ai.solver import Solver, solver, Generate, TaskState, basic_agent, system_message
from inspect_ai.model import ChatMessageUser, ChatMessageAssistant, get_model, execute_tools, trim_messages
from inspect_ai.tool import ToolDef
from inspect_ai.log import transcript
from inspect_ai.util import display_counter

# Multi-agent support
from multiagent_inspect import SubAgentConfig, init_sub_agents

from config.simulation_config import SimulationConfig
from src.environment import VendingEnvironment
from src.tools import VendingTools
from src.prompts import (
    build_system_prompt,
    build_subagent_system_prompt,
    build_main_agent_prompt_with_subagent
)


def create_direct_tools(vending_tools: VendingTools) -> List[ToolDef]:
    """
    Create tools that the main agent can access directly (remote/digital tools).

    Per VendingBench paper: "Tools related to tasks that can be carried out
    remotely are available directly to the agent"
    """

    async def check_balance() -> str:
        """Get current cash balance and net worth estimate."""
        result = vending_tools.check_balance()
        return json.dumps(result)

    async def check_storage_inventory() -> str:
        """Check inventory levels in storage warehouse."""
        result = vending_tools.check_storage_inventory()
        return json.dumps(result)

    async def order_inventory(product: str, quantity: int) -> str:
        """Order new inventory from supplier. Orders take 3 days to arrive."""
        result = vending_tools.order_inventory(product, quantity)
        return json.dumps(result)

    async def check_pending_orders() -> str:
        """Check status of orders currently in transit."""
        result = vending_tools.check_pending_orders()
        return json.dumps(result)

    async def research_market(query: str) -> str:
        """Research market information using internet search."""
        result = vending_tools.research_market(query)
        return json.dumps(result)

    async def wait_for_next_day() -> str:
        """End current day and advance to next day. Overnight sales will be processed."""
        result = vending_tools.wait_for_next_day()
        return json.dumps(result)

    # Memory tools
    async def scratchpad_write(key: str, content: str) -> str:
        """Write a note to the scratchpad."""
        result = vending_tools.scratchpad_write(key, content)
        return json.dumps(result)

    async def scratchpad_read(key: str) -> str:
        """Read a note from the scratchpad."""
        result = vending_tools.scratchpad_read(key)
        return json.dumps(result)

    async def scratchpad_list() -> str:
        """List all keys in the scratchpad."""
        result = vending_tools.scratchpad_list()
        return json.dumps(result)

    async def kv_store_write(key: str, value: str) -> str:
        """Write structured data to key-value store."""
        try:
            parsed_value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            parsed_value = value
        result = vending_tools.kv_store_write(key, parsed_value)
        return json.dumps(result)

    async def kv_store_read(key: str) -> str:
        """Read data from key-value store."""
        result = vending_tools.kv_store_read(key)
        return json.dumps(result)

    async def kv_store_list() -> str:
        """List all keys in the key-value store."""
        result = vending_tools.kv_store_list()
        return json.dumps(result)

    return [
        ToolDef(tool=check_balance, name="check_balance",
                description="Get current cash balance and net worth estimate."),
        ToolDef(tool=check_storage_inventory, name="check_storage_inventory",
                description="Check inventory levels in storage warehouse."),
        ToolDef(tool=order_inventory, name="order_inventory",
                description="Order new inventory from supplier. Orders take 3 days to arrive.",
                parameters={"product": "Product name to order", "quantity": "Number of units to order"}),
        ToolDef(tool=check_pending_orders, name="check_pending_orders",
                description="Check status of orders currently in transit."),
        ToolDef(tool=research_market, name="research_market",
                description="Research market information using internet search.",
                parameters={"query": "Search query for market research"}),
        ToolDef(tool=wait_for_next_day, name="wait_for_next_day",
                description="End current day and advance to next day. Overnight sales will be processed."),
        ToolDef(tool=scratchpad_write, name="scratchpad_write",
                description="Write a note to the scratchpad.",
                parameters={"key": "Key/name for this note", "content": "Text content to save"}),
        ToolDef(tool=scratchpad_read, name="scratchpad_read",
                description="Read a note from the scratchpad.",
                parameters={"key": "Key of the note to read"}),
        ToolDef(tool=scratchpad_list, name="scratchpad_list",
                description="List all keys in the scratchpad."),
        ToolDef(tool=kv_store_write, name="kv_store_write",
                description="Write structured data to key-value store.",
                parameters={"key": "Key for this data", "value": "JSON string of value to store"}),
        ToolDef(tool=kv_store_read, name="kv_store_read",
                description="Read data from key-value store.",
                parameters={"key": "Key to read"}),
        ToolDef(tool=kv_store_list, name="kv_store_list",
                description="List all keys in the key-value store."),
    ]


def create_physical_tools(vending_tools: VendingTools) -> List[ToolDef]:
    """
    Create tools that require physical world interaction (sub-agent tools).

    Per VendingBench paper: "some parts of operating a vending machine requires
    actions in the physical world" - accessed via sub-agent.
    """

    async def stock_machine(product: str, quantity: int) -> str:
        """Move items from storage to vending machine."""
        result = vending_tools.stock_machine(product, quantity)
        return json.dumps(result)

    async def collect_cash() -> str:
        """Collect revenue from vending machine sales."""
        result = vending_tools.collect_cash()
        return json.dumps(result)

    async def get_machine_inventory() -> str:
        """Get current inventory in the vending machine (what customers can buy)."""
        result = vending_tools.get_machine_inventory()
        return json.dumps(result)

    async def set_price(product: str, price: float) -> str:
        """Set selling price for a product on the vending machine."""
        result = vending_tools.set_price(product, price)
        return json.dumps(result)

    async def get_prices() -> str:
        """Get current prices for all products from the vending machine."""
        result = vending_tools.get_prices()
        return json.dumps(result)

    return [
        ToolDef(tool=stock_machine, name="stock_machine",
                description="Move items from storage to vending machine.",
                parameters={"product": "Product name (coffee, chocolate, chips, soda)", "quantity": "Number of units to stock"}),
        ToolDef(tool=collect_cash, name="collect_cash",
                description="Collect revenue from vending machine sales."),
        ToolDef(tool=get_machine_inventory, name="get_machine_inventory",
                description="Get current inventory in the vending machine (what customers can buy)."),
        ToolDef(tool=set_price, name="set_price",
                description="Set selling price for a product on the vending machine.",
                parameters={"product": "Product name", "price": "New price in dollars"}),
        ToolDef(tool=get_prices, name="get_prices",
                description="Get current prices for all products from the vending machine."),
    ]


def create_all_tools(vending_tools: VendingTools) -> List[ToolDef]:
    """
    Create ALL tools for direct access mode (no sub-agent).
    Used when use_subagent=False.
    """
    return create_direct_tools(vending_tools) + create_physical_tools(vending_tools)


def create_vending_tools(vending_tools: VendingTools) -> List[ToolDef]:
    """
    Create inspect_ai ToolDef objects from VendingTools instance.

    Uses ToolDef for dynamic tool creation at runtime.
    """

    # Define async tool functions with type annotations (required for ToolDef)
    async def check_balance() -> str:
        result = vending_tools.check_balance()
        return json.dumps(result)

    async def collect_cash() -> str:
        result = vending_tools.collect_cash()
        return json.dumps(result)

    async def get_machine_inventory() -> str:
        result = vending_tools.get_machine_inventory()
        return json.dumps(result)

    async def check_storage_inventory() -> str:
        result = vending_tools.check_storage_inventory()
        return json.dumps(result)

    async def stock_machine(product: str, quantity: int) -> str:
        result = vending_tools.stock_machine(product, quantity)
        return json.dumps(result)

    async def order_inventory(product: str, quantity: int) -> str:
        result = vending_tools.order_inventory(product, quantity)
        return json.dumps(result)

    async def check_pending_orders() -> str:
        result = vending_tools.check_pending_orders()
        return json.dumps(result)

    async def set_price(product: str, price: float) -> str:
        result = vending_tools.set_price(product, price)
        return json.dumps(result)

    async def get_prices() -> str:
        result = vending_tools.get_prices()
        return json.dumps(result)

    async def research_market(query: str) -> str:
        result = vending_tools.research_market(query)
        return json.dumps(result)

    async def wait_for_next_day() -> str:
        result = vending_tools.wait_for_next_day()
        return json.dumps(result)

    async def scratchpad_write(key: str, content: str) -> str:
        result = vending_tools.scratchpad_write(key, content)
        return json.dumps(result)

    async def scratchpad_read(key: str) -> str:
        result = vending_tools.scratchpad_read(key)
        return json.dumps(result)

    async def scratchpad_list() -> str:
        result = vending_tools.scratchpad_list()
        return json.dumps(result)

    async def kv_store_write(key: str, value: str) -> str:
        try:
            parsed_value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            parsed_value = value
        result = vending_tools.kv_store_write(key, parsed_value)
        return json.dumps(result)

    async def kv_store_read(key: str) -> str:
        result = vending_tools.kv_store_read(key)
        return json.dumps(result)

    async def kv_store_list() -> str:
        result = vending_tools.kv_store_list()
        return json.dumps(result)

    # Create ToolDef objects for each function
    return [
        ToolDef(
            tool=check_balance,
            name="check_balance",
            description="Get current cash balance and net worth estimate."
        ),
        ToolDef(
            tool=collect_cash,
            name="collect_cash",
            description="Collect revenue from vending machine sales."
        ),
        ToolDef(
            tool=get_machine_inventory,
            name="get_machine_inventory",
            description="Get current inventory in the vending machine (what customers can buy)."
        ),
        ToolDef(
            tool=check_storage_inventory,
            name="check_storage_inventory",
            description="Check inventory levels in storage warehouse."
        ),
        ToolDef(
            tool=stock_machine,
            name="stock_machine",
            description="Move items from storage to vending machine.",
            parameters={"product": "Product name (coffee, chocolate, chips, soda)", "quantity": "Number of units to stock"}
        ),
        ToolDef(
            tool=order_inventory,
            name="order_inventory",
            description="Order new inventory from supplier. Orders take 3 days to arrive.",
            parameters={"product": "Product name to order", "quantity": "Number of units to order"}
        ),
        ToolDef(
            tool=check_pending_orders,
            name="check_pending_orders",
            description="Check status of orders currently in transit."
        ),
        ToolDef(
            tool=set_price,
            name="set_price",
            description="Set selling price for a product.",
            parameters={"product": "Product name", "price": "New price in dollars"}
        ),
        ToolDef(
            tool=get_prices,
            name="get_prices",
            description="Get current prices for all products."
        ),
        ToolDef(
            tool=research_market,
            name="research_market",
            description="Research market information.",
            parameters={"query": "Search query for market research"}
        ),
        ToolDef(
            tool=wait_for_next_day,
            name="wait_for_next_day",
            description="End current day and advance to next day. Overnight sales will be processed."
        ),
        ToolDef(
            tool=scratchpad_write,
            name="scratchpad_write",
            description="Write a note to the scratchpad.",
            parameters={"key": "Key/name for this note", "content": "Text content to save"}
        ),
        ToolDef(
            tool=scratchpad_read,
            name="scratchpad_read",
            description="Read a note from the scratchpad.",
            parameters={"key": "Key of the note to read"}
        ),
        ToolDef(
            tool=scratchpad_list,
            name="scratchpad_list",
            description="List all keys in the scratchpad."
        ),
        ToolDef(
            tool=kv_store_write,
            name="kv_store_write",
            description="Write structured data to key-value store.",
            parameters={"key": "Key for this data", "value": "JSON string of value to store"}
        ),
        ToolDef(
            tool=kv_store_read,
            name="kv_store_read",
            description="Read data from key-value store.",
            parameters={"key": "Key to read"}
        ),
        ToolDef(
            tool=kv_store_list,
            name="kv_store_list",
            description="List all keys in the key-value store."
        ),
    ]


@task
def vending_baseline(
    simulation_days: int = 3,
    starting_cash: float = 500.0,
    event_complexity: str = "simple",
    customer_model: str = "anthropic/claude-sonnet-4-5-20241022"
) -> Task:
    """
    Baseline vending machine task without memory.

    Args:
        simulation_days: Number of days to simulate (1, 3, 30, or 365)
        starting_cash: Starting cash balance
        event_complexity: Event complexity level ("simple", "medium", "full")
        customer_model: Model to use for the agent

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
                "event_complexity": event_complexity,
                "customer_model": customer_model
            }
        )
    ]

    # Extract short model name for task name (e.g., "openai/gpt-4o" -> "gpt-4o")
    model_short = customer_model.split("/")[-1] if "/" in customer_model else customer_model

    return Task(
        dataset=dataset,
        solver=[baseline_agent(config)],
        scorer=[profit_scorer(), survival_scorer()],
        name=f"vending_{model_short}_{simulation_days}d",
        model=customer_model  # Pass the model to inspect_ai
    )


@solver
def baseline_agent(config: SimulationConfig) -> Solver:
    """
    Baseline agent using inspect_ai's native model abstraction.

    Uses get_model().generate() and execute_tools() for multi-provider support.
    """
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Initialize simulation
        env = VendingEnvironment(config)
        vending_tools = VendingTools(env)

        # Create inspect_ai tools from VendingTools
        tools = create_vending_tools(vending_tools)

        # Build system prompt
        system_prompt = build_system_prompt(
            tools=vending_tools,
            starting_cash=config.starting_cash,
            daily_fee=config.daily_fee,
            simulation_days=config.simulation_days
        )

        # Track all tool calls and model outputs for logging
        all_tool_calls = []
        all_model_outputs = []  # Store full model outputs including usage/reasoning
        total_usage = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "total_tokens": 0}

        # Build initial morning briefing (Day 0 start)
        morning_briefing = _build_morning_briefing(env, is_first_day=True)

        # Get the model (uses the model specified in Task or eval command)
        model = get_model()

        # Progress logging - start
        print(f"\n{'='*60}")
        print(f"VENDING SIMULATION STARTED")
        print(f"  Model: {model.name}")
        print(f"  Days: {config.simulation_days} | Starting Cash: ${config.starting_cash:.2f}")
        print(f"  Machine Capacity: 12 slots (6 small + 6 large)")
        print(f"{'='*60}")

        # Log to inspect transcript
        transcript().info({
            "event": "simulation_start",
            "model": model.name,
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

        # Initialize conversation with system prompt and morning briefing
        # Store the system prompt separately for context management
        system_message = f"{system_prompt}\n\n{morning_briefing}"
        state.messages = [
            ChatMessageUser(content=system_message)
        ]

        # Main agent-driven loop using inspect_ai's native abstractions
        while not env.is_complete:
            # Apply context window truncation using inspect_ai's built-in trim_messages()
            # This properly handles:
            # - Retaining system messages
            # - Preserving tool call/response pairs
            # - Keeping most recent conversation (preserve=0.7 by default)
            trimmed_messages = await trim_messages(state.messages, preserve=0.7)

            # Generate model response with tools
            output = await model.generate(
                input=trimmed_messages,
                tools=tools,
            )

            # Capture full model output for logging
            model_output_record = {
                "day": env.current_day,
                "message_content": output.message.content if hasattr(output.message, 'content') else None,
                "tool_calls": [{"function": tc.function, "arguments": tc.arguments, "id": tc.id}
                              for tc in (output.message.tool_calls or [])],
                "stop_reason": output.stop_reason if hasattr(output, 'stop_reason') else None,
            }

            # Capture usage statistics if available
            if hasattr(output, 'usage') and output.usage:
                usage = output.usage
                model_output_record["usage"] = {
                    "input_tokens": getattr(usage, 'input_tokens', 0),
                    "output_tokens": getattr(usage, 'output_tokens', 0),
                    "reasoning_tokens": getattr(usage, 'reasoning_tokens', 0) if hasattr(usage, 'reasoning_tokens') else 0,
                    "total_tokens": getattr(usage, 'total_tokens', 0),
                }
                # Accumulate totals
                total_usage["input_tokens"] += model_output_record["usage"]["input_tokens"] or 0
                total_usage["output_tokens"] += model_output_record["usage"]["output_tokens"] or 0
                total_usage["reasoning_tokens"] += model_output_record["usage"]["reasoning_tokens"] or 0
                total_usage["total_tokens"] += model_output_record["usage"]["total_tokens"] or 0

            # Capture reasoning content if available (extended thinking)
            if hasattr(output.message, 'reasoning') and output.message.reasoning:
                model_output_record["reasoning"] = output.message.reasoning

            all_model_outputs.append(model_output_record)

            # Add assistant response to messages
            state.messages.append(output.message)

            # Check if model made tool calls
            if output.message.tool_calls:
                # Execute tools using inspect_ai's execute_tools
                # execute_tools expects the full message list and finds tool calls in the last assistant message
                execute_result = await execute_tools(state.messages, tools)
                tool_messages = execute_result.messages
                state.messages.extend(tool_messages)

                # Track tool calls with results for logging
                for i, tc in enumerate(output.message.tool_calls):
                    # Get the corresponding tool result
                    tool_result = None
                    if i < len(tool_messages) and hasattr(tool_messages[i], 'content'):
                        try:
                            tool_result = json.loads(tool_messages[i].content) if isinstance(tool_messages[i].content, str) else tool_messages[i].content
                        except (json.JSONDecodeError, TypeError):
                            tool_result = tool_messages[i].content

                    all_tool_calls.append({
                        "day": env.current_day,
                        "tool": tc.function,
                        "input": tc.arguments,
                        "result": tool_result,
                        "tool_call_id": tc.id
                    })

                    # Special handling for wait_for_next_day
                    if tc.function == "wait_for_next_day":
                        # Find the specific tool message matching this tool call
                        for tm in tool_messages:
                            # Match by tool_call_id to get the correct result
                            if hasattr(tm, 'tool_call_id') and tm.tool_call_id == tc.id and hasattr(tm, 'content'):
                                try:
                                    result = json.loads(tm.content) if isinstance(tm.content, str) else tm.content
                                    if isinstance(result, dict) and "new_day" in result:
                                        sales = result.get("overnight_sales", {})
                                        new_day = result.get("new_day", "?")
                                        cash = result.get("cash_balance", 0)
                                        revenue = sales.get("total_revenue", 0)
                                        units = sales.get("total_units_sold", 0)

                                        print(f"  Day {new_day}: ${cash:.2f} cash | ${revenue:.2f} revenue | {units} units sold | {len(all_tool_calls)} tools")

                                        # Update display counters
                                        if isinstance(new_day, int):
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

                                        # Log to transcript
                                        transcript().info({
                                            "event": "day_complete",
                                            "day": new_day,
                                            "cash_balance": cash,
                                            "revenue": revenue,
                                            "units_sold": units,
                                            "total_tool_calls": len(all_tool_calls)
                                        })

                                        if result.get("is_simulation_complete"):
                                            env.is_complete = True
                                            print(f"  Simulation complete at Day {new_day}")
                                except (json.JSONDecodeError, TypeError):
                                    pass
                                break  # Found the matching tool message, stop searching
            else:
                # No tool calls - model might be done or need prompting
                if not env.is_complete:
                    # Add continuation prompt
                    state.messages.append(ChatMessageUser(
                        content="Continue managing your vending machine business. Use your tools to check inventory, stock the machine, and advance to the next day with wait_for_next_day()."
                    ))

            # Check for bankruptcy
            if env.is_complete and env.consecutive_bankrupt_days >= env.bankruptcy_threshold:
                print(f"⚠️  BANKRUPT! Could not pay daily fee for {env.consecutive_bankrupt_days} consecutive days.")
                break

            # Safety check: prevent infinite loops
            if len(all_tool_calls) > 2000:
                print("[SYSTEM] Maximum tool calls reached. Ending simulation.")
                break

        # Calculate final metrics
        metrics = env.calculate_final_metrics()
        memory_stats = vending_tools.get_memory_stats()

        # Progress logging - final summary
        print(f"\n{'='*60}")
        print(f"SIMULATION COMPLETE")
        print(f"  Final Cash Balance: ${metrics.get('final_cash_balance', 0):.2f}")
        print(f"  Final Net Worth: ${metrics['final_net_worth']:.2f}")
        print(f"  Profit/Loss: ${metrics['profit_loss']:.2f}")
        print(f"  Total Revenue: ${metrics['total_revenue']:.2f}")
        print(f"  Days Simulated: {metrics['days_simulated']}")
        print(f"  Total Tool Calls: {len(all_tool_calls)}")
        print(f"  Total Model Calls: {len(all_model_outputs)}")
        print(f"  Token Usage:")
        print(f"    Input:  {total_usage['input_tokens']:,}")
        print(f"    Output: {total_usage['output_tokens']:,}")
        if total_usage['reasoning_tokens'] > 0:
            print(f"    Reasoning: {total_usage['reasoning_tokens']:,}")
        print(f"    Total:  {total_usage['total_tokens']:,}")
        print(f"{'='*60}\n")

        # Log to transcript
        transcript().info({
            "event": "simulation_complete",
            "final_cash_balance": metrics.get('final_cash_balance', 0),
            "final_net_worth": metrics['final_net_worth'],
            "profit_loss": metrics['profit_loss'],
            "total_revenue": metrics['total_revenue'],
            "days_simulated": metrics['days_simulated'],
            "total_tool_calls": len(all_tool_calls),
            "total_model_calls": len(all_model_outputs),
            "total_usage": total_usage
        })

        # Store results in state (full output capture)
        state.metadata["simulation_results"] = {
            "final_metrics": metrics,
            "tool_calls": all_tool_calls,
            "model_outputs": all_model_outputs,  # Full model outputs with usage/reasoning
            "total_usage": total_usage,  # Aggregated token usage
            "memory_stats": memory_stats,
            "agent_type": "baseline",
            "model_name": model.name
        }

        # Add completion message
        state.messages.append(ChatMessageAssistant(
            content=f"Simulation Complete. Final Net Worth: ${metrics['final_net_worth']:.2f}"
        ))

        return state

    return solve


def _build_morning_briefing(env: VendingEnvironment, is_first_day: bool = False) -> str:
    """Build the morning briefing message for the agent."""
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


@scorer(metrics=[mean()])
def profit_scorer() -> Scorer:
    """Score based on final profit/loss."""
    async def score(state: TaskState, target: Any) -> Score:
        results = state.metadata.get("simulation_results", {})
        metrics = results.get("final_metrics", {})
        profit_loss = metrics.get("profit_loss", 0.0)

        # Normalize to 0-1 scale
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
    """Score based on whether agent survived the simulation."""
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


# =============================================================================
# SUB-AGENT ARCHITECTURE (using multiagent-inspect from Andon Labs)
# =============================================================================

@task
def vending_subagent(
    simulation_days: int = 3,
    starting_cash: float = 500.0,
    event_complexity: str = "simple",
    customer_model: str = "anthropic/claude-sonnet-4-5-20241022",
    subagent_model: str = None,
    max_subagent_steps: int = 10
) -> Task:
    """
    Vending machine task with sub-agent architecture (matches VendingBench paper).

    Uses multiagent-inspect from Andon Labs for sub-agent orchestration.
    Main agent has direct tools; physical world tools go through sub-agent.

    Args:
        simulation_days: Number of days to simulate
        starting_cash: Starting cash balance
        event_complexity: Event complexity level
        customer_model: Model for the main agent
        subagent_model: Model for the sub-agent (defaults to same as main)
        max_subagent_steps: Maximum steps the sub-agent can take per invocation
    """
    # Default sub-agent model to same as main agent
    resolved_subagent_model = subagent_model if subagent_model else customer_model

    config = SimulationConfig(
        simulation_days=simulation_days,
        starting_cash=starting_cash,
        event_complexity=event_complexity,
        max_messages=2000
    )

    # Initialize environment and tools
    env = VendingEnvironment(config)
    vending_tools = VendingTools(env)

    # Create tool sets
    direct_tools = create_direct_tools(vending_tools)
    physical_tools = create_physical_tools(vending_tools)

    # Build system prompt for main agent
    main_system_prompt = build_main_agent_prompt_with_subagent(
        starting_cash=config.starting_cash,
        daily_fee=config.daily_fee,
        simulation_days=config.simulation_days
    )

    # Configure sub-agent for physical world tasks
    physical_subagent = SubAgentConfig(
        tools=physical_tools,
        model=resolved_subagent_model,
        max_steps=max_subagent_steps
    )

    # Create main agent with sub-agent using multiagent-inspect
    # Set message_limit high enough for full simulation (~20 messages per day)
    agent = basic_agent(
        init=init_sub_agents([physical_subagent]),
        tools=direct_tools,
        message_limit=simulation_days * 25,  # Allow enough messages for full simulation
    )

    # Dataset
    dataset = [
        Sample(
            input=f"{main_system_prompt}\n\n{_build_morning_briefing(env, is_first_day=True)}",
            metadata={
                "simulation_days": simulation_days,
                "starting_cash": starting_cash,
                "event_complexity": event_complexity,
                "customer_model": customer_model,
                "subagent_model": resolved_subagent_model,
                "architecture": "subagent"
            }
        )
    ]

    model_short = customer_model.split("/")[-1] if "/" in customer_model else customer_model

    return Task(
        dataset=dataset,
        solver=agent,
        scorer=[profit_scorer(), survival_scorer()],
        name=f"vending_subagent_{model_short}_{simulation_days}d",
        model=customer_model
    )
