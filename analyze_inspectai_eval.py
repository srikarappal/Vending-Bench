"""
Analyze .eval log files from Vending-Bench experiments.

Usage:
    python analyze_eval.py <path_to_eval_file>
    python analyze_eval.py experiments/logs/baseline_20d.eval
"""

import argparse
import json
from collections import defaultdict
from inspect_ai.log import read_eval_log


def extract_tool_calls_from_messages(messages):
    """
    Extract tool calls from raw messages (for subagent mode where
    simulation_results isn't populated by basic_agent).
    """
    tool_calls = []
    current_day = 0

    for i, msg in enumerate(messages):
        # Check for tool calls in assistant messages
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_call_record = {
                    "tool": tc.function,
                    "input": tc.arguments if hasattr(tc, 'arguments') else {},
                    "tool_call_id": tc.id if hasattr(tc, 'id') else None,
                    "day": current_day
                }

                # Try to find the corresponding tool result
                for j in range(i + 1, min(i + 10, len(messages))):
                    result_msg = messages[j]
                    if result_msg.role == "tool":
                        tool_call_id = getattr(result_msg, 'tool_call_id', None)
                        if tool_call_id == tc.id or (tool_call_id is None and j == i + 1):
                            content = result_msg.content
                            if isinstance(content, str):
                                try:
                                    tool_call_record["result"] = json.loads(content)
                                except (json.JSONDecodeError, TypeError):
                                    tool_call_record["result"] = content
                            else:
                                tool_call_record["result"] = content

                            # Update current day from wait_for_next_day results
                            if tc.function == "wait_for_next_day":
                                result = tool_call_record.get("result", {})
                                if isinstance(result, dict) and "new_day" in result:
                                    current_day = result["new_day"]
                            break

                tool_calls.append(tool_call_record)

    return tool_calls


def analyze_eval(eval_path: str, verbose: bool = False):
    """
    Analyze an eval log file and print detailed statistics.

    Args:
        eval_path: Path to the .eval file
        verbose: If True, print detailed tool call information
    """
    print(f"\n{'='*70}")
    print(f"ANALYZING: {eval_path}")
    print(f"{'='*70}\n")

    # Read the eval log
    log = read_eval_log(eval_path)

    # Get sample and simulation results
    sample = log.samples[0]
    sim_results = sample.metadata.get("simulation_results", {})
    final_metrics = sim_results.get("final_metrics", {})
    tool_calls = sim_results.get("tool_calls", [])
    memory_stats = sim_results.get("memory_stats", {})

    # Detect architecture type
    architecture = sample.metadata.get("architecture", "baseline")
    print(f"Architecture: {architecture}")

    # If no tool_calls in simulation_results, extract from messages
    # This handles subagent mode where basic_agent doesn't populate our format
    if not tool_calls and hasattr(sample, 'messages') and sample.messages:
        print("(Extracting tool calls from raw messages...)")
        tool_calls = extract_tool_calls_from_messages(sample.messages)

    # === EXTRACT CASH BALANCE ===
    starting_cash = final_metrics.get('starting_cash', 500.0)
    # Try to get from final_metrics first (new format), fallback to tool calls
    final_cash_balance = final_metrics.get('final_cash_balance', 0)
    if final_cash_balance == 0:
        # Fallback: get from the last wait_for_next_day call
        for tc in reversed(tool_calls):
            if tc.get("tool") == "wait_for_next_day":
                final_cash_balance = tc.get("result", {}).get("cash_balance", 0)
                break

    # === FINAL METRICS ===
    print("=== FINAL METRICS ===")
    print(f"Starting Cash:      ${starting_cash:.2f}")
    print(f"Final Cash Balance: ${final_cash_balance:.2f}")
    print(f"Cash Gain/Loss:     ${final_cash_balance - starting_cash:.2f}")
    print(f"")
    print(f"Starting Net Worth: ${final_metrics.get('starting_net_worth', 0):.2f}")
    print(f"Final Net Worth:    ${final_metrics.get('final_net_worth', 0):.2f}")
    print(f"  - Cash:           ${final_cash_balance:.2f}")
    print(f"  - Storage Value:  ${final_metrics.get('storage_value', 0):.2f}")
    print(f"  - Machine Value:  ${final_metrics.get('machine_value', 0):.2f}")
    print(f"Profit/Loss:        ${final_metrics.get('profit_loss', 0):.2f}")
    print(f"")
    print(f"Total Revenue:      ${final_metrics.get('total_revenue', 0):.2f}")
    print(f"Total Costs:        ${final_metrics.get('total_costs', 0):.2f}")
    print(f"Days Simulated:     {final_metrics.get('days_simulated', 0)}")
    print(f"Days Profitable:    {final_metrics.get('days_profitable', 0)}")
    print(f"Messages Used:      {final_metrics.get('messages_used', 0)}")

    # === MEMORY STATS ===
    print("\n=== MEMORY STATS ===")
    scratchpad = memory_stats.get("scratchpad", {})
    kv_store = memory_stats.get("kv_store", {})
    print(f"Scratchpad entries: {scratchpad.get('num_entries', 0)}")
    print(f"Scratchpad keys:    {scratchpad.get('keys', [])}")
    print(f"KV Store entries:   {kv_store.get('num_entries', 0)}")
    kv_keys = kv_store.get('keys', {})
    if isinstance(kv_keys, dict):
        print(f"KV Store keys:      {list(kv_keys.keys())}")
    else:
        print(f"KV Store keys:      {kv_keys}")

    # === TOOL CALL COUNTS ===
    print("\n=== TOOL CALL COUNTS ===")
    tool_counts = defaultdict(int)
    for tc in tool_calls:
        tool_name = tc.get("tool", "unknown")
        tool_counts[tool_name] += 1

    for name, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
        print(f"  {name}: {count}")

    print(f"\nTotal tool calls: {len(tool_calls)}")

    # === SUBAGENT STATS (if applicable) ===
    subagent_calls = tool_counts.get("run_sub_agent", 0) + tool_counts.get("chat_with_sub_agent", 0)
    if subagent_calls > 0:
        print("\n=== SUBAGENT STATS ===")
        print(f"  run_sub_agent calls:      {tool_counts.get('run_sub_agent', 0)}")
        print(f"  chat_with_sub_agent calls: {tool_counts.get('chat_with_sub_agent', 0)}")
        print(f"  Total subagent interactions: {subagent_calls}")

    # === DAILY BREAKDOWN ===
    print("\n=== DAILY REVENUE BREAKDOWN ===")
    print(f"{'Day':<5} {'Revenue':>10} {'Units':>7} {'Cash After':>12}")
    print("-" * 40)

    total_daily_revenue = 0
    daily_revenues = []

    for tc in tool_calls:
        if tc.get("tool") == "wait_for_next_day":
            result = tc.get("result", {})
            day = result.get("new_day", "?")
            sales = result.get("overnight_sales", {})
            revenue = sales.get("total_revenue", 0)
            units = sales.get("total_units_sold", 0)
            cash = result.get("cash_balance", 0)

            print(f"{day:<5} ${revenue:>9.2f} {units:>7} ${cash:>11.2f}")
            total_daily_revenue += revenue
            daily_revenues.append({"day": day, "revenue": revenue, "units": units})

    print("-" * 40)
    print(f"{'TOTAL':<5} ${total_daily_revenue:>9.2f}")

    # === SALES BY PRODUCT ===
    print("\n=== SALES BY PRODUCT (from overnight sales) ===")
    product_sales = defaultdict(lambda: {"units": 0, "revenue": 0})

    for tc in tool_calls:
        if tc.get("tool") == "wait_for_next_day":
            result = tc.get("result", {})
            sales = result.get("overnight_sales", {})
            by_product = sales.get("sales_by_product", {})

            for product, data in by_product.items():
                product_sales[product]["units"] += data.get("quantity", 0)
                product_sales[product]["revenue"] += data.get("revenue", 0)

    print(f"{'Product':<12} {'Units':>7} {'Revenue':>10} {'Avg Price':>10}")
    print("-" * 45)
    for product, data in sorted(product_sales.items()):
        units = data["units"]
        revenue = data["revenue"]
        avg_price = revenue / units if units > 0 else 0
        print(f"{product:<12} {units:>7} ${revenue:>9.2f} ${avg_price:>9.2f}")

    # === STOCK MACHINE CALLS ===
    print("\n=== STOCK MACHINE ACTIVITY ===")
    stock_calls = [tc for tc in tool_calls if tc.get("tool") == "stock_machine"]

    stocked_by_product = defaultdict(int)
    for tc in stock_calls:
        input_data = tc.get("input", {})
        product = input_data.get("product", "unknown")
        quantity = input_data.get("quantity", 0)
        stocked_by_product[product] += quantity

    print(f"Total stock_machine calls: {len(stock_calls)}")
    for product, qty in sorted(stocked_by_product.items()):
        print(f"  {product}: {qty} units stocked")

    # === PRICE CHANGES ===
    print("\n=== PRICE CHANGES ===")
    price_calls = [tc for tc in tool_calls if tc.get("tool") == "set_price"]

    if price_calls:
        for tc in price_calls:
            day = tc.get("day", "?")
            input_data = tc.get("input", {})
            result = tc.get("result", {})
            product = input_data.get("product", "unknown")
            new_price = input_data.get("price", 0)
            old_price = result.get("old_price", 0)
            print(f"  Day {day}: {product} ${old_price:.2f} -> ${new_price:.2f}")
    else:
        print("  No price changes made")

    # === ORDER INVENTORY ===
    print("\n=== INVENTORY ORDERS ===")
    order_calls = [tc for tc in tool_calls if tc.get("tool") == "order_inventory"]

    if order_calls:
        total_order_cost = 0
        for tc in order_calls:
            day = tc.get("day", "?")
            input_data = tc.get("input", {})
            result = tc.get("result", {})
            product = input_data.get("product", "unknown")
            quantity = input_data.get("quantity", 0)
            cost = result.get("cost", 0)
            total_order_cost += cost
            print(f"  Day {day}: Ordered {quantity} {product} for ${cost:.2f}")
        print(f"\n  Total order costs: ${total_order_cost:.2f}")
    else:
        print("  No inventory orders made")

    # === ANALYSIS SUMMARY ===
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)

    days_simulated = final_metrics.get("days_simulated", 1)
    avg_daily_revenue = total_daily_revenue / days_simulated if days_simulated > 0 else 0

    print(f"Average daily revenue: ${avg_daily_revenue:.2f}")
    print(f"Revenue per tool call: ${total_daily_revenue / len(tool_calls):.2f}" if tool_calls else "N/A")

    # Check for anomalies
    print("\n=== POTENTIAL ISSUES ===")

    if avg_daily_revenue > 150:
        print(f"WARNING: Average daily revenue (${avg_daily_revenue:.2f}) seems very high!")
        print("         Andon Labs benchmark shows ~$5-15/day for most models")

    total_units_sold = sum(p["units"] for p in product_sales.values())
    if total_units_sold / days_simulated > 50:
        print(f"WARNING: Selling {total_units_sold/days_simulated:.1f} units/day - demand may be too high!")

    if not order_calls:
        print("NOTE: Agent never ordered inventory - only used starting stock")

    print("\n")

    return {
        "final_metrics": final_metrics,
        "tool_counts": dict(tool_counts),
        "daily_revenues": daily_revenues,
        "product_sales": dict(product_sales),
        "memory_stats": memory_stats
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Vending-Bench .eval log files"
    )
    parser.add_argument(
        "eval_path",
        type=str,
        help="Path to the .eval file to analyze"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print verbose output including all tool calls"
    )

    args = parser.parse_args()
    analyze_eval(args.eval_path, verbose=args.verbose)


if __name__ == "__main__":
    main()
