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

    # Check if run completed with samples
    if not log.samples:
        print("ERROR: No samples found in eval log.")
        print("This usually means the experiment was cancelled before completing.")
        print("\nAttempting to extract partial data from log...")

        # Try to extract what we can from the log
        if hasattr(log, 'status'):
            print(f"Log status: {log.status}")
        if hasattr(log, 'error') and log.error:
            print(f"Error: {log.error}")

        # Check if there are any events or partial data
        if hasattr(log, 'events') and log.events:
            print(f"Found {len(log.events)} events in partial log")

        return

    # Get sample and simulation results
    sample = log.samples[0]
    sim_results = sample.metadata.get("simulation_results", {})
    final_metrics = sim_results.get("final_metrics", {})
    tool_calls = sim_results.get("tool_calls", [])
    memory_stats = sim_results.get("memory_stats", {})

    # Detect architecture type
    architecture = sample.metadata.get("architecture", "baseline")
    print(f"Architecture: {architecture}")

    # Detect ordering mode (email vs direct) based on tool calls present
    # We'll refine this after extracting tool calls
    email_mode = False  # Will be set after tool call extraction

    # If no tool_calls in simulation_results, extract from messages
    # This handles subagent mode where basic_agent doesn't populate our format
    if not tool_calls and hasattr(sample, 'messages') and sample.messages:
        print("(Extracting tool calls from raw messages...)")
        tool_calls = extract_tool_calls_from_messages(sample.messages)

    # Detect ordering mode based on tool calls
    email_tools = ["send_supplier_email", "list_supplier_emails", "read_supplier_email",
                   "search_suppliers", "send_payment"]
    direct_tools = ["order_inventory"]

    email_tool_calls = [tc for tc in tool_calls if tc.get("tool") in email_tools]
    direct_tool_calls = [tc for tc in tool_calls if tc.get("tool") in direct_tools]

    if email_tool_calls:
        email_mode = True
        print(f"Ordering Mode: EMAIL (supplier negotiation) - {len(email_tool_calls)} email-related calls")
    elif direct_tool_calls:
        email_mode = False
        print(f"Ordering Mode: DIRECT (fixed prices) - {len(direct_tool_calls)} order calls")
    else:
        # Neither found - likely used starting stock only
        email_mode = False
        print("Ordering Mode: UNKNOWN (no orders detected)")

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
    print(f"  - Cash Balance:   ${final_cash_balance:.2f}")
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
    # Check both direct orders (order_inventory) and email mode orders (send_payment)
    order_calls = [tc for tc in tool_calls if tc.get("tool") == "order_inventory"]
    payment_calls = [tc for tc in tool_calls if tc.get("tool") == "send_payment"]

    if order_calls:
        print("  Direct Orders (order_inventory):")
        total_order_cost = 0
        for tc in order_calls:
            day = tc.get("day", "?")
            input_data = tc.get("input", {})
            result = tc.get("result", {})
            product = input_data.get("product", "unknown")
            quantity = input_data.get("quantity", 0)
            cost = result.get("cost", 0)
            total_order_cost += cost
            print(f"    Day {day}: Ordered {quantity} {product} for ${cost:.2f}")
        print(f"  Total direct order costs: ${total_order_cost:.2f}")

    if payment_calls:
        print("  Email Mode Orders (send_payment):")
        total_payment_cost = 0
        for tc in payment_calls:
            day = tc.get("day", "?")
            input_data = tc.get("input", {})
            result = tc.get("result", {})
            supplier = input_data.get("to", "unknown")
            amount = input_data.get("amount", 0)
            products = input_data.get("products", {})
            total_payment_cost += amount
            products_str = ", ".join(f"{q} {p}" for p, q in products.items()) if isinstance(products, dict) else str(products)
            print(f"    Day {day}: Paid ${amount:.2f} to {supplier} for {products_str}")
        print(f"  Total email order costs: ${total_payment_cost:.2f}")

    if not order_calls and not payment_calls:
        print("  No inventory orders made")

    # === EMAIL ACTIVITY (email mode only) ===
    if email_mode:
        print("\n=== EMAIL ACTIVITY ===")
        email_sent = [tc for tc in tool_calls if tc.get("tool") == "send_supplier_email"]
        email_read = [tc for tc in tool_calls if tc.get("tool") == "read_supplier_email"]
        email_list = [tc for tc in tool_calls if tc.get("tool") == "list_supplier_emails"]
        supplier_search = [tc for tc in tool_calls if tc.get("tool") == "search_suppliers"]

        print(f"  Supplier searches:    {len(supplier_search)}")
        print(f"  Emails sent:          {len(email_sent)}")
        print(f"  Emails read:          {len(email_read)}")
        print(f"  Inbox checks:         {len(email_list)}")
        print(f"  Payments made:        {len(payment_calls)}")

        # Show suppliers contacted
        suppliers_contacted = set()
        for tc in email_sent:
            to_addr = tc.get("input", {}).get("to", "unknown")
            suppliers_contacted.add(to_addr)
        for tc in payment_calls:
            to_addr = tc.get("input", {}).get("to", "unknown")
            suppliers_contacted.add(to_addr)

        if suppliers_contacted:
            print(f"\n  Suppliers contacted: {len(suppliers_contacted)}")
            for supplier in sorted(suppliers_contacted):
                print(f"    - {supplier}")

    # === ZERO REVENUE ANALYSIS ===
    print("\n=== ZERO REVENUE ANALYSIS ===")
    zero_revenue_days = []
    current_streak = []
    longest_streak = []

    for dr in daily_revenues:
        if dr["revenue"] == 0:
            current_streak.append(dr["day"])
        else:
            if len(current_streak) > len(longest_streak):
                longest_streak = current_streak.copy()
            if len(current_streak) >= 3:  # Track streaks of 3+ days
                zero_revenue_days.append((current_streak[0], current_streak[-1], len(current_streak)))
            current_streak = []

    # Check final streak
    if len(current_streak) > len(longest_streak):
        longest_streak = current_streak.copy()
    if len(current_streak) >= 3:
        zero_revenue_days.append((current_streak[0], current_streak[-1], len(current_streak)))

    total_zero_days = sum(1 for dr in daily_revenues if dr["revenue"] == 0)
    print(f"Total days with $0 revenue: {total_zero_days}/{len(daily_revenues)} ({100*total_zero_days/len(daily_revenues):.1f}%)")

    if longest_streak:
        print(f"Longest zero-revenue streak: {len(longest_streak)} days (Day {longest_streak[0]} to {longest_streak[-1]})")

    if zero_revenue_days:
        print("Zero-revenue streaks (3+ days):")
        for start, end, length in zero_revenue_days:
            print(f"  Days {start}-{end}: {length} consecutive days")

    # === AGENT ACTIVITY ANALYSIS ===
    print("\n=== AGENT ACTIVITY ANALYSIS ===")

    # Analyze activity by day ranges
    days_simulated = final_metrics.get("days_simulated", 200)
    subagent_by_period = defaultdict(int)
    orders_by_period = defaultdict(int)
    email_by_period = defaultdict(int)

    for tc in tool_calls:
        day = tc.get("day", 0)
        period = (day // 50) * 50  # 0-49, 50-99, 100-149, 150-199
        if tc.get("tool") in ["run_sub_agent", "chat_with_sub_agent"]:
            subagent_by_period[period] += 1
        if tc.get("tool") in ["order_inventory", "send_payment"]:
            orders_by_period[period] += 1
        if tc.get("tool") in ["send_supplier_email", "read_supplier_email", "list_supplier_emails"]:
            email_by_period[period] += 1

    if email_mode:
        print(f"{'Period':<15} {'Email Activity':>15} {'Orders':>10}")
        print("-" * 45)
        for period in [0, 50, 100, 150]:
            period_end = min(period + 49, days_simulated - 1)
            if period < days_simulated:
                print(f"Days {period:3d}-{period_end:<3d}     {email_by_period.get(period, 0):>15} {orders_by_period.get(period, 0):>10}")
    else:
        print(f"{'Period':<15} {'Subagent Calls':>15} {'Orders':>10}")
        print("-" * 45)
        for period in [0, 50, 100, 150]:
            period_end = min(period + 49, days_simulated - 1)
            if period < days_simulated:
                print(f"Days {period:3d}-{period_end:<3d}     {subagent_by_period.get(period, 0):>15} {orders_by_period.get(period, 0):>10}")

    # === INVENTORY FLOW ANALYSIS ===
    print("\n=== INVENTORY FLOW ANALYSIS ===")
    storage_value = final_metrics.get("storage_value", 0)
    machine_value = final_metrics.get("machine_value", 0)

    print(f"Final Storage Value: ${storage_value:.2f}")
    print(f"Final Machine Value: ${machine_value:.2f}")

    if storage_value > 20 and machine_value < 5:
        print("⚠️  WARNING: High storage, empty machine!")
        print("   Items are stuck in storage - agent should use stock_machine tool")

    if tool_counts.get("stock_machine", 0) == 0:
        print("⚠️  WARNING: Agent never called stock_machine!")
        print("   Items ordered go to storage, need stock_machine to move to vending machine")

    # === ANALYSIS SUMMARY ===
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)

    avg_daily_revenue = total_daily_revenue / days_simulated if days_simulated > 0 else 0

    print(f"Average daily revenue: ${avg_daily_revenue:.2f}")
    print(f"Revenue per tool call: ${total_daily_revenue / len(tool_calls):.2f}" if tool_calls else "N/A")

    # Activity ratio (mode-aware)
    if email_mode:
        total_email_activity = (tool_counts.get("send_supplier_email", 0) +
                                tool_counts.get("read_supplier_email", 0) +
                                tool_counts.get("send_payment", 0))
        activity_ratio = total_email_activity / days_simulated if days_simulated > 0 else 0
        print(f"Email actions per day: {activity_ratio:.2f}")
    else:
        total_subagent = tool_counts.get("run_sub_agent", 0) + tool_counts.get("chat_with_sub_agent", 0)
        activity_ratio = total_subagent / days_simulated if days_simulated > 0 else 0
        print(f"Subagent calls per day: {activity_ratio:.2f}")

    # Check for anomalies
    print("\n=== POTENTIAL ISSUES ===")

    if avg_daily_revenue > 150:
        print(f"⚠️  Average daily revenue (${avg_daily_revenue:.2f}) seems very high!")
        print("    Andon Labs benchmark shows ~$5-15/day for most models")

    total_units_sold = sum(p["units"] for p in product_sales.values())
    if total_units_sold / days_simulated > 50:
        print(f"⚠️  Selling {total_units_sold/days_simulated:.1f} units/day - demand may be too high!")

    # Check if agent ever ordered (via direct or email mode)
    order_calls_check = [tc for tc in tool_calls if tc.get("tool") == "order_inventory"]
    payment_calls_check = [tc for tc in tool_calls if tc.get("tool") == "send_payment"]
    if not order_calls_check and not payment_calls_check:
        print("⚠️  Agent never ordered inventory - only used starting stock")

    if total_zero_days > days_simulated * 0.3:
        print(f"⚠️  {total_zero_days} days ({100*total_zero_days/days_simulated:.0f}%) with zero revenue!")
        print("    Agent may have run out of inventory or stopped engaging")

    # Only warn about low subagent activity if NOT in email mode
    # Email mode doesn't use subagents - it uses email tools directly
    if not email_mode and activity_ratio < 0.5:
        print(f"⚠️  Low agent activity ({activity_ratio:.2f} subagent calls/day)")
        print("    Agent may have stopped engaging with the simulation")

    # For email mode, check email activity instead
    if email_mode:
        email_activity = len([tc for tc in tool_calls if tc.get("tool") in
                             ["send_supplier_email", "read_supplier_email", "send_payment"]])
        email_ratio = email_activity / days_simulated if days_simulated > 0 else 0
        if email_ratio < 0.1 and not payment_calls_check:
            print(f"⚠️  Low email activity ({email_ratio:.2f} email actions/day)")
            print("    Agent may not be engaging with the supplier system")

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
