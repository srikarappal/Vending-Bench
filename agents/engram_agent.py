"""
Engram-powered vending machine agent.

Integrates customer-facing LLM with memLLM-R for long-term memory coherence.
"""

import sys
import os
import json
from typing import Dict, List, Any, Optional
from anthropic import Anthropic

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'engram-backend'))

from phase1_frontier_memLLM import FrontierMemoryLLM
from phase1_dumb_DBs import create_storage
from phase1_config import get_config

from src.environment import VendingEnvironment
from src.tools import VendingTools
from src.prompts import build_system_prompt
from agents.vending_prompts import build_vending_ingest_prompt, build_vending_retrieve_prompt


class EngramVendingAgent:
    """
    Vending machine agent powered by Engram architecture.

    Architecture:
    - Customer LLM: Makes business decisions and uses tools
    - memLLM-R: Manages long-term memory (ingest/retrieve)
    - Storage: Multi-modal storage (vector, graph, fulltext)
    """

    def __init__(
        self,
        customer_llm_model: str = "claude-sonnet-4-5",
        memory_llm_model: str = "claude-sonnet-4-5",
        storage_path: Optional[str] = None,
        allowed_search_types: Optional[List[str]] = None,
        debug: bool = False
    ):
        """
        Initialize Engram vending agent.

        Args:
            customer_llm_model: Model for business decisions (customer-facing LLM)
            memory_llm_model: Model for memory operations (memLLM-R)
            storage_path: Path to storage directory (if None, uses default)
            allowed_search_types: Search types for retrieval (default: all)
            debug: Enable debug mode
        """
        self.debug = debug

        # Initialize customer LLM (makes business decisions)
        self.customer_client = Anthropic()
        self.customer_model = customer_llm_model

        # Initialize memLLM-R (manages memory)
        config = get_config()
        if storage_path:
            config.storage.vector_store_path = f"{storage_path}/chroma"
            config.storage.text_store_path = f"{storage_path}/sqlite.db"
            config.storage.graph_store_path = f"{storage_path}/neo4j"

        storage = create_storage(config.storage)

        self.memllm = FrontierMemoryLLM(
            provider="anthropic",
            model=memory_llm_model,
            storage=storage,
            allowed_search_types=allowed_search_types or ["semantic", "fulltext", "graph"],
            debug=debug
        )

        # Track conversation history
        self.conversation_history: List[Dict[str, str]] = []

        # Track business context for memory operations
        self.business_context = {
            "current_day": 0,
            "recent_events": [],
            "recent_actions": []
        }

        if self.debug:
            print(f"✓ Initialized EngramVendingAgent")
            print(f"  Customer LLM: {customer_llm_model}")
            print(f"  Memory LLM: {memory_llm_model}")

    def reset(self):
        """Reset agent state for new simulation."""
        self.conversation_history = []
        self.business_context = {
            "current_day": 0,
            "recent_events": [],
            "recent_actions": []
        }

    def handle_event(
        self,
        event: Dict[str, Any],
        env: VendingEnvironment,
        tools: VendingTools
    ) -> Dict[str, Any]:
        """
        Handle a business event.

        Args:
            event: Event dictionary from EventGenerator
            env: Simulation environment
            tools: Available tools

        Returns:
            Dict with agent's response and actions taken
        """
        # Update business context
        self.business_context["current_day"] = env.current_day
        self.business_context["recent_events"].append(event)
        if len(self.business_context["recent_events"]) > 5:
            self.business_context["recent_events"] = self.business_context["recent_events"][-5:]

        # Ingest event into memory
        self._ingest_event(event)

        # Retrieve relevant memories for decision making
        memories = self._retrieve_memories(event)

        # Make business decision using customer LLM
        decision = self._make_decision(event, memories, env, tools)

        return decision

    def _ingest_event(self, event: Dict[str, Any]):
        """
        Ingest business event into memory.

        Args:
            event: Event dictionary
        """
        # Format event as business content
        content = self._format_event_for_ingest(event)

        # Build domain-specific ingest prompt
        prompt = build_vending_ingest_prompt(content)

        # Ingest using memLLM-R
        response = self.memllm.ingest(content, custom_prompt=prompt)

        if self.debug:
            print(f"\n📥 Ingested event (Day {event.get('day')}): {event.get('description', 'Unknown')}")
            print(f"   Stored {len(response.memory_ids)} memories")

    def _retrieve_memories(self, event: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories for event handling.

        Args:
            event: Current event

        Returns:
            List of retrieved memories
        """
        # Build query based on event type
        query = self._build_retrieval_query(event)

        # Build domain-specific retrieve prompt
        context = self._build_business_context()
        prompt = build_vending_retrieve_prompt(
            query=query,
            context=context,
            allowed_search_types=self.memllm.allowed_search_types
        )

        # Retrieve using memLLM-R
        response = self.memllm.retrieve(query, custom_prompt=prompt)

        if self.debug:
            print(f"\n📤 Retrieved {len(response.results)} memories for: {query}")

        return response.results

    def _make_decision(
        self,
        event: Dict[str, Any],
        memories: List[Dict[str, Any]],
        env: VendingEnvironment,
        tools: VendingTools
    ) -> Dict[str, Any]:
        """
        Make business decision using customer LLM.

        Args:
            event: Current event
            memories: Retrieved memories
            env: Simulation environment
            tools: Available tools

        Returns:
            Dict with decision and actions
        """
        # Build prompt for customer LLM
        system_prompt = self._build_system_prompt(tools, env)
        user_message = self._build_decision_prompt(event, memories, env)

        # Call customer LLM
        response = self.customer_client.messages.create(
            model=self.customer_model,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_message}
            ],
            max_tokens=4096,
            temperature=0.1,
            tools=self._get_tool_definitions(tools)
        )

        # Process response and execute tool calls
        actions_taken = []
        reasoning = ""

        for block in response.content:
            if block.type == "text":
                reasoning = block.text
                if self.debug:
                    print(f"\n💭 Agent reasoning: {reasoning}")

            elif block.type == "tool_use":
                # Execute tool call
                tool_result = self._execute_tool(block, tools)
                actions_taken.append({
                    "tool": block.name,
                    "input": block.input,
                    "result": tool_result
                })

                if self.debug:
                    print(f"\n🔧 Tool call: {block.name}({block.input})")
                    print(f"   Result: {tool_result}")

        # Ingest the decision and outcome for future reference
        if actions_taken:
            decision_content = self._format_decision_for_ingest(
                event, reasoning, actions_taken
            )
            self._ingest_event({
                "type": "decision",
                "day": event.get("day"),
                "description": decision_content
            })

        # Track actions in business context
        self.business_context["recent_actions"].extend(actions_taken)
        if len(self.business_context["recent_actions"]) > 10:
            self.business_context["recent_actions"] = self.business_context["recent_actions"][-10:]

        return {
            "event": event,
            "reasoning": reasoning,
            "actions": actions_taken,
            "memories_used": len(memories)
        }

    def _format_event_for_ingest(self, event: Dict[str, Any]) -> str:
        """Format event as business content for ingestion."""
        event_type = event.get("type", "unknown")
        day = event.get("day", 0)

        if event_type == "purchase":
            sales_summary = []
            for product, sale_data in event.get("sales", {}).items():
                sales_summary.append(
                    f"{sale_data['quantity']} units of {product} at ${sale_data['price']:.2f} "
                    f"(revenue: ${sale_data['revenue']:.2f})"
                )
            return f"Day {day}: Customer purchases: {', '.join(sales_summary)}. Total revenue: ${event.get('total_revenue', 0):.2f}."

        elif event_type == "email":
            return f"Day {day}: Email from {event.get('from', 'unknown')}: {event.get('subject', 'No subject')}"

        elif event_type == "competitor":
            prices = event.get("competitor_prices", {})
            return f"Day {day}: Competitor opened nearby. Their prices: {', '.join(f'{p}: ${v:.2f}' for p, v in prices.items())}"

        elif event_type == "maintenance":
            return f"Day {day}: Maintenance issue: {event.get('issue', 'unknown')} (cost: ${event.get('cost', 0):.2f})"

        elif event_type == "decision":
            return event.get("description", "Business decision made")

        else:
            return f"Day {day}: {event.get('description', 'Business event')}"

    def _build_retrieval_query(self, event: Dict[str, Any]) -> str:
        """Build retrieval query based on event type."""
        event_type = event.get("type", "unknown")

        if event_type == "purchase":
            products = list(event.get("sales", {}).keys())
            return f"Past sales patterns and pricing strategies for {', '.join(products)}"

        elif event_type == "email":
            subject = event.get("subject", "")
            return f"Supplier communications and {subject}"

        elif event_type == "competitor":
            return "Competitive pricing strategies and past responses to competition"

        elif event_type == "maintenance":
            return f"Maintenance costs and operational expenses"

        else:
            return "General business operations and past decisions"

    def _build_business_context(self) -> str:
        """Build current business context string."""
        day = self.business_context["current_day"]
        recent_events = self.business_context["recent_events"][-3:]

        context_parts = [f"Current day: {day}"]

        if recent_events:
            context_parts.append("Recent events:")
            for evt in recent_events:
                context_parts.append(f"- {evt.get('description', 'Unknown event')}")

        return "\n".join(context_parts)

    def _build_system_prompt(self, tools: VendingTools, env: VendingEnvironment) -> str:
        """Build system prompt for customer LLM using centralized Andon Labs specification."""
        base_prompt = build_system_prompt(
            tools=tools,
            starting_cash=env.config.starting_cash,
            daily_fee=env.config.daily_fee,
            simulation_days=env.config.simulation_days
        )

        # Add memory-specific context for Engram agent
        memory_context = """

## LONG-TERM MEMORY SYSTEM

You have access to a long-term memory system that helps you learn from past experiences:
- **Past sales data and patterns**: What products sold well, when, and at what prices
- **Pricing experiments and outcomes**: Results of past pricing changes
- **Supplier information**: Order history and supplier reliability
- **Business principles learned**: Strategies that worked or failed

When making decisions:
1. **Consider retrieved memories** from past experiences shown in your prompt
2. **Think about causal relationships**: What led to success or failure?
3. **Balance short-term and long-term profitability**: Don't just optimize for today
4. **Learn from mistakes**: Avoid repeating past failures

Be concise in your reasoning and decisive in your actions."""

        return base_prompt + memory_context

    def _format_tools_description(self, tools: VendingTools) -> str:
        """Format tool descriptions for system prompt."""
        tool_list = tools.get_tool_list()
        descriptions = []
        for tool in tool_list:
            params = ", ".join(f"{k}: {v}" for k, v in tool.get("parameters", {}).items())
            descriptions.append(f"- {tool['name']}({params}): {tool['description']}")
        return "\n".join(descriptions)

    def _build_decision_prompt(
        self,
        event: Dict[str, Any],
        memories: List[Dict[str, Any]],
        env: VendingEnvironment
    ) -> str:
        """Build decision prompt for customer LLM."""
        # Format event
        event_desc = event.get("description", "Unknown event")

        # Format memories
        memory_text = ""
        if memories:
            memory_text = "\n\nRelevant memories from past experiences:\n"
            for i, mem in enumerate(memories[:10], 1):  # Top 10 memories
                content = mem.get("content", "")
                memory_text += f"{i}. {content}\n"

        # Get current state
        state = env.get_state()

        prompt = f"""Day {state['day']} - Business Event:

{event_desc}

Current Business State:
- Cash Balance: ${state['cash_balance']:.2f}
- Machine Inventory: {', '.join(f'{k}={v}' for k, v in state['machine_inventory'].items())}
- Storage Inventory: {', '.join(f'{k}={v}' for k, v in state['storage_inventory'].items())}
- Current Prices: {', '.join(f'{k}=${v:.2f}' for k, v in state['prices'].items())}
{memory_text}

What actions should you take in response to this event?
Think about:
1. What does this event mean for the business?
2. What do past experiences tell us?
3. What actions make sense given our current state?
4. What are the expected outcomes?

Use the available tools to check state and take actions as needed."""

        return prompt

    def _get_tool_definitions(self, tools: VendingTools) -> List[Dict[str, Any]]:
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
                input_schema["required"].append(param_name)

            definitions.append({
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": input_schema
            })

        return definitions

    def _execute_tool(self, tool_call, tools: VendingTools) -> Dict[str, Any]:
        """Execute a tool call."""
        tool_name = tool_call.name
        tool_input = tool_call.input

        # Get the tool method
        if hasattr(tools, tool_name):
            tool_method = getattr(tools, tool_name)

            # Call the tool
            if tool_input:
                result = tool_method(**tool_input)
            else:
                result = tool_method()

            return result
        else:
            return {
                "success": False,
                "error": f"Unknown tool: {tool_name}"
            }

    def _format_decision_for_ingest(
        self,
        event: Dict[str, Any],
        reasoning: str,
        actions: List[Dict[str, Any]]
    ) -> str:
        """Format decision and outcome for ingestion."""
        day = event.get("day", 0)
        event_desc = event.get("description", "Unknown event")

        # Format actions taken
        action_summary = []
        for action in actions:
            tool = action["tool"]
            result = action["result"]
            if result.get("success"):
                action_summary.append(f"{tool}: {result.get('message', 'Success')}")

        decision_text = f"""Day {day} Decision:

Context: {event_desc}

Reasoning: {reasoning[:200]}...

Actions Taken:
{chr(10).join(f'- {a}' for a in action_summary)}

This decision can inform future business choices."""

        return decision_text
