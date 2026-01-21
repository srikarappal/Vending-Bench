"""
Custom sub-agent implementation for Vending-Bench.

Replaces multiagent_inspect with a simpler, more robust implementation
that works across Anthropic, OpenAI, and Google models using inspect-ai's
model abstraction.

Key improvements over multiagent_inspect:
1. Proper message trimming that preserves tool call/result pairs
2. Works with inspect-ai's ToolDef objects directly
3. Better error handling and logging
4. Multi-provider support via inspect-ai's get_model()
"""

from typing import List, Callable, Any, Optional
from dataclasses import dataclass, field
import json

from inspect_ai.model import (
    get_model,
    ChatMessageSystem,
    ChatMessageUser,
    ChatMessageAssistant,
    ChatMessageTool,
    ChatMessage,
)
from inspect_ai.tool import ToolDef, tool
from inspect_ai.model._call_tools import call_tools


@dataclass
class SubAgentConfig:
    """Configuration for a sub-agent."""
    tools: List[Callable]  # Raw tool functions (not ToolDef)
    model: str  # Model name (e.g., "anthropic/claude-sonnet-4-5-20250929")
    max_steps: int = 10  # Maximum tool call steps
    max_tokens: int = 8000  # Max context tokens before trimming
    system_prompt: Optional[str] = None  # Custom system prompt
    description: str = "Sub-agent"  # Public description


@dataclass
class SubAgentState:
    """Runtime state for a sub-agent."""
    config: SubAgentConfig
    messages: List[ChatMessage] = field(default_factory=list)
    tool_defs: List[ToolDef] = field(default_factory=list)


def _trim_messages_safe(messages: List[ChatMessage], max_messages: int = 50) -> List[ChatMessage]:
    """
    Safely trim messages while preserving tool call/result pairs.

    Unlike multiagent_inspect's _trim_messages, this ensures we never
    orphan a tool_result without its corresponding tool_use.

    The key insight: tool_result messages MUST follow a tool_use message.
    If we cut between them, the API will reject the request.

    Strategy:
    1. Always keep system messages
    2. Find "safe cut points" - positions where we can cut without orphaning
    3. A safe cut point is BEFORE a user message or BEFORE an assistant message
       that doesn't have tool calls (or at the very start of non-system messages)

    Args:
        messages: List of chat messages
        max_messages: Maximum number of messages to keep

    Returns:
        Trimmed message list with intact tool call/result pairs
    """
    if len(messages) <= max_messages:
        return messages

    # Always keep system message if present
    system_msgs = [m for m in messages if isinstance(m, ChatMessageSystem)]
    non_system = [m for m in messages if not isinstance(m, ChatMessageSystem)]

    if not non_system:
        return system_msgs

    # Find safe cut points (indices where we can start keeping messages)
    # Safe points: before user messages, or before assistant messages without tool calls
    safe_cut_points = []
    for i, msg in enumerate(non_system):
        if isinstance(msg, ChatMessageUser):
            safe_cut_points.append(i)
        elif isinstance(msg, ChatMessageAssistant):
            # Only safe if this assistant message has no tool calls
            if not (hasattr(msg, 'tool_calls') and msg.tool_calls):
                safe_cut_points.append(i)

    if not safe_cut_points:
        # No safe cut points found - keep everything or just system
        # This shouldn't happen in normal conversation flow
        return system_msgs + non_system[-max_messages:]

    # Find the latest safe cut point that gives us <= max_messages
    target_start = len(non_system) - (max_messages - len(system_msgs))
    target_start = max(0, target_start)

    # Find the safe cut point >= target_start
    best_cut = safe_cut_points[0]
    for cut_point in safe_cut_points:
        if cut_point >= target_start:
            best_cut = cut_point
            break
        best_cut = cut_point  # Keep updating to get closest to target

    # Use the best cut point that's >= target (or the last one before target)
    for cut_point in safe_cut_points:
        if cut_point >= target_start:
            best_cut = cut_point
            break

    trimmed = non_system[best_cut:]

    # Final safety check: ensure first message isn't a tool result
    while trimmed and _is_tool_result_message(trimmed[0]):
        trimmed = trimmed[1:]

    return system_msgs + trimmed


def _is_tool_result_message(msg: ChatMessage) -> bool:
    """Check if message contains tool results."""
    # Check if it's a ChatMessageTool (inspect-ai's tool result type)
    if isinstance(msg, ChatMessageTool):
        return True
    # Check role attribute
    if hasattr(msg, 'role') and msg.role == 'tool':
        return True
    # Also check for tool content in message
    if hasattr(msg, 'content') and isinstance(msg.content, list):
        for item in msg.content:
            if hasattr(item, 'type') and item.type == 'tool_result':
                return True
    return False


def _create_tool_defs(tools: List[Callable]) -> List[ToolDef]:
    """Create ToolDef objects from callable functions."""
    tool_defs = []
    for t in tools:
        if isinstance(t, ToolDef):
            tool_defs.append(t)
        elif callable(t):
            # Wrap in ToolDef - inspect-ai will extract metadata from docstring/signature
            tool_defs.append(ToolDef(tool=t))
        else:
            raise ValueError(f"Tool must be callable or ToolDef, got {type(t)}")
    return tool_defs


async def run_subagent(
    config: SubAgentConfig,
    instruction: str,
    debug: bool = False
) -> str:
    """
    Run a sub-agent with the given instruction.

    Args:
        config: Sub-agent configuration
        instruction: Natural language instruction for the sub-agent
        debug: If True, print debug info

    Returns:
        Sub-agent's final response as a string
    """
    # Get the model using inspect-ai's abstraction (works with any provider)
    model = get_model(config.model)

    # Create tool definitions
    tool_defs = _create_tool_defs(config.tools)

    # Build system prompt
    system_prompt = config.system_prompt or f"""You are a helpful assistant that performs physical tasks.

You have access to the following tools:
{chr(10).join(f'- {td.name}: {td.description}' for td in tool_defs)}

When given an instruction, use your tools to complete the task, then provide a summary of what you did.
Be precise and efficient. Complete the requested tasks and report results accurately."""

    # Initialize messages
    messages: List[ChatMessage] = [
        ChatMessageSystem(content=system_prompt),
        ChatMessageUser(content=instruction)
    ]

    # Run agent loop
    final_response = ""

    for step in range(config.max_steps):
        # Trim messages if needed (safely preserving tool pairs)
        messages = _trim_messages_safe(messages, max_messages=40)

        if debug:
            print(f"    [SubAgent Step {step+1}] {len(messages)} messages", flush=True)

        # Generate response
        try:
            output = await model.generate(input=messages, tools=tool_defs)
        except Exception as e:
            error_msg = f"Sub-agent generation error: {str(e)}"
            if debug:
                print(f"    [SubAgent ERROR] {error_msg}", flush=True)
            return error_msg

        # Add assistant message to history
        messages.append(output.message)

        # Check if model made tool calls
        if output.message.tool_calls:
            if debug:
                tool_names = [tc.function for tc in output.message.tool_calls]
                print(f"    [SubAgent] Calling tools: {tool_names}", flush=True)

            # Execute tools using inspect-ai's call_tools
            try:
                tool_results = await call_tools(output.message, tool_defs)
                messages.extend(tool_results)
            except Exception as e:
                error_msg = f"Tool execution error: {str(e)}"
                if debug:
                    print(f"    [SubAgent ERROR] {error_msg}", flush=True)
                return error_msg
        else:
            # No tool calls - agent is done
            final_response = output.message.content
            if isinstance(final_response, list):
                # Extract text from content blocks
                final_response = " ".join(
                    item.text if hasattr(item, 'text') else str(item)
                    for item in final_response
                    if hasattr(item, 'text') or isinstance(item, str)
                )
            break
    else:
        # Max steps reached
        final_response = f"Sub-agent reached maximum steps ({config.max_steps}). Last action may be incomplete."

    if debug:
        short_response = final_response[:100] + "..." if len(final_response) > 100 else final_response
        print(f"    [SubAgent] Done: {short_response}", flush=True)

    return final_response


def create_subagent_tool(config: SubAgentConfig, debug: bool = False) -> ToolDef:
    """
    Create a tool that runs the sub-agent.

    This tool can be added to the main agent's tool list, allowing
    the main agent to delegate tasks to the sub-agent.

    Args:
        config: Sub-agent configuration
        debug: If True, print debug info during sub-agent execution

    Returns:
        ToolDef for running the sub-agent
    """

    async def run_sub_agent(instruction: str) -> str:
        """
        Run the sub-agent with the given instruction.

        Args:
            instruction: Natural language instruction describing the task

        Returns:
            Sub-agent's response describing what was done
        """
        return await run_subagent(config, instruction, debug=debug)

    return ToolDef(
        tool=run_sub_agent,
        name="run_sub_agent",
        description=f"Delegate a task to the sub-agent. {config.description}",
        parameters={
            "instruction": "Natural language instruction for the sub-agent (e.g., 'Stock 6 chips and 6 chocolate in the vending machine')"
        }
    )


def create_subagent_tools(configs: List[SubAgentConfig], debug: bool = False) -> List[ToolDef]:
    """
    Create tools for multiple sub-agents.

    If only one config is provided, creates a single `run_sub_agent` tool.
    If multiple configs are provided, creates named tools like `run_sub_agent_0`, etc.

    Args:
        configs: List of sub-agent configurations
        debug: If True, print debug info

    Returns:
        List of ToolDef objects for running sub-agents
    """
    if len(configs) == 1:
        return [create_subagent_tool(configs[0], debug=debug)]

    tools = []
    for i, config in enumerate(configs):
        tool_def = create_subagent_tool(config, debug=debug)
        # Rename for multiple agents
        tool_def = ToolDef(
            tool=tool_def.tool,
            name=f"run_sub_agent_{i}",
            description=f"Run sub-agent {i}: {config.description}",
            parameters=tool_def.parameters
        )
        tools.append(tool_def)

    return tools
