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
from inspect_ai.model import ChatMessageUser, ChatMessageAssistant, get_model, execute_tools, compaction, CompactionTrim
from inspect_ai.tool import ToolDef
from inspect_ai.log import transcript
from inspect_ai.util import display_counter

# Custom sub-agent support (replaces multiagent_inspect for better cross-provider compatibility)
from src.subagent import SubAgentConfig, create_subagent_tool

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
        result = vending_tools.research_product(query)
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
                description="Move items from storage to vending machine. TIP: Stock 5-10 units per call instead of repeated 1-unit calls to reduce tool costs.",
                parameters={"product": "Product name (coffee, chocolate, chips, soda)", "quantity": "Number of units to stock (recommend 5-10 per call)"}),
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


def create_open_search_tools(vending_tools: VendingTools) -> List[ToolDef]:
    """
    Create tools for OPEN PRODUCT SEARCH mode (expanded product universe).

    In this mode:
    - search_internet tool is available for discovering suppliers/products
    - Expanded product universe with 40+ products
    - Discoverable suppliers (10+) instead of fixed 4
    - Email-based ordering (no order_inventory)
    """

    # Internet search tool (open search mode only)
    async def search_internet(query: str) -> str:
        """Search the internet for suppliers, products, or market info."""
        result = vending_tools.search_internet(query)
        return json.dumps(result)

    # Email/Supplier tools (same as email mode)
    async def search_suppliers(query: str = "") -> str:
        """Search for wholesale suppliers."""
        result = vending_tools.search_suppliers(query)
        return json.dumps(result)

    async def send_supplier_email(to: str, subject: str, body: str) -> str:
        """Send email to a supplier."""
        result = vending_tools.send_supplier_email(to, subject, body)
        return json.dumps(result)

    async def list_supplier_emails(unread_only: bool = False) -> str:
        """List emails in inbox from suppliers."""
        result = vending_tools.list_supplier_emails(unread_only)
        return json.dumps(result)

    async def read_supplier_email(email_id: int) -> str:
        """Read a specific email from a supplier."""
        result = vending_tools.read_supplier_email(email_id)
        return json.dumps(result)

    async def send_payment(to: str, amount: float, products: str, description: str = "") -> str:
        """Send payment to supplier to place order."""
        try:
            products_dict = json.loads(products)
        except (json.JSONDecodeError, TypeError):
            return json.dumps({"success": False, "error": f"Invalid products format. Expected JSON dict."})
        result = vending_tools.send_payment(to, amount, products_dict, description)
        return json.dumps(result)

    # Standard tools (same as email mode but without order_inventory)
    async def check_balance() -> str:
        result = vending_tools.check_balance()
        return json.dumps(result)

    async def check_storage_inventory() -> str:
        result = vending_tools.check_storage_inventory()
        return json.dumps(result)

    async def check_pending_orders() -> str:
        result = vending_tools.check_pending_orders()
        return json.dumps(result)

    async def get_machine_inventory() -> str:
        result = vending_tools.get_machine_inventory()
        return json.dumps(result)

    async def stock_machine(product: str, quantity: int) -> str:
        result = vending_tools.stock_machine(product, quantity)
        return json.dumps(result)

    async def set_price(product: str, price: float) -> str:
        result = vending_tools.set_price(product, price)
        return json.dumps(result)

    async def research_market(query: str) -> str:
        result = vending_tools.research_product(query)
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

    async def scratchpad_delete(key: str) -> str:
        result = vending_tools.scratchpad_delete(key)
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

    async def kv_store_delete(key: str) -> str:
        result = vending_tools.kv_store_delete(key)
        return json.dumps(result)

    async def collect_cash() -> str:
        result = vending_tools.collect_cash()
        return json.dumps(result)

    async def get_prices() -> str:
        result = vending_tools.get_prices()
        return json.dumps(result)

    return [
        # INTERNET SEARCH (OPEN SEARCH MODE ONLY)
        ToolDef(tool=search_internet, name="search_internet",
                description="Search the internet for vending suppliers, products, or market info. Returns suppliers you can contact via email.",
                parameters={"query": "Search query (e.g., 'vending suppliers san francisco', 'energy drink wholesale')"}),
        # EMAIL/SUPPLIER TOOLS
        ToolDef(tool=search_suppliers, name="search_suppliers",
                description="Search for wholesale suppliers. Use this if search_internet doesn't give enough options.",
                parameters={"query": "(Optional) Search query"}),
        ToolDef(tool=send_supplier_email, name="send_supplier_email",
                description="Send email to a supplier. Ask about their products and prices!",
                parameters={"to": "Supplier email", "subject": "Email subject", "body": "Your message"}),
        ToolDef(tool=list_supplier_emails, name="list_supplier_emails",
                description="List emails in your inbox from suppliers.",
                parameters={"unread_only": "(Optional) Only show unread"}),
        ToolDef(tool=read_supplier_email, name="read_supplier_email",
                description="Read a specific email from a supplier.",
                parameters={"email_id": "Email ID (integer)"}),
        ToolDef(tool=send_payment, name="send_payment",
                description="Send payment to supplier after negotiating via email. Places your order.",
                parameters={"to": "Supplier email", "amount": "Total payment",
                           "products": "JSON dict of products e.g. {\"coca_cola_12oz\": 50}",
                           "description": "(Optional) Order notes"}),
        # STANDARD TOOLS
        ToolDef(tool=check_balance, name="check_balance",
                description="Get current cash balance."),
        ToolDef(tool=check_storage_inventory, name="check_storage_inventory",
                description="Check inventory in storage warehouse."),
        ToolDef(tool=check_pending_orders, name="check_pending_orders",
                description="Check status of orders in transit."),
        ToolDef(tool=get_machine_inventory, name="get_machine_inventory",
                description="Get inventory in vending machine."),
        ToolDef(tool=stock_machine, name="stock_machine",
                description="Move items from storage to vending machine. TIP: Stock 5-10 units per call instead of repeated 1-unit calls to reduce tool costs.",
                parameters={"product": "Product ID", "quantity": "Units to move (recommend 5-10 per call)"}),
        ToolDef(tool=set_price, name="set_price",
                description="Set retail price for a product.",
                parameters={"product": "Product ID", "price": "Price in dollars"}),
        ToolDef(tool=research_market, name="research_market",
                description="Research market information (use search_internet for more detail).",
                parameters={"query": "Search query"}),
        ToolDef(tool=wait_for_next_day, name="wait_for_next_day",
                description="End current day and advance to next. Overnight sales processed, supplier emails arrive."),
        ToolDef(tool=scratchpad_write, name="scratchpad_write",
                description="Write notes for future reference.",
                parameters={"key": "Note name", "content": "Content"}),
        ToolDef(tool=scratchpad_read, name="scratchpad_read",
                description="Read a note.",
                parameters={"key": "Note name"}),
        ToolDef(tool=scratchpad_list, name="scratchpad_list",
                description="List all notes."),
        ToolDef(tool=scratchpad_delete, name="scratchpad_delete",
                description="Delete a note.",
                parameters={"key": "Note name"}),
        ToolDef(tool=kv_store_write, name="kv_store_write",
                description="Store structured data.",
                parameters={"key": "Data key", "value": "JSON value"}),
        ToolDef(tool=kv_store_read, name="kv_store_read",
                description="Read structured data.",
                parameters={"key": "Key to read"}),
        ToolDef(tool=kv_store_list, name="kv_store_list",
                description="List all stored keys."),
        ToolDef(tool=kv_store_delete, name="kv_store_delete",
                description="Delete stored data.",
                parameters={"key": "Key to delete"}),
        ToolDef(tool=collect_cash, name="collect_cash",
                description="Collect revenue from vending machine."),
        ToolDef(tool=get_prices, name="get_prices",
                description="Get current retail prices."),
    ]


def create_email_mode_tools(vending_tools: VendingTools) -> List[ToolDef]:
    """
    Create tools for EMAIL MODE (VendingBench 2 supplier negotiation).

    In this mode:
    - order_inventory is NOT available (must use email negotiation)
    - Agent uses search_suppliers, send_supplier_email, send_payment
    """

    # Email/Supplier tools
    async def search_suppliers(query: str = "") -> str:
        """Search for wholesale suppliers."""
        result = vending_tools.search_suppliers(query)
        return json.dumps(result)

    async def send_supplier_email(to: str, subject: str, body: str) -> str:
        """Send email to a supplier."""
        result = vending_tools.send_supplier_email(to, subject, body)
        return json.dumps(result)

    async def list_supplier_emails(unread_only: bool = False) -> str:
        """List emails in inbox from suppliers."""
        result = vending_tools.list_supplier_emails(unread_only)
        return json.dumps(result)

    async def read_supplier_email(email_id: int) -> str:
        """Read a specific email from a supplier."""
        result = vending_tools.read_supplier_email(email_id)
        return json.dumps(result)

    async def send_payment(to: str, amount: float, products: str, description: str = "") -> str:
        """Send payment to supplier to place order."""
        # Parse products JSON string
        try:
            products_dict = json.loads(products)
        except (json.JSONDecodeError, TypeError):
            return json.dumps({"success": False, "error": f"Invalid products format. Expected JSON dict like {{\"coffee\": 50}}"})
        result = vending_tools.send_payment(to, amount, products_dict, description)
        return json.dumps(result)

    # Standard tools (same as direct mode but without order_inventory)
    async def check_balance() -> str:
        result = vending_tools.check_balance()
        return json.dumps(result)

    async def check_storage_inventory() -> str:
        result = vending_tools.check_storage_inventory()
        return json.dumps(result)

    async def check_pending_orders() -> str:
        result = vending_tools.check_pending_orders()
        return json.dumps(result)

    async def get_machine_inventory() -> str:
        result = vending_tools.get_machine_inventory()
        return json.dumps(result)

    async def stock_machine(product: str, quantity: int) -> str:
        result = vending_tools.stock_machine(product, quantity)
        return json.dumps(result)

    async def set_price(product: str, price: float) -> str:
        result = vending_tools.set_price(product, price)
        return json.dumps(result)

    async def research_market(query: str) -> str:
        result = vending_tools.research_product(query)
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

    async def scratchpad_delete(key: str) -> str:
        result = vending_tools.scratchpad_delete(key)
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

    async def kv_store_delete(key: str) -> str:
        result = vending_tools.kv_store_delete(key)
        return json.dumps(result)

    async def collect_cash() -> str:
        result = vending_tools.collect_cash()
        return json.dumps(result)

    async def get_prices() -> str:
        result = vending_tools.get_prices()
        return json.dumps(result)

    return [
        # EMAIL/SUPPLIER TOOLS (EMAIL MODE ONLY)
        ToolDef(tool=search_suppliers, name="search_suppliers",
                description="Search for wholesale suppliers. Returns list of supplier names and emails.",
                parameters={"query": "(Optional) Search query"}),
        ToolDef(tool=send_supplier_email, name="send_supplier_email",
                description="Send email to a supplier to inquire about products/prices or negotiate. Response arrives after wait_for_next_day().",
                parameters={"to": "Supplier email address", "subject": "Email subject", "body": "Your message"}),
        ToolDef(tool=list_supplier_emails, name="list_supplier_emails",
                description="List emails in your inbox from suppliers.",
                parameters={"unread_only": "(Optional) If true, only show unread emails"}),
        ToolDef(tool=read_supplier_email, name="read_supplier_email",
                description="Read a specific email from a supplier.",
                parameters={"email_id": "Email ID to read (integer)"}),
        ToolDef(tool=send_payment, name="send_payment",
                description="Send payment to supplier after negotiating terms via email. This places your order.",
                parameters={"to": "Supplier email address", "amount": "Total payment amount",
                           "products": "JSON dict of products e.g. {\"coffee\": 50, \"chips\": 30}",
                           "description": "(Optional) Order description"}),
        # STANDARD TOOLS (no order_inventory!)
        ToolDef(tool=check_balance, name="check_balance",
                description="Get current cash balance."),
        ToolDef(tool=check_storage_inventory, name="check_storage_inventory",
                description="Check inventory in storage warehouse."),
        ToolDef(tool=check_pending_orders, name="check_pending_orders",
                description="Check status of orders in transit."),
        ToolDef(tool=get_machine_inventory, name="get_machine_inventory",
                description="Get inventory in vending machine (what customers can buy)."),
        ToolDef(tool=stock_machine, name="stock_machine",
                description="Move items from storage to vending machine. TIP: Stock 5-10 units per call instead of repeated 1-unit calls to reduce tool costs.",
                parameters={"product": "Product name", "quantity": "Units to move (recommend 5-10 per call)"}),
        ToolDef(tool=set_price, name="set_price",
                description="Set retail price for a product.",
                parameters={"product": "Product name", "price": "New price in dollars"}),
        ToolDef(tool=research_market, name="research_market",
                description="Research market information.",
                parameters={"query": "Search query"}),
        ToolDef(tool=wait_for_next_day, name="wait_for_next_day",
                description="End current day and advance to next day. Overnight sales processed, supplier emails arrive."),
        ToolDef(tool=scratchpad_write, name="scratchpad_write",
                description="Write notes for future reference.",
                parameters={"key": "Note name", "content": "Content"}),
        ToolDef(tool=scratchpad_read, name="scratchpad_read",
                description="Read a note.",
                parameters={"key": "Note name"}),
        ToolDef(tool=scratchpad_list, name="scratchpad_list",
                description="List all notes."),
        ToolDef(tool=scratchpad_delete, name="scratchpad_delete",
                description="Delete a note from scratchpad.",
                parameters={"key": "Note name to delete"}),
        # KEY-VALUE STORE
        ToolDef(tool=kv_store_write, name="kv_store_write",
                description="Store structured data (numbers, lists, dicts) for tracking.",
                parameters={"key": "Data key", "value": "JSON string of value to store"}),
        ToolDef(tool=kv_store_read, name="kv_store_read",
                description="Read structured data from key-value store.",
                parameters={"key": "Key to read"}),
        ToolDef(tool=kv_store_list, name="kv_store_list",
                description="List all keys in key-value store."),
        ToolDef(tool=kv_store_delete, name="kv_store_delete",
                description="Delete data from key-value store.",
                parameters={"key": "Key to delete"}),
        # ADDITIONAL TOOLS
        ToolDef(tool=collect_cash, name="collect_cash",
                description="Collect and acknowledge revenue from vending machine sales."),
        ToolDef(tool=get_prices, name="get_prices",
                description="Get current retail prices for all products."),
    ]


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
        result = vending_tools.research_product(query)
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
    customer_model: str = "anthropic/claude-sonnet-4-5-20241022",
    email_system_enabled: bool = False,
    open_product_search: bool = False,
    verbose: bool = False
) -> Task:
    """
    Baseline vending machine task without memory.

    Args:
        simulation_days: Number of days to simulate (1, 3, 30, or 365)
        starting_cash: Starting cash balance
        event_complexity: Event complexity level ("simple", "medium", "full")
        customer_model: Model to use for the agent
        email_system_enabled: If True, use VendingBench 2 email-based supplier negotiation
        open_product_search: If True, use expanded product universe with discoverable suppliers
                             (implies email_system_enabled=True)
        verbose: If True, enable debug logging (tool calls, stuck agent hints).
                 Set to False for clean benchmark runs matching Andon Labs setup.

    Returns:
        inspect_ai Task
    """
    config = SimulationConfig(
        simulation_days=simulation_days,
        starting_cash=starting_cash,
        event_complexity=event_complexity,
        max_messages=2000,
        verbose=verbose
    )

    # Create dataset with single sample (the simulation)
    dataset = [
        Sample(
            input=f"Run a {simulation_days}-day vending machine simulation starting with ${starting_cash:.2f}",
            metadata={
                "simulation_days": simulation_days,
                "starting_cash": starting_cash,
                "event_complexity": event_complexity,
                "customer_model": customer_model,
                "email_system_enabled": email_system_enabled,
                "open_product_search": open_product_search
            }
        )
    ]

    # Extract short model name for task name (e.g., "openai/gpt-4o" -> "gpt-4o")
    model_short = customer_model.split("/")[-1] if "/" in customer_model else customer_model
    if open_product_search:
        mode_suffix = "_open"
    elif email_system_enabled:
        mode_suffix = "_email"
    else:
        mode_suffix = ""

    return Task(
        dataset=dataset,
        solver=[baseline_agent(
            config,
            email_system_enabled=email_system_enabled,
            open_product_search=open_product_search
        )],
        scorer=[profit_scorer(), survival_scorer()],
        name=f"vending_{model_short}_{simulation_days}d{mode_suffix}",
        model=customer_model  # Pass the model to inspect_ai
    )


@solver
def baseline_agent(
    config: SimulationConfig,
    email_system_enabled: bool = False,
    open_product_search: bool = False
) -> Solver:
    """
    Baseline agent using inspect_ai's native model abstraction.

    Uses get_model().generate() and execute_tools() for multi-provider support.

    Args:
        config: Simulation configuration
        email_system_enabled: If True, use email-based supplier negotiation
        open_product_search: If True, use expanded product universe with discoverable suppliers
    """
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Debug: Check state type
        print(f"[DEBUG] state type: {type(state)}", flush=True)
        print(f"[DEBUG] state dir: {dir(state)[:10]}", flush=True)

        # Initialize simulation with flags
        env = VendingEnvironment(
            config,
            email_system_enabled=email_system_enabled,
            open_product_search=open_product_search
        )
        vending_tools = VendingTools(env, open_product_search=open_product_search)

        # Create inspect_ai tools based on mode - dispatch at high level
        if open_product_search:
            tools = create_open_search_tools(vending_tools)
        elif email_system_enabled:
            tools = create_email_mode_tools(vending_tools)
        else:
            tools = create_vending_tools(vending_tools)

        # Build system prompt based on mode - dispatch at high level
        if open_product_search:
            from src.prompts import build_open_search_system_prompt
            system_prompt = build_open_search_system_prompt(
                starting_cash=config.starting_cash,
                daily_fee=config.daily_fee,
                simulation_days=config.simulation_days
            )
        elif email_system_enabled:
            from src.prompts import build_email_mode_system_prompt
            system_prompt = build_email_mode_system_prompt(
                starting_cash=config.starting_cash,
                daily_fee=config.daily_fee,
                simulation_days=config.simulation_days
            )
        else:
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
        # Mode 3: open_search=True (implies email=True) → 40+ products, 10+ suppliers
        # Mode 2: email=True, open_search=False → 4 products, 4 suppliers
        # Mode 1: email=False, open_search=False → Direct ordering, fixed prices
        if open_product_search:
            mode_str = "OPEN SEARCH MODE (40+ products, 10+ suppliers, email negotiation)"
        elif email_system_enabled:
            mode_str = "EMAIL MODE (4 products, 4 suppliers, email negotiation)"
        else:
            mode_str = "DIRECT MODE (4 products, fixed catalog prices)"
        print(f"\n{'='*60}")
        print(f"VENDING SIMULATION STARTED")
        print(f"  Model: {model.name}")
        print(f"  Mode: {mode_str}")
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
        system_message_text = f"{system_prompt}\n\n{morning_briefing}"

        # Initialize messages - handle both object and dict state types
        initial_message = ChatMessageUser(content=system_message_text)

        # CRITICAL: State must have messages as an attribute, not just a dict key
        # inspect_ai's serialization expects state.messages to be accessible as an attribute
        try:
            state.messages = [initial_message]
        except (AttributeError, TypeError) as e:
            # If we can't set state.messages as an attribute, something is wrong with the state object
            # This should never happen with a proper TaskState object from inspect_ai
            print(f"[CRITICAL ERROR] Cannot set state.messages as attribute: {type(state)}, error: {e}", flush=True)
            # Try dict-style as last resort, but this will likely cause serialization errors later
            if isinstance(state, dict):
                state['messages'] = [initial_message]
            else:
                raise

        # Main agent-driven loop using inspect_ai's native abstractions
        while not env.is_complete:
            # Get messages - handle both object and dict access patterns
            try:
                messages = state.messages
            except (AttributeError, KeyError, TypeError) as e:
                # Try dict access
                try:
                    messages = state['messages']
                except (KeyError, TypeError):
                    # Last resort fallback
                    messages = [initial_message]
                    if config.verbose:
                        print(f"[WARNING] Could not access state.messages (Day {env.current_day}): {type(e).__name__}", flush=True)

            # Apply token-aware context compaction if messages getting large
            # Match Andon Labs VendingBench 2 settings: 69k context window
            if len(messages) > 100:  # Rough heuristic for token count
                # Preserve system prompt (first message) and recent messages (last 61%)
                preserve_count = max(int(len(messages) * 0.61), 20)
                system_msg = messages[0]
                recent_msgs = messages[-preserve_count:]
                messages = [system_msg] + recent_msgs

                # Update state with compacted messages
                # ALWAYS set as attribute - inspect_ai needs state.messages as attribute
                state.messages = messages

            input_messages = messages

            # Generate model response with tools
            output = await model.generate(
                input=input_messages,
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

                # Track output tokens for weekly cost calculation (VendingBench 2: $100/million)
                output_tokens = model_output_record["usage"]["output_tokens"] or 0
                env.add_output_tokens(output_tokens)

            # Capture reasoning content if available (extended thinking)
            if hasattr(output.message, 'reasoning') and output.message.reasoning:
                model_output_record["reasoning"] = output.message.reasoning

            all_model_outputs.append(model_output_record)

            # Add assistant response to messages
            messages.append(output.message)
            # ALWAYS set as attribute - inspect_ai needs state.messages as attribute
            state.messages = messages

            # Check if model made tool calls
            if output.message.tool_calls:
                # Execute tools using inspect_ai's execute_tools
                # execute_tools expects the full message list and finds tool calls in the last assistant message
                execute_result = await execute_tools(messages, tools)
                tool_messages = execute_result.messages
                messages.extend(tool_messages)

                # Update state with new messages
                # ALWAYS set as attribute - inspect_ai needs state.messages as attribute
                state.messages = messages

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

                    # Verbose logging: print each tool call (helps debug stuck agents)
                    if config.verbose and tc.function != "wait_for_next_day":
                        args_str = str(tc.arguments)[:100]  # Truncate long args
                        print(f"    [TOOL] Day {env.current_day}: {tc.function}({args_str})", flush=True)

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

                                        # Get current inventory levels
                                        state = env.get_state()
                                        machine_inv = state.get("machine_inventory", {})
                                        storage_inv = state.get("storage_inventory", {})

                                        # Format inventory compactly (total units)
                                        machine_total = sum(machine_inv.values()) if machine_inv else 0
                                        storage_total = sum(storage_inv.values()) if storage_inv else 0

                                        # Build daily summary with inventory
                                        inv_str = f"Machine: {machine_total}u | Storage: {storage_total}u"
                                        print(f"  Day {new_day}: ${cash:.2f} cash | ${revenue:.2f} revenue | {units} sold | {inv_str} | {len(all_tool_calls)} tools")

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

                                        # Weekly token cost charge (VendingBench 2: $100 per million output tokens)
                                        if isinstance(new_day, int) and new_day % 7 == 0:
                                            token_charge = env.process_weekly_token_charge()

                                        if result.get("is_simulation_complete"):
                                            env.is_complete = True
                                            print(f"  Simulation complete at Day {new_day}", flush=True)

                                        # FIX #6: Inject daily morning briefing after wait_for_next_day()
                                        # This provides adaptive warnings and guidance each day
                                        if not env.is_complete:
                                            new_briefing = _build_morning_briefing(env, is_first_day=False)
                                            briefing_message = ChatMessageUser(content=new_briefing)
                                            messages.append(briefing_message)

                                            # Update state with the new briefing message
                                            # ALWAYS set as attribute - inspect_ai needs state.messages as attribute
                                            state.messages = messages

                                            # Print briefing to debug log so user can see adaptive warnings
                                            if config.verbose:
                                                print(new_briefing, flush=True)

                                except (json.JSONDecodeError, TypeError):
                                    pass
                                break  # Found the matching tool message, stop searching
            else:
                # No tool calls - model might be done or need prompting
                if not env.is_complete:
                    # Add continuation prompt
                    continuation_msg = ChatMessageUser(
                        content="Continue managing your vending machine business. Use your tools to check inventory, stock the machine, and advance to the next day with wait_for_next_day()."
                    )
                    messages.append(continuation_msg)

                    # Update state - ALWAYS set as attribute
                    state.messages = messages

            # Check for bankruptcy
            if env.is_complete and env.consecutive_bankrupt_days >= env.bankruptcy_threshold:
                print(f"⚠️  BANKRUPT! Could not pay daily fee for {env.consecutive_bankrupt_days} consecutive days.")
                break

            # Detect stuck agent (no progress after 10+ days) and give explicit hint
            # NOTE: Only enabled in debug/development mode. Disable for benchmarking!
            debug_hints_enabled = config.verbose  # Use verbose flag as debug mode indicator
            if debug_hints_enabled and env.current_day >= 10 and env.cash_balance < config.starting_cash and env.current_day % 5 == 0:
                # Check if agent has made NO revenue in the last 10 days
                if all(metrics.get("total_revenue", 0) == 0 for metrics in env.daily_reports[-min(10, len(env.daily_reports)):]):
                    # Count recent tool diversity (not just wait_for_next_day)
                    recent_tools = [tc["tool"] for tc in all_tool_calls[-20:]]
                    unique_recent_tools = set(recent_tools) - {"wait_for_next_day", "check_balance"}

                    if len(unique_recent_tools) < 2:
                        # Agent is stuck in a loop! Give explicit help
                        if env.open_product_search:
                            hint_msg = """
⚠️ SYSTEM NOTICE: You've made no revenue for 10+ days and are losing money.

IMMEDIATE ACTION NEEDED:
1. Call search_internet("vending machine suppliers") to find suppliers
2. Read the results and identify 2-3 supplier email addresses
3. Call send_supplier_email() to contact each supplier asking about their products/prices
4. Call wait_for_next_day() to get their responses
5. Check your inbox with list_supplier_emails() and read responses
6. Negotiate if needed, then use send_payment() to place your first order

You're bleeding $2/day in fees. Take action NOW or you'll go bankrupt!
"""
                        elif env.email_system_enabled:
                            hint_msg = """
⚠️ SYSTEM NOTICE: You've made no revenue for 10+ days and are losing money.

IMMEDIATE ACTION NEEDED:
1. Call search_suppliers() to find wholesale suppliers
2. Call send_supplier_email() to contact suppliers about products/prices
3. Call wait_for_next_day() to get responses
4. Check inbox with list_supplier_emails() and read responses
5. Use send_payment() to place an order
6. Stock machine when delivery arrives

You're bleeding $2/day in fees. Take action NOW or you'll go bankrupt!
"""
                        else:
                            hint_msg = """
⚠️ SYSTEM NOTICE: You've made no revenue for 10+ days and are losing money.

IMMEDIATE ACTION NEEDED:
1. Call order_inventory() to buy products
2. Call stock_machine() to stock the vending machine
3. Call set_price() to set competitive prices
4. Call wait_for_next_day() to process sales

You're bleeding $2/day in fees. Take action NOW or you'll go bankrupt!
"""
                        hint_message = ChatMessageUser(content=hint_msg)
                        messages.append(hint_message)

                        # Update state - ALWAYS set as attribute
                        state.messages = messages

                        print(f"  [SYSTEM HINT] Injected stuck agent help at Day {env.current_day}", flush=True)

            # Safety check: prevent infinite loops
            if len(all_tool_calls) > 2000:
                print("[SYSTEM] Maximum tool calls reached. Ending simulation.")
                break

            # Safety check: prevent infinite loops when model makes no tool calls
            if len(all_model_outputs) > 3000:
                print("[SYSTEM] Maximum model calls reached. Ending simulation.")
                break

            # Safety check: detect stuck agent (no tool calls in last N model outputs)
            if len(all_model_outputs) > 50:
                recent_outputs = all_model_outputs[-50:]
                recent_tool_calls = sum(1 for o in recent_outputs if o.get("tool_calls"))
                if recent_tool_calls == 0:
                    print("[SYSTEM] Agent stuck: no tool calls in last 50 model outputs. Ending simulation.")
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
        # Calculate total from components (total_tokens from API can be unreliable)
        calculated_total = total_usage['input_tokens'] + total_usage['output_tokens'] + total_usage['reasoning_tokens']
        print(f"    Total:  {calculated_total:,}")
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
        # Handle both object and dict state types
        simulation_results = {
            "final_metrics": metrics,
            "tool_calls": all_tool_calls,
            "model_outputs": all_model_outputs,  # Full model outputs with usage/reasoning
            "total_usage": total_usage,  # Aggregated token usage
            "memory_stats": memory_stats,
            "agent_type": "baseline",
            "model_name": model.name,
            "email_system_enabled": email_system_enabled
        }

        try:
            state.metadata["simulation_results"] = simulation_results
        except (AttributeError, TypeError):
            # If state is a dict, use dict-style access
            if isinstance(state, dict):
                if "metadata" not in state:
                    state["metadata"] = {}
                state["metadata"]["simulation_results"] = simulation_results

        # Add completion message
        completion_msg = ChatMessageAssistant(
            content=f"Simulation Complete. Final Net Worth: ${metrics['final_net_worth']:.2f}"
        )

        # Append completion message
        # ALWAYS use attribute access - inspect_ai needs state.messages as attribute for serialization
        state.messages.append(completion_msg)

        return state

    return solve


def _build_morning_briefing(env: VendingEnvironment, is_first_day: bool = False) -> str:
    """Build the morning briefing message for the agent."""
    state = env.get_state()

    # Format inventory/price lists with fallback for empty dicts
    def format_inventory(items_dict):
        if not items_dict:
            return "  (empty)"
        # Filter out products with 0 units to avoid confusion
        items_with_stock = {p: q for p, q in items_dict.items() if q > 0}
        if not items_with_stock:
            return "  (empty)"
        return chr(10).join(f'  - {product}: {qty} units' for product, qty in items_with_stock.items())

    def format_prices(prices_dict):
        if not prices_dict:
            return "  (no prices set yet)"
        return chr(10).join(f'  - {product}: ${price:.2f}' for product, price in prices_dict.items())

    if is_first_day:
        # Mode-specific Day 0 instructions
        if env.open_product_search:
            workflow_instructions = """IMPORTANT - HOW THIS WORKS (OPEN SEARCH MODE):
1. You start with NOTHING - you must discover suppliers and products
2. Use search_internet() to find suppliers and learn what products exist
3. Email suppliers to ask about their products and wholesale prices
4. Negotiate for better prices, then use send_payment() to place orders
5. Orders arrive in STORAGE in 2-3 days after payment
6. Move items from storage to MACHINE using stock_machine()
7. Set prices with set_price(), then call wait_for_next_day() to process sales
8. Customers buy from your MACHINE overnight (not from storage!)

NEXT STEPS:
- Use search_internet("vending suppliers") to find suppliers
- Email promising suppliers to learn what they offer
- Research what products are profitable for vending machines"""
        elif env.email_system_enabled:
            workflow_instructions = """IMPORTANT - HOW THIS WORKS (EMAIL MODE):
1. You start with some inventory in STORAGE (check below)
2. Use search_suppliers() to find wholesale suppliers
3. Email suppliers to inquire about products and prices
4. Negotiate, then use send_payment() to place orders
5. Orders arrive in STORAGE in 2-3 days after payment
6. Move items from storage to MACHINE using stock_machine()
7. Set prices with set_price(), then call wait_for_next_day() to process sales
8. Customers buy from your MACHINE overnight (not from storage!)

NEXT STEPS:
- Stock your vending machine from storage inventory (if you have any)
- Search for suppliers to restock when you run low"""
        else:
            workflow_instructions = """IMPORTANT - HOW THIS WORKS:
1. You have inventory in STORAGE that needs to be moved to the MACHINE
2. Customers can ONLY buy from the vending machine (not storage!)
3. Use stock_machine() to move items from storage to the vending machine
4. When you're done with today's activities, use wait_for_next_day()
5. Overnight, customers will buy from your machine based on your prices
6. You'll receive a sales report each morning

NEXT STEPS:
- Start by stocking your vending machine!"""

        intro = f"""
════════════════════════════════════════════════════════════════════════════════
WELCOME TO YOUR VENDING MACHINE BUSINESS!
════════════════════════════════════════════════════════════════════════════════

You are starting Day {state['day']} with ${state['cash_balance']:.2f} in cash.

YOUR GOAL: Maximize your bank account balance over {env.config.simulation_days} days.

{workflow_instructions}

STORAGE INVENTORY (what you have in your warehouse):
{format_inventory(state['storage_inventory'])}

MACHINE INVENTORY (what customers can buy from the vending machine):
{format_inventory(state['machine_inventory'])}

CURRENT PRICES (what customers pay):
{format_prices(state['prices'])}

DAILY OPERATING FEE: ${env.config.daily_fee:.2f} (charged each night)

What would you like to do?
"""
    else:
        # Daily briefing with adaptive strategic hints
        hints = []

        # 1. Check for empty inventory
        if not state['machine_inventory'] and not state['storage_inventory']:
            if env.open_product_search:
                hints.append("💡 TIP: Your machine and storage are empty! Use search_internet() to find suppliers.")
            elif env.email_system_enabled:
                hints.append("💡 TIP: Your machine and storage are empty! Use search_suppliers() to find suppliers.")

        # 2. Check for dead capital (inventory in storage, not machine)
        storage_total = sum(state['storage_inventory'].values()) if state['storage_inventory'] else 0
        machine_total = sum(state['machine_inventory'].values()) if state['machine_inventory'] else 0

        if storage_total > 0 and machine_total == 0:
            hints.append("⚠️ WARNING: You have inventory in storage but machine is EMPTY! Use stock_machine() to stock it!")
        elif storage_total > 50 and machine_total < 10:
            capacity_remaining = 12 - machine_total
            hints.append(f"⚠️ INVENTORY ALERT: Machine has {machine_total}/12 slots filled, but storage has {storage_total}u available!")
            hints.append(f"ACTION: Stock {capacity_remaining} more units to fill machine to capacity for maximum sales.")
        elif machine_total < 3 and storage_total > 0:
            hints.append("💡 TIP: Your machine is low on inventory. Consider restocking from storage.")

        # 3. Check for too many product varieties (choice multiplier penalty)
        num_products = len([p for p, q in state['machine_inventory'].items() if q > 0]) if state['machine_inventory'] else 0
        if num_products >= 5:
            hints.append(f"⚠️ CRITICAL: You have {num_products} different products in the machine!")
            hints.append(f"ACTION REQUIRED: Reduce to 3-4 products. Use unstock_machine() to remove lowest sellers.")
            hints.append(f"Consumer psychology research: Too many choices overwhelm customers, reducing purchases.")
            hints.append(f"💡 STRATEGY: Try 3 proven bestsellers + 1 experimental product to test new items while keeping revenue stable.")

        # 4. Check for cash flow issues
        if state['cash_balance'] < 100 and env.current_day > 10:
            hints.append("⚠️ CASH FLOW WARNING: Low cash balance. Focus on profitable items and avoid large orders.")

        # 5. Check pricing (if available)
        if state['prices']:
            # Check if prices are set to 0 or very low
            zero_prices = [p for p, price in state['prices'].items() if price <= 0.01]
            if zero_prices:
                hints.append(f"⚠️ PRICING ERROR: {', '.join(zero_prices)} priced at $0! Set competitive prices ($1.50-2.50).")

            # Check if prices are extremely high
            high_prices = [p for p, price in state['prices'].items() if price >= 4.00]
            if high_prices:
                hints.append(f"💡 TIP: {', '.join(high_prices)} priced very high (${state['prices'][high_prices[0]]:.2f}+). High prices may reduce sales.")

        hint_section = "\n" + "\n".join(hints) + "\n" if hints else ""

        intro = f"""
════════════════════════════════════════════════════════════════════════════════
DAY {state['day']} - MORNING BRIEFING
════════════════════════════════════════════════════════════════════════════════

CURRENT STATUS:
- Cash Balance: ${state['cash_balance']:.2f}
- Days Remaining: {state['days_remaining']}
{hint_section}
MACHINE INVENTORY (what customers can buy):
{format_inventory(state['machine_inventory'])}

STORAGE INVENTORY:
{format_inventory(state['storage_inventory'])}

CURRENT PRICES:
{format_prices(state['prices'])}

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
    Custom solver provides simulation loop control and progress logging.

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

    # Dataset with metadata
    dataset = [
        Sample(
            input=f"Run a {simulation_days}-day vending machine simulation starting with ${starting_cash:.2f}",
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
        solver=[subagent_agent(config, resolved_subagent_model, max_subagent_steps)],
        scorer=[profit_scorer(), survival_scorer()],
        name=f"vending_subagent_{model_short}_{simulation_days}d",
        model=customer_model
    )


@solver
def subagent_agent(
    config: SimulationConfig,
    subagent_model: str,
    max_subagent_steps: int
) -> Solver:
    """
    Custom solver for sub-agent architecture with proper simulation loop control.

    Uses multiagent-inspect for sub-agent tool creation, but maintains our own
    simulation loop for:
    - Checking env.is_complete (proper termination)
    - Progress logging
    - Full output capture
    """
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Initialize simulation
        env = VendingEnvironment(config)
        vending_tools = VendingTools(env)

        # Create tool sets
        direct_tools = create_direct_tools(vending_tools)
        physical_tools = create_physical_tools(vending_tools)

        # Build system prompt for main agent (sub-agent aware)
        system_prompt = build_main_agent_prompt_with_subagent(
            starting_cash=config.starting_cash,
            daily_fee=config.daily_fee,
            simulation_days=config.simulation_days
        )

        # Configure sub-agent for physical world tasks using our custom implementation
        # (Replaces multiagent_inspect which had message trimming bugs)
        # Extract the underlying tool functions from our ToolDef objects
        physical_tool_functions = [td.tool for td in physical_tools]

        # Build sub-agent system prompt
        subagent_system_prompt = build_subagent_system_prompt()

        physical_subagent_config = SubAgentConfig(
            tools=physical_tool_functions,
            model=subagent_model,
            max_steps=max_subagent_steps,
            system_prompt=subagent_system_prompt,
            description="Physical world assistant for vending machine tasks (stock machine, set prices, check inventory, collect cash)"
        )

        # Create the run_sub_agent tool using our custom implementation
        subagent_tool = create_subagent_tool(physical_subagent_config, debug=True)

        # Combine direct tools with sub-agent tool
        all_tools = direct_tools + [subagent_tool]

        # Track all tool calls and model outputs for logging
        all_tool_calls = []
        all_model_outputs = []
        total_usage = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "total_tokens": 0}
        subagent_call_count = 0

        # Build initial morning briefing
        morning_briefing = _build_morning_briefing(env, is_first_day=True)

        # Get the model
        model = get_model()

        # Progress logging - start (flush=True for immediate output)
        print(f"\n{'='*60}", flush=True)
        print(f"VENDING SIMULATION STARTED (Sub-Agent Architecture)", flush=True)
        print(f"  Main Agent: {model.name}", flush=True)
        print(f"  Sub-Agent: {subagent_model}", flush=True)
        print(f"  Days: {config.simulation_days} | Starting Cash: ${config.starting_cash:.2f}", flush=True)
        print(f"{'='*60}", flush=True)

        # Log to inspect transcript
        transcript().info({
            "event": "simulation_start",
            "architecture": "subagent",
            "main_model": model.name,
            "subagent_model": subagent_model,
            "simulation_days": config.simulation_days,
            "starting_cash": config.starting_cash
        })

        # Initial display counters
        display_counter("Day", f"0/{config.simulation_days}")
        display_counter("Cash Balance", f"${config.starting_cash:.2f}")
        display_counter("Cash +/-", "$0.00")
        display_counter("Daily Revenue", "$0.00")
        display_counter("SubAgent Calls", "0")
        display_counter("Total Tools", "0")

        # Initialize conversation with system prompt and morning briefing
        system_message_content = f"{system_prompt}\n\n{morning_briefing}"
        state.messages = [
            ChatMessageUser(content=system_message_content)
        ]

        # Create token-aware compaction handler
        # Match Andon Labs VendingBench 2 settings: 69k context window, 61% preserve
        compact = compaction(
            strategy=CompactionTrim(
                threshold=69000,  # Trigger compaction at 69k tokens (per Andon Labs spec)
                preserve=0.61     # Keep 61% of conversation messages (per Andon Labs spec)
            ),
            prefix=[state.messages[0]],  # Always preserve the system prompt
            tools=all_tools               # Include tools in token count
        )

        # Main agent-driven loop with SIMULATION STATE CHECK
        while not env.is_complete:
            # Apply token-aware context compaction
            input_messages, supplemental = await compact(state.messages)
            if supplemental:
                state.messages.append(supplemental)

            # Generate model response with all tools (direct + sub-agent)
            output = await model.generate(
                input=input_messages,
                tools=all_tools,
            )

            # Capture model output for logging
            model_output_record = {
                "day": env.current_day,
                "message_content": output.message.content if hasattr(output.message, 'content') else None,
                "tool_calls": [{"function": tc.function, "arguments": tc.arguments, "id": tc.id}
                              for tc in (output.message.tool_calls or [])],
                "stop_reason": output.stop_reason if hasattr(output, 'stop_reason') else None,
            }

            # Capture usage statistics
            if hasattr(output, 'usage') and output.usage:
                usage = output.usage
                model_output_record["usage"] = {
                    "input_tokens": getattr(usage, 'input_tokens', 0),
                    "output_tokens": getattr(usage, 'output_tokens', 0),
                    "reasoning_tokens": getattr(usage, 'reasoning_tokens', 0) if hasattr(usage, 'reasoning_tokens') else 0,
                    "total_tokens": getattr(usage, 'total_tokens', 0),
                }
                total_usage["input_tokens"] += model_output_record["usage"]["input_tokens"] or 0
                total_usage["output_tokens"] += model_output_record["usage"]["output_tokens"] or 0
                total_usage["reasoning_tokens"] += model_output_record["usage"]["reasoning_tokens"] or 0
                total_usage["total_tokens"] += model_output_record["usage"]["total_tokens"] or 0

                # Track output tokens for weekly cost calculation (VendingBench 2: $100/million)
                output_tokens = model_output_record["usage"]["output_tokens"] or 0
                env.add_output_tokens(output_tokens)

            if hasattr(output.message, 'reasoning') and output.message.reasoning:
                model_output_record["reasoning"] = output.message.reasoning

            all_model_outputs.append(model_output_record)

            # Add assistant response to messages
            state.messages.append(output.message)

            # Check if model made tool calls
            if output.message.tool_calls:
                # Execute tools
                execute_result = await execute_tools(state.messages, all_tools)
                tool_messages = execute_result.messages
                state.messages.extend(tool_messages)

                # Track tool calls with results
                for i, tc in enumerate(output.message.tool_calls):
                    tool_result = None
                    # Find matching tool result by tool_call_id
                    for tm in tool_messages:
                        if hasattr(tm, 'tool_call_id') and tm.tool_call_id == tc.id:
                            if hasattr(tm, 'content'):
                                try:
                                    tool_result = json.loads(tm.content) if isinstance(tm.content, str) else tm.content
                                except (json.JSONDecodeError, TypeError):
                                    tool_result = tm.content
                            break

                    # Track sub-agent calls
                    is_subagent_call = tc.function in ["run_sub_agent", "chat_with_sub_agent"]
                    if is_subagent_call:
                        subagent_call_count += 1
                        # Log subagent instruction for debugging
                        instruction = tc.arguments.get("instruction", tc.arguments.get("message", ""))
                        if instruction:
                            # Truncate long instructions
                            short_instr = instruction[:80] + "..." if len(instruction) > 80 else instruction
                            print(f"    [SubAgent #{subagent_call_count}] {short_instr}", flush=True)

                    all_tool_calls.append({
                        "day": env.current_day,
                        "tool": tc.function,
                        "input": tc.arguments,
                        "result": tool_result,
                        "tool_call_id": tc.id,
                        "is_subagent": is_subagent_call
                    })

                    # Special handling for wait_for_next_day - progress logging
                    if tc.function == "wait_for_next_day":
                        for tm in tool_messages:
                            if hasattr(tm, 'tool_call_id') and tm.tool_call_id == tc.id and hasattr(tm, 'content'):
                                try:
                                    result = json.loads(tm.content) if isinstance(tm.content, str) else tm.content
                                    if isinstance(result, dict) and "new_day" in result:
                                        sales = result.get("overnight_sales", {})
                                        new_day = result.get("new_day", "?")
                                        cash = result.get("cash_balance", 0)
                                        revenue = sales.get("total_revenue", 0)
                                        units = sales.get("total_units_sold", 0)

                                        # Get current inventory from environment
                                        machine_inv = env.machine_inventory
                                        storage_inv = env.storage_inventory

                                        # Helper to get quantity from storage (handles InventoryItem objects)
                                        def get_storage_qty(product):
                                            items = storage_inv.get(product, [])
                                            if isinstance(items, list):
                                                return sum(item.quantity if hasattr(item, 'quantity') else 0 for item in items)
                                            return items if isinstance(items, int) else 0

                                        # Count small (chips, chocolate) vs large (coffee, soda) items
                                        small_machine = machine_inv.get("chips", 0) + machine_inv.get("chocolate", 0)
                                        large_machine = machine_inv.get("coffee", 0) + machine_inv.get("soda", 0)
                                        small_storage = get_storage_qty("chips") + get_storage_qty("chocolate")
                                        large_storage = get_storage_qty("coffee") + get_storage_qty("soda")

                                        # Check orders placed yesterday (day before we slept)
                                        # Note: new_day is the day we just woke up to, orders were placed on new_day - 1
                                        prev_day = new_day - 1 if isinstance(new_day, int) else env.current_day - 1
                                        yesterdays_orders = [tc for tc in all_tool_calls
                                                            if tc.get("tool") == "order_inventory" and tc.get("day") == prev_day]
                                        order_str = ""
                                        if yesterdays_orders:
                                            order_details = []
                                            for order in yesterdays_orders:
                                                inp = order.get("input", {})
                                                product = inp.get("product", "?")
                                                qty = inp.get("quantity", 0)
                                                order_details.append(f"{qty} {product}")
                                            order_str = f" | Ordered yesterday: {', '.join(order_details)}"

                                        print(f"  Day {new_day}: ${cash:.2f} cash | ${revenue:.2f} rev | {units} sold | {subagent_call_count} subagent{order_str}", flush=True)
                                        print(f"           Machine: {small_machine} small, {large_machine} large | Storage: {small_storage} small, {large_storage} large", flush=True)

                                        # Update display counters
                                        if isinstance(new_day, int):
                                            cash_change = cash - config.starting_cash
                                            cash_change_str = f"+${cash_change:.2f}" if cash_change >= 0 else f"-${abs(cash_change):.2f}"
                                            display_counter("Day", f"{new_day}/{config.simulation_days}")
                                            display_counter("Cash Balance", f"${cash:.2f}")
                                            display_counter("Cash +/-", cash_change_str)
                                            display_counter("Daily Revenue", f"${revenue:.2f}")
                                            display_counter("SubAgent Calls", str(subagent_call_count))
                                            display_counter("Total Tools", str(len(all_tool_calls)))

                                            # Weekly token cost charge (VendingBench 2: $100 per million output tokens)
                                            if new_day % 7 == 0:
                                                token_charge = env.process_weekly_token_charge()

                                        # Log to transcript
                                        transcript().info({
                                            "event": "day_complete",
                                            "day": new_day,
                                            "cash_balance": cash,
                                            "revenue": revenue,
                                            "units_sold": units,
                                            "subagent_calls": subagent_call_count,
                                            "total_tool_calls": len(all_tool_calls)
                                        })

                                        if result.get("is_simulation_complete"):
                                            env.is_complete = True
                                            print(f"  Simulation complete at Day {new_day}", flush=True)
                                except (json.JSONDecodeError, TypeError):
                                    pass
                                break
            else:
                # No tool calls - prompt continuation
                if not env.is_complete:
                    state.messages.append(ChatMessageUser(
                        content="Continue managing your vending machine business. Use your direct tools or ask the sub-agent for physical tasks. Remember to call wait_for_next_day() when you're done with today's activities."
                    ))

            # Check for bankruptcy
            if env.is_complete and env.consecutive_bankrupt_days >= env.bankruptcy_threshold:
                print(f"⚠️  BANKRUPT! Could not pay daily fee for {env.consecutive_bankrupt_days} consecutive days.", flush=True)
                break

            # Safety check: prevent infinite loops
            if len(all_tool_calls) > 3000:
                print("[SYSTEM] Maximum tool calls reached. Ending simulation.", flush=True)
                break

            # Safety check: prevent infinite loops when model makes no tool calls
            if len(all_model_outputs) > 4000:
                print("[SYSTEM] Maximum model calls reached. Ending simulation.", flush=True)
                break

            # Safety check: detect stuck agent (no tool calls in last N model outputs)
            if len(all_model_outputs) > 50:
                recent_outputs = all_model_outputs[-50:]
                recent_tool_calls = sum(1 for o in recent_outputs if o.get("tool_calls"))
                if recent_tool_calls == 0:
                    print("[SYSTEM] Agent stuck: no tool calls in last 50 model outputs. Ending simulation.", flush=True)
                    break

        # Calculate final metrics
        metrics = env.calculate_final_metrics()
        memory_stats = vending_tools.get_memory_stats()

        # Progress logging - final summary (flush=True for immediate output)
        print(f"\n{'='*60}", flush=True)
        print(f"SIMULATION COMPLETE (Sub-Agent Architecture)", flush=True)
        print(f"  Final Cash Balance: ${metrics.get('final_cash_balance', 0):.2f}", flush=True)
        print(f"  Final Net Worth: ${metrics['final_net_worth']:.2f}", flush=True)
        print(f"  Profit/Loss: ${metrics['profit_loss']:.2f}", flush=True)
        print(f"  Total Revenue: ${metrics['total_revenue']:.2f}", flush=True)
        print(f"  Days Simulated: {metrics['days_simulated']}", flush=True)
        print(f"  Total Tool Calls: {len(all_tool_calls)}", flush=True)
        print(f"  Sub-Agent Invocations: {subagent_call_count}", flush=True)
        print(f"  Token Usage: {total_usage['total_tokens']:,} total", flush=True)
        print(f"{'='*60}\n", flush=True)

        # Log to transcript
        transcript().info({
            "event": "simulation_complete",
            "architecture": "subagent",
            "final_cash_balance": metrics.get('final_cash_balance', 0),
            "final_net_worth": metrics['final_net_worth'],
            "profit_loss": metrics['profit_loss'],
            "total_revenue": metrics['total_revenue'],
            "days_simulated": metrics['days_simulated'],
            "total_tool_calls": len(all_tool_calls),
            "subagent_calls": subagent_call_count,
            "total_usage": total_usage
        })

        # Store results in state (full output capture)
        state.metadata["simulation_results"] = {
            "final_metrics": metrics,
            "tool_calls": all_tool_calls,
            "model_outputs": all_model_outputs,
            "total_usage": total_usage,
            "memory_stats": memory_stats,
            "agent_type": "subagent",
            "model_name": model.name,
            "subagent_model": subagent_model,
            "subagent_calls": subagent_call_count
        }

        # Add completion message
        completion_msg = ChatMessageAssistant(
            content=f"Simulation Complete. Final Net Worth: ${metrics['final_net_worth']:.2f}"
        )

        # Append completion message
        # ALWAYS use attribute access - inspect_ai needs state.messages as attribute for serialization
        state.messages.append(completion_msg)

        return state

    return solve
