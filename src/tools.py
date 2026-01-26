"""
Agent tools for vending machine business operations.

Based on Vending-Bench 2 paper specification.
"""

from typing import Dict, List, Any, Optional
import uuid

from src.environment import VendingEnvironment, InventoryItem, PendingOrder
from src.products import PRODUCT_CATALOG, get_product_info


# Delivery delay in days (orders take this long to arrive)
DELIVERY_DELAY_DAYS = 3


# Mock research database (Option A: Fast iteration)
MOCK_RESEARCH_DB = {
    "coffee prices": "Industry average: $1.50 supplier cost, $3.00 retail. Premium coffee can reach $4.00.",
    "coffee demand": "Coffee shows stable demand year-round with slight increase in winter months. Relatively price-inelastic.",
    "chocolate prices": "$0.75 supplier cost, $2.00 typical retail. Seasonal peaks during holidays.",
    "chocolate demand": "Chocolate is moderately price-sensitive. Higher demand in winter, lower in summer.",
    "chips prices": "$0.50 supplier cost, $1.50 typical retail. Consistent margins across brands.",
    "chips demand": "Chips show steady demand with moderate price elasticity. Popular impulse purchase.",
    "soda prices": "$0.60 supplier cost, $2.50 typical retail. Higher margins for premium brands.",
    "soda demand": "Soda demand increases significantly in summer months. Moderately price-sensitive.",
    "competitor pricing": "Typical vending machines price items $1.00-$3.50. Location-dependent.",
    "popular snacks": "Top sellers: chips (high volume), chocolate (steady), protein bars (growing trend).",
    "vending margins": "Successful vending operations target 40-60% gross margins. Watch spoilage costs.",
    "inventory best practices": "Restock high-turnover items frequently. Monitor expiration dates. Stock 2-3 days of inventory in machine.",
    "pricing strategy": "Balance margin and volume. Test price changes gradually. Monitor competitor pricing.",
    "seasonal trends": "Hot beverages peak in winter (Nov-Feb). Cold beverages peak in summer (Jun-Aug)."
}


class VendingTools:
    """Tools available to the agent for business operations."""

    def __init__(self, environment: VendingEnvironment, open_product_search: bool = False):
        """
        Initialize tools with reference to simulation environment.

        Args:
            environment: The VendingEnvironment instance
            open_product_search: If True, enable open product search with expanded
                                 product universe and discoverable suppliers
        """
        self.env = environment
        self.open_product_search = open_product_search

        # Memory systems for baseline agent
        # Scratchpad: Free-form notes (key -> text content)
        self.scratchpad: Dict[str, str] = {}
        # Key-Value Store: Structured data (key -> any JSON-serializable value)
        self.kv_store: Dict[str, Any] = {}

    # =========================================================================
    # Financial Tools
    # =========================================================================

    def check_balance(self) -> Dict[str, Any]:
        """
        Get current cash balance.

        Returns:
            Dict with cash balance and net worth estimate
        """
        self.env.message_count += 1

        storage_value = sum(
            sum(item.supplier_cost * item.quantity for item in items)
            for items in self.env.storage_inventory.values()
        )

        return {
            "success": True,
            "cash_balance": self.env.cash_balance,
            "storage_value": storage_value,
            "estimated_net_worth": self.env.cash_balance + storage_value,
            "day": self.env.current_day
        }

    def collect_cash(self) -> Dict[str, Any]:
        """
        Collect revenue from vending machine sales.

        Note: In this simulation, sales are processed automatically each day.
        This tool is mainly for checking/acknowledging revenue.

        Returns:
            Dict with cash collected (or status)
        """
        self.env.message_count += 1

        # Get today's sales from transaction history
        today_sales = [
            t for t in self.env.transaction_history
            if t.day == self.env.current_day and t.transaction_type == "sale"
        ]

        total_collected = sum(t.amount for t in today_sales)

        return {
            "success": True,
            "amount_collected": total_collected,
            "num_transactions": len(today_sales),
            "day": self.env.current_day,
            "message": f"Collected ${total_collected:.2f} from today's sales"
        }

    # =========================================================================
    # Inventory Management Tools
    # =========================================================================

    def check_storage_inventory(self) -> Dict[str, Any]:
        """
        View items in storage (not yet in vending machine).

        Returns:
            Dict with storage inventory details
        """
        self.env.message_count += 1

        inventory_details = {}
        for product, items in self.env.storage_inventory.items():
            total_qty = sum(item.quantity for item in items)
            if total_qty > 0:
                oldest_item = min(items, key=lambda x: x.purchase_date)
                inventory_details[product] = {
                    "quantity": total_qty,
                    "value": sum(item.supplier_cost * item.quantity for item in items),
                    "oldest_purchase_day": oldest_item.purchase_date,
                    "expiration_day": oldest_item.expiration_day
                }

        return {
            "success": True,
            "storage_inventory": inventory_details,
            "day": self.env.current_day
        }

    def order_inventory(self, product: str, quantity: int) -> Dict[str, Any]:
        """
        Order products from supplier (delivery takes 3 days).

        IMPORTANT: Orders are NOT instant! Products will arrive in your storage
        after the delivery delay (currently 3 days). Plan ahead!

        Args:
            product: Product name
            quantity: Number of units to order

        Returns:
            Dict with operation result including expected delivery day
        """
        self.env.message_count += 1

        # Validation
        if product not in PRODUCT_CATALOG:
            return {
                "success": False,
                "error": f"Unknown product: {product}. Available: {list(PRODUCT_CATALOG.keys())}"
            }

        if quantity <= 0:
            return {
                "success": False,
                "error": "Quantity must be positive"
            }

        # Calculate cost
        supplier_cost = PRODUCT_CATALOG[product]["supplier_cost"]
        total_cost = quantity * supplier_cost

        # Check if enough cash
        if total_cost > self.env.cash_balance:
            return {
                "success": False,
                "error": f"Insufficient funds. Need ${total_cost:.2f}, have ${self.env.cash_balance:.2f}"
            }

        # Deduct cash immediately (payment on order)
        old_balance = self.env.cash_balance
        self.env.cash_balance -= total_cost

        # Debug log for order
        print(f"    [ORDER] {quantity} {product} @ ${supplier_cost:.2f} = ${total_cost:.2f} | Cash: ${old_balance:.2f} → ${self.env.cash_balance:.2f}", flush=True)

        # Calculate delivery day
        delivery_day = self.env.current_day + DELIVERY_DELAY_DAYS

        # Create pending order (will be delivered later)
        order_id = str(uuid.uuid4())[:8]
        pending_order = PendingOrder(
            order_id=order_id,
            product=product,
            quantity=quantity,
            supplier_cost=supplier_cost,
            order_day=self.env.current_day,
            delivery_day=delivery_day,
            total_cost=total_cost
        )
        self.env.pending_orders.append(pending_order)

        # Record transaction
        self.env._record_transaction(
            transaction_type="purchase",
            product=product,
            quantity=quantity,
            amount=-total_cost,
            notes=f"Ordered {quantity} units of {product} - delivery on day {delivery_day}"
        )

        return {
            "success": True,
            "order_id": order_id,
            "product": product,
            "quantity": quantity,
            "cost": total_cost,
            "unit_cost": supplier_cost,
            "order_day": self.env.current_day,
            "delivery_day": delivery_day,
            "days_until_delivery": DELIVERY_DELAY_DAYS,
            "new_cash_balance": self.env.cash_balance,
            "message": f"Ordered {quantity} units of {product} for ${total_cost:.2f}. Will arrive on Day {delivery_day} ({DELIVERY_DELAY_DAYS} days)."
        }

    def check_pending_orders(self) -> Dict[str, Any]:
        """
        Check status of orders that are in transit.

        Returns:
            Dict with list of pending orders and their expected delivery dates
        """
        self.env.message_count += 1

        pending = self.env.get_pending_orders()

        return {
            "success": True,
            "pending_orders": pending,
            "num_pending": len(pending),
            "day": self.env.current_day,
            "message": f"You have {len(pending)} order(s) in transit" if pending else "No orders in transit"
        }

    def get_machine_inventory(self) -> Dict[str, Any]:
        """
        View items currently in the vending machine.

        Returns:
            Dict with machine inventory, prices, and slot usage
        """
        self.env.message_count += 1

        slot_status = self.env.get_machine_slot_status()

        return {
            "success": True,
            "machine_inventory": self.env.machine_inventory.copy(),
            "current_prices": self.env.current_prices.copy(),
            "slot_status": slot_status,
            "day": self.env.current_day,
            "note": f"Machine capacity: {slot_status['total_used']}/{slot_status['total_max']} slots used (Small: {slot_status['small_slots_used']}/{slot_status['small_slots_max']}, Large: {slot_status['large_slots_used']}/{slot_status['large_slots_max']})"
        }

    def stock_machine(self, product: str, quantity: int) -> Dict[str, Any]:
        """
        Move items from storage to vending machine.

        NOTE: Machine has limited capacity (12 slots total):
        - 6 slots for small items (chips, chocolate)
        - 6 slots for large items (coffee, soda)
        Each unit takes one slot.

        EFFICIENCY TIP: Stock multiple units per call instead of making repeated
        single-unit calls. For example, use stock_machine(product, 10) once rather
        than calling stock_machine(product, 1) ten times. This reduces tool costs.

        Args:
            product: Product name
            quantity: Number of units to move (recommend 5-10 units per call)

        Returns:
            Dict with operation result
        """
        self.env.message_count += 1

        # Validation - check product exists in appropriate catalog
        if self.open_product_search:
            from src.product_universe import PRODUCT_UNIVERSE
            product_info = self.env._get_product_info(product)
            if not product_info:
                return {
                    "success": False,
                    "error": f"Unknown product: {product}"
                }
        else:
            if product not in PRODUCT_CATALOG:
                return {
                    "success": False,
                    "error": f"Unknown product: {product}. Available: {list(PRODUCT_CATALOG.keys())}"
                }
            product_info = PRODUCT_CATALOG[product]

        if quantity <= 0:
            return {
                "success": False,
                "error": "Quantity must be positive"
            }

        # Check machine slot capacity
        can_stock, reason, max_stockable = self.env.can_stock_product(product, quantity)
        if not can_stock:
            slot_status = self.env.get_machine_slot_status()
            product_size = product_info["size"]
            return {
                "success": False,
                "error": reason,
                "product_size": product_size,
                "max_stockable": max_stockable,
                "slot_status": slot_status,
                "hint": f"Try stocking {max_stockable} units instead, or wait for items to sell"
            }

        # Check storage availability
        storage_items = self.env.storage_inventory.get(product, [])
        total_in_storage = sum(item.quantity for item in storage_items)

        if total_in_storage == 0:
            return {
                "success": False,
                "error": f"No {product} in storage. Order from suppliers first!"
            }

        if total_in_storage < quantity:
            return {
                "success": False,
                "error": f"Insufficient storage inventory. Have {total_in_storage}, requested {quantity}"
            }

        # Move items (FIFO - oldest first)
        remaining = quantity
        items_to_remove = []

        for item in storage_items:
            if remaining <= 0:
                break

            if item.quantity <= remaining:
                # Take entire item
                remaining -= item.quantity
                items_to_remove.append(item)
            else:
                # Take partial
                item.quantity -= remaining
                remaining = 0

        # Remove fully consumed items
        for item in items_to_remove:
            storage_items.remove(item)

        # Add to machine (updates slot usage)
        self.env.add_to_machine(product, quantity)

        slot_status = self.env.get_machine_slot_status()

        # Debug log for stocking operations
        machine_qty = self.env.machine_inventory.get(product, 0)
        print(f"    [STOCK] {quantity} {product} → Machine now has {machine_qty}", flush=True)

        # Build efficiency hint for small quantities
        efficiency_note = ""
        storage_after = sum(item.quantity for item in storage_items)
        if quantity <= 2 and storage_after >= 5:
            efficiency_note = f" TIP: You have {storage_after} units in storage - consider stocking 5-10 units at once to reduce tool calls."

        return {
            "success": True,
            "product": product,
            "quantity": quantity,
            "machine_inventory_after": machine_qty,
            "storage_inventory_after": storage_after,
            "slot_status": slot_status,
            "message": f"Stocked {quantity} units of {product} in machine. Slots: {slot_status['total_used']}/{slot_status['total_max']} used.{efficiency_note}"
        }

    def unstock_machine(self, product: str, quantity: int) -> Dict[str, Any]:
        """
        Remove items from vending machine and return them to storage.

        This tool enables you to actively optimize your product mix by removing
        slow-selling items to make room for better performers. Essential for
        implementing explore-exploit strategies where you test new products and
        quickly replace underperformers.

        Use cases:
        - Remove slow sellers to make room for high-demand products
        - Quickly adjust product mix based on sales data
        - Clear space when testing new products
        - Remove items before they expire

        STRATEGY TIP: When you have 5+ products and warnings about choice overload,
        use unstock_machine() to immediately remove the lowest sellers and replace
        with higher-demand items. Don't wait for natural depletion - be proactive!

        Args:
            product: Product name to remove from machine
            quantity: Number of units to remove and return to storage

        Returns:
            Dict with operation result
        """
        self.env.message_count += 1

        # Debug print when --debug flag is used
        if self.env.config.verbose:
            print(f"    [DEBUG] unstock_machine() called with product='{product}', quantity={quantity} (Day {self.env.current_day})", flush=True)

        # Validation - check product exists in appropriate catalog
        if self.open_product_search:
            from src.product_universe import PRODUCT_UNIVERSE
            product_info = self.env._get_product_info(product)
            if not product_info:
                return {
                    "success": False,
                    "error": f"Unknown product: {product}"
                }
            # Use base_wholesale as the cost when returning to storage
            supplier_cost = product_info.get("base_wholesale", 0.50)
        else:
            if product not in PRODUCT_CATALOG:
                return {
                    "success": False,
                    "error": f"Unknown product: {product}. Available: {list(PRODUCT_CATALOG.keys())}"
                }
            product_info = PRODUCT_CATALOG[product]
            supplier_cost = product_info["supplier_cost"]

        if quantity <= 0:
            return {
                "success": False,
                "error": "Quantity must be positive"
            }

        # Check machine inventory
        machine_qty = self.env.machine_inventory.get(product, 0)
        if machine_qty == 0:
            return {
                "success": False,
                "error": f"No {product} in machine to unstock"
            }

        if machine_qty < quantity:
            return {
                "success": False,
                "error": f"Insufficient machine inventory. Have {machine_qty}, requested {quantity}"
            }

        # Remove from machine (updates slot usage automatically)
        self.env.remove_from_machine(product, quantity)

        # Return items to storage as new InventoryItem
        # Create new inventory item representing returned stock
        returned_item = InventoryItem(
            product=product,
            quantity=quantity,
            purchase_date=self.env.current_day,
            supplier_cost=supplier_cost,
            expiration_day=None  # Returned items don't have new expiration
        )

        # Add to storage inventory
        if product not in self.env.storage_inventory:
            self.env.storage_inventory[product] = []
        self.env.storage_inventory[product].append(returned_item)

        # Get updated counts
        machine_qty_after = self.env.machine_inventory.get(product, 0)
        storage_items = self.env.storage_inventory.get(product, [])
        storage_qty_after = sum(item.quantity for item in storage_items)
        slot_status = self.env.get_machine_slot_status()

        # Debug log
        print(f"    [UNSTOCK] {quantity} {product} → Storage (Machine: {machine_qty} → {machine_qty_after}, Storage: {storage_qty_after})", flush=True)

        return {
            "success": True,
            "product": product,
            "quantity": quantity,
            "machine_inventory_after": machine_qty_after,
            "storage_inventory_after": storage_qty_after,
            "slot_status": slot_status,
            "message": f"Unstocked {quantity} units of {product} from machine to storage. Machine now has {machine_qty_after} units. Freed slots: {slot_status['total_max'] - slot_status['total_used']}/{slot_status['total_max']} available."
        }

    # =========================================================================
    # Pricing Tools
    # =========================================================================

    def set_price(self, product: str, price: float) -> Dict[str, Any]:
        """
        Set retail price for a product.

        Args:
            product: Product name
            price: New retail price

        Returns:
            Dict with operation result
        """
        self.env.message_count += 1

        # Validation - check product exists
        if self.open_product_search:
            product_info = self.env._get_product_info(product)
            if not product_info:
                return {
                    "success": False,
                    "error": f"Unknown product: {product}"
                }
            # In open search, base_wholesale is the supplier cost
            supplier_cost = product_info.get("base_wholesale", 0)
        else:
            if product not in PRODUCT_CATALOG:
                return {
                    "success": False,
                    "error": f"Unknown product: {product}"
                }
            supplier_cost = PRODUCT_CATALOG[product]["supplier_cost"]

        if price <= 0:
            return {
                "success": False,
                "error": "Price must be positive"
            }

        old_price = self.env.current_prices.get(product, 0.0)
        self.env.current_prices[product] = price

        # Calculate margin
        margin = ((price - supplier_cost) / price) * 100 if price > 0 and supplier_cost > 0 else 0

        result = {
            "success": True,
            "product": product,
            "old_price": old_price,
            "new_price": price,
            "message": f"Updated {product} price: ${old_price:.2f} → ${price:.2f}"
        }

        if supplier_cost > 0:
            result["profit_margin"] = f"{margin:.1f}%"
            result["supplier_cost"] = supplier_cost

        return result

    def get_prices(self) -> Dict[str, Any]:
        """
        Get current retail prices for all products.

        Returns:
            Dict with current prices, supplier costs, and margins
        """
        self.env.message_count += 1

        prices = {}
        for product, price in self.env.current_prices.items():
            if self.open_product_search:
                product_info = self.env._get_product_info(product)
                supplier_cost = product_info.get("base_wholesale", 0) if product_info else 0
            else:
                supplier_cost = PRODUCT_CATALOG.get(product, {}).get("supplier_cost", 0)

            margin = ((price - supplier_cost) / price) * 100 if price > 0 and supplier_cost > 0 else 0
            prices[product] = {
                "retail_price": price,
                "supplier_cost": supplier_cost,
                "profit_margin": f"{margin:.1f}%"
            }

        return {
            "success": True,
            "prices": prices,
            "message": "Current prices retrieved"
        }

    # =========================================================================
    # Email/Communication Tools
    # =========================================================================

    def read_email(self, email_id: str) -> Dict[str, Any]:
        """
        Read an email from inbox.

        Args:
            email_id: Email ID to read

        Returns:
            Dict with email content
        """
        self.env.message_count += 1

        for email in self.env.email_inbox:
            if email["id"] == email_id:
                email["read"] = True
                return {
                    "success": True,
                    "email": email
                }

        return {
            "success": False,
            "error": f"Email {email_id} not found"
        }

    def list_emails(self) -> Dict[str, Any]:
        """
        List all emails in inbox.

        Returns:
            Dict with email list
        """
        self.env.message_count += 1

        return {
            "success": True,
            "emails": [
                {
                    "id": email["id"],
                    "from": email["from"],
                    "subject": email["subject"],
                    "date": email["date"],
                    "read": email.get("read", False)
                }
                for email in self.env.email_inbox
            ],
            "unread_count": sum(1 for e in self.env.email_inbox if not e.get("read", False))
        }

    def write_email(self, to: str, subject: str, body: str) -> Dict[str, Any]:
        """
        Send an email to supplier or customer.

        Args:
            to: Recipient email address
            subject: Email subject
            body: Email body

        Returns:
            Dict with operation result
        """
        self.env.message_count += 1

        email = {
            "id": str(uuid.uuid4())[:8],
            "from": "agent@vendingbusiness.com",
            "to": to,
            "subject": subject,
            "body": body,
            "date": self.env.current_day,
            "sent": True
        }

        self.env.email_sent.append(email)

        return {
            "success": True,
            "email_id": email["id"],
            "message": f"Email sent to {to}"
        }

    # =========================================================================
    # Email System Tools (VendingBench 2 Supplier Negotiation)
    # =========================================================================

    def search_suppliers(self, query: str = None) -> Dict[str, Any]:
        """
        Search for wholesale suppliers for vending machine products.

        Use this to discover suppliers you can contact via email.
        Different suppliers have different prices and reliability.

        Args:
            query: Optional search query (e.g., "wholesale snacks", "bulk beverages")

        Returns:
            Dict with list of supplier names and email addresses
        """
        self.env.message_count += 1

        if not self.env.email_system_enabled:
            return {
                "success": False,
                "error": "Email system not enabled. Use order_inventory() for direct ordering."
            }

        # Dispatch based on mode
        if self.open_product_search:
            return self._search_suppliers_open_mode(query)
        else:
            return self._search_suppliers_legacy_mode()

    def _search_suppliers_legacy_mode(self) -> Dict[str, Any]:
        """Legacy search: return all 4 base suppliers."""
        from src.suppliers import list_all_suppliers

        suppliers = list_all_suppliers()

        # Format supplier list for agent
        supplier_list = [
            {
                "name": s.name,
                "email": s.email,
                "min_order": s.min_order_quantity,
                "delivery_days": s.delivery_days
            }
            for s in suppliers
        ]

        return {
            "success": True,
            "suppliers": supplier_list,
            "count": len(supplier_list),
            "message": f"Found {len(supplier_list)} suppliers. Contact them via email to inquire about products and prices."
        }

    def _search_suppliers_open_mode(self, query: str = None) -> Dict[str, Any]:
        """Open mode: search discoverable suppliers based on query."""
        from src.suppliers import search_discoverable_suppliers
        from src.product_universe import is_search_query_allowed

        if not query:
            query = "wholesale vending suppliers"

        # Check guardrails
        if not is_search_query_allowed(query):
            return {
                "success": True,
                "results": [],
                "count": 0,
                "message": "No relevant suppliers found for this query."
            }

        # Search discoverable suppliers
        suppliers = search_discoverable_suppliers(query)

        # Format results as search results (more realistic)
        results = [
            {
                "name": s.name,
                "email": s.email,
                "description": s.description,
                "min_order": s.min_order_quantity,
                "delivery_days": s.delivery_days
            }
            for s in suppliers
        ]

        return {
            "success": True,
            "query": query,
            "results": results,
            "count": len(results),
            "message": f"Found {len(results)} suppliers. Contact them via email to inquire about products and pricing."
        }

    def search_internet(self, query: str) -> Dict[str, Any]:
        """
        Search the internet for suppliers, products, or market information.

        This is only available in open product search mode.

        Args:
            query: Search query (e.g., "vending machine suppliers san francisco")

        Returns:
            Search results with supplier/product information
        """
        self.env.message_count += 1

        if not self.open_product_search:
            return {
                "success": False,
                "error": "Internet search not available. Use search_suppliers() to find suppliers."
            }

        from src.product_universe import is_search_query_allowed
        from src.suppliers import search_discoverable_suppliers

        # Check guardrails
        if not is_search_query_allowed(query):
            return {
                "success": True,
                "query": query,
                "results": [],
                "message": "No relevant results found for this query."
            }

        # Determine if this is a supplier search or product search
        query_lower = query.lower()
        is_supplier_search = any(term in query_lower for term in
            ["supplier", "wholesale", "distributor", "vendor", "order", "buy"])

        if is_supplier_search:
            # Search for suppliers
            suppliers = search_discoverable_suppliers(query)

            results = [
                {
                    "type": "supplier",
                    "title": s.name,
                    "snippet": s.description,
                    "contact": s.email,
                    "details": {
                        "min_order": s.min_order_quantity,
                        "delivery_days": s.delivery_days
                    }
                }
                for s in suppliers
            ]
        else:
            # General product/market research
            from src.product_universe import get_categories_for_search, get_products_by_category

            categories = get_categories_for_search(query)

            # Sample a few products from relevant categories
            results = []
            for category in categories[:3]:  # Top 3 categories
                products = get_products_by_category(category)
                for pid, pinfo in list(products.items())[:2]:  # 2 products per category
                    results.append({
                        "type": "product_info",
                        "title": pinfo["name"],
                        "snippet": f"Category: {category}. Typical retail: ${pinfo['typical_retail']:.2f}. Contact suppliers for wholesale pricing.",
                        "category": category
                    })

        return {
            "success": True,
            "query": query,
            "results": results,
            "count": len(results),
            "message": f"Found {len(results)} results for '{query}'"
        }

    def send_supplier_email(self, to: str, subject: str, body: str) -> Dict[str, Any]:
        """
        Send an email to a supplier to inquire about products or negotiate prices.

        The supplier will respond within 1-2 days (after you call wait_for_next_day).
        Check your inbox with list_supplier_emails() and read_supplier_email().

        Args:
            to: Supplier's email address
            subject: Email subject
            body: Your message to the supplier

        Returns:
            Dict with confirmation and expected response time
        """
        self.env.message_count += 1

        if not self.env.email_system_enabled:
            return {
                "success": False,
                "error": "Email system not enabled. Use order_inventory() for direct ordering."
            }

        result = self.env.queue_outgoing_email(to, subject, body)

        if result["success"]:
            return {
                "success": True,
                "email_id": result["email_id"],
                "to": result["to"],
                "subject": subject,
                "expected_response": f"Day {result['response_expected_day']}",
                "message": f"Email sent to {result['to']}. Expect a response by Day {result['response_expected_day']}. Use wait_for_next_day() to advance time."
            }
        else:
            return result

    def list_supplier_emails(self, unread_only: bool = False) -> Dict[str, Any]:
        """
        List all emails in your supplier inbox.

        Use this to see responses from suppliers after waiting for the next day.

        Args:
            unread_only: If True, only show unread emails

        Returns:
            Dict with list of emails (id, from, subject, day, read status)
        """
        self.env.message_count += 1

        if not self.env.email_system_enabled:
            return {
                "success": False,
                "error": "Email system not enabled."
            }

        emails = self.env.get_supplier_inbox(unread_only)

        return {
            "success": True,
            "emails": emails,
            "total": len(emails),
            "unread": sum(1 for e in emails if not e["read"]),
            "message": f"You have {len(emails)} emails ({sum(1 for e in emails if not e['read'])} unread)"
        }

    def read_supplier_email(self, email_id: int) -> Dict[str, Any]:
        """
        Read a specific email from a supplier.

        Args:
            email_id: The ID of the email to read

        Returns:
            Dict with full email content
        """
        self.env.message_count += 1

        if not self.env.email_system_enabled:
            return {
                "success": False,
                "error": "Email system not enabled."
            }

        # Coerce email_id to int (LLM might pass string from JSON)
        try:
            email_id = int(email_id)
        except (ValueError, TypeError):
            return {
                "success": False,
                "error": f"Invalid email_id: {email_id}. Must be a number."
            }

        email = self.env.read_supplier_email(email_id)

        if email:
            return {
                "success": True,
                "email": email,
                "message": f"Email from {email['from']}"
            }
        else:
            return {
                "success": False,
                "error": f"Email with ID {email_id} not found"
            }

    def send_payment(
        self,
        to: str,
        amount: float,
        products: Dict[str, int],
        description: str = ""
    ) -> Dict[str, Any]:
        """
        Send payment to a supplier to complete an order.

        IMPORTANT: Only send payment after agreeing on terms via email!
        The amount should match what you negotiated with the supplier.

        Args:
            to: Supplier's email address
            amount: Total payment amount
            products: Dict of products to order, e.g., {"coffee": 50, "chips": 30}
            description: Order description (optional)

        Returns:
            Dict with payment confirmation and expected delivery
        """
        self.env.message_count += 1

        if not self.env.email_system_enabled:
            return {
                "success": False,
                "error": "Email system not enabled. Use order_inventory() for direct ordering."
            }

        result = self.env.process_supplier_payment(to, amount, products, description)

        if result["success"]:
            # Log the payment with remaining cash balance
            print(f"    [PAYMENT] ${amount:.2f} to {result['supplier']} for {products} | Delivery: Day {result['expected_delivery_day']} | Cash: ${self.env.cash_balance:.2f}", flush=True)

            return {
                "success": True,
                "order_id": result["order_id"],
                "amount_paid": result["amount_paid"],
                "products": result["products"],
                "supplier": result["supplier"],
                "expected_delivery": f"Day {result['expected_delivery_day']}",
                "new_balance": self.env.cash_balance,
                "message": f"Payment of ${amount:.2f} sent to {result['supplier']}. Order {result['order_id']} will arrive on Day {result['expected_delivery_day']}."
            }
        else:
            return result

    # =========================================================================
    # Time Control Tool (CRITICAL for agent-driven simulation)
    # =========================================================================

    def wait_for_next_day(self) -> Dict[str, Any]:
        """
        End your activities for today and sleep until tomorrow morning.

        IMPORTANT: This is how you advance time in the simulation!

        When you call this tool:
        1. Customers will make purchases from your vending machine overnight
           (based on what's currently stocked and your prices)
        2. The daily operating fee ($2) will be charged
        3. Any expired items in storage will be removed
        4. You'll wake up to a new day with a report of what happened

        Use this tool when you've completed your activities for the day.
        Remember: If your machine is empty, no sales will happen overnight!

        Returns:
            Dict with overnight sales report and morning briefing
        """
        self.env.message_count += 1

        # Process overnight activities and advance to next day
        overnight_result = self.env.process_overnight_and_advance_day()

        # Format the morning briefing
        sales = overnight_result["overnight_sales"]
        spoiled = overnight_result["spoiled_items"]
        deliveries = overnight_result.get("deliveries", [])
        failed_deliveries = overnight_result.get("failed_deliveries", [])

        # Build sales summary
        if sales["total_units_sold"] > 0:
            sales_lines = []
            for product, data in sales["sales_by_product"].items():
                sales_lines.append(
                    f"  - {product.capitalize()}: {data['quantity']} units @ ${data['price']:.2f} = ${data['revenue']:.2f}"
                )
            sales_summary = "\n".join(sales_lines)
        else:
            sales_summary = "  No sales overnight (machine may have been empty)"

        # Build delivery summary
        if deliveries:
            delivery_lines = [
                f"  - {d['product'].capitalize()}: {d['quantity']} units arrived (ordered Day {d['ordered_day']})"
                for d in deliveries
            ]
            delivery_summary = "\n".join(delivery_lines)
        else:
            delivery_summary = "  No deliveries today"

        # Build failed delivery warning (scammer detection)
        failed_delivery_warning = ""
        if failed_deliveries:
            failed_lines = [
                f"  - {d['product'].capitalize()}: {d['quantity']} units (Order {d['order_id']}) - NEVER ARRIVED!"
                for d in failed_deliveries
            ]
            failed_delivery_warning = f"""
🚨 FAILED DELIVERIES 🚨
The following orders were expected but never arrived - you may have been scammed!
{chr(10).join(failed_lines)}
Consider avoiding this supplier in the future.
"""

        # Build spoilage summary
        if spoiled:
            spoilage_lines = [
                f"  - {item['product'].capitalize()}: {item['quantity']} units (lost ${item['cost']:.2f})"
                for item in spoiled
            ]
            spoilage_summary = "\n".join(spoilage_lines)
        else:
            spoilage_summary = "  No items spoiled"

        # Get pending orders
        pending_orders = self.env.get_pending_orders()
        if pending_orders:
            pending_lines = [
                f"  - {o['product'].capitalize()}: {o['quantity']} units arriving Day {o['delivery_day']} ({o['days_until_delivery']} days)"
                for o in pending_orders
            ]
            pending_summary = "\n".join(pending_lines)
        else:
            pending_summary = "  No orders in transit"

        # Get current state for briefing
        state = self.env.get_state()

        # Build email notification (for email mode)
        email_notification = ""
        new_email_count = overnight_result.get("new_emails", 0)
        if new_email_count > 0:
            email_notification = f"""
📧 NEW SUPPLIER EMAILS
You have {new_email_count} new email(s) from suppliers!
Use list_supplier_emails() to see them, then read_supplier_email(id) to read.
"""

        # Build bankruptcy warning if applicable
        bankruptcy_warning = ""
        consecutive_bankrupt = overnight_result.get("consecutive_bankrupt_days", 0)
        if consecutive_bankrupt > 0:
            days_until_termination = self.env.bankruptcy_threshold - consecutive_bankrupt
            bankruptcy_warning = f"""
⚠️  BANKRUPTCY WARNING ⚠️
You could not pay the daily fee! Consecutive bankrupt days: {consecutive_bankrupt}/10
If this continues for {days_until_termination} more days, the simulation will END.
You need to generate revenue to recover!
"""

        # Build inventory warning - CRITICAL for long-term coherence
        inventory_warning = ""
        total_machine = sum(state['machine_inventory'].values())
        total_storage = sum(state['storage_inventory'].values())
        total_pending = sum(o['quantity'] for o in pending_orders) if pending_orders else 0

        if total_machine == 0 and total_storage == 0 and total_pending == 0:
            # CRITICAL: No inventory anywhere and nothing coming
            # Message varies based on email mode vs direct mode
            if self.env.email_system_enabled:
                inventory_warning = """
🚨 CRITICAL INVENTORY ALERT 🚨
You have ZERO inventory in the machine, ZERO in storage, and NO orders in transit!
You CANNOT make any sales until you order more inventory.
ACTION REQUIRED: Contact suppliers via email and send_payment() to place orders!
Remember: Orders take 2-5 days to arrive. Every day without inventory = $0 revenue.
"""
            else:
                inventory_warning = """
🚨 CRITICAL INVENTORY ALERT 🚨
You have ZERO inventory in the machine, ZERO in storage, and NO orders in transit!
You CANNOT make any sales until you order more inventory.
ACTION REQUIRED: Use order_inventory() NOW to order products!
Remember: Orders take 3 days to arrive. Every day without inventory = $0 revenue.
"""
        elif total_machine == 0 and total_storage == 0:
            # No immediate inventory but orders coming
            inventory_warning = f"""
⚠️ INVENTORY WARNING ⚠️
Your machine AND storage are EMPTY! No sales possible today.
Orders in transit: {total_pending} units arriving soon.
Consider ordering more inventory to maintain continuous stock.
"""
        elif total_machine == 0 and total_storage > 0:
            # Machine empty but storage has items
            inventory_warning = f"""
⚠️ MACHINE EMPTY ⚠️
Your vending machine has no items! Customers cannot buy anything.
You have {total_storage} units in storage - use stock_machine() to restock NOW!
"""
        elif total_machine <= 3:
            # Low machine inventory
            inventory_warning = f"""
📦 LOW INVENTORY NOTICE
Machine inventory is low ({total_machine} units). Consider restocking.
Storage has {total_storage} units available.
"""
        elif total_storage == 0 and total_pending == 0 and total_machine <= 10:
            # Storage depleted, machine running low
            inventory_warning = f"""
📦 RESTOCK REMINDER
Storage is empty and no orders in transit.
Machine has {total_machine} units left - consider ordering more inventory soon!
Orders take 3 days to arrive.
"""

        morning_briefing = f"""
Good morning! It's Day {overnight_result['new_day']}.
{bankruptcy_warning}{inventory_warning}{failed_delivery_warning}{email_notification}
OVERNIGHT SALES REPORT:
{sales_summary}
Total Revenue: ${sales['total_revenue']:.2f}

DELIVERIES ARRIVED:
{delivery_summary}

ORDERS IN TRANSIT:
{pending_summary}

SPOILAGE:
{spoilage_summary}

DAILY FEE: -${overnight_result['daily_fee_charged']:.2f}

CURRENT STATUS:
- Cash Balance: ${overnight_result['cash_balance']:.2f}
- Machine Inventory: {', '.join(f'{k}={v}' for k, v in state['machine_inventory'].items())}
- Storage Inventory: {', '.join(f'{k}={v}' for k, v in state['storage_inventory'].items())}
- Days Remaining: {state['days_remaining']}
"""

        result = {
            "success": True,
            "new_day": overnight_result["new_day"],
            "overnight_sales": sales,
            "spoiled_items": spoiled,
            "daily_fee_charged": overnight_result["daily_fee_charged"],
            "cash_balance": overnight_result["cash_balance"],
            "is_simulation_complete": overnight_result["is_complete"],
            "morning_briefing": morning_briefing,
            "message": f"Advanced to Day {overnight_result['new_day']}. {sales['total_units_sold']} items sold for ${sales['total_revenue']:.2f} revenue."
        }

        # Add email count for email mode
        if new_email_count > 0:
            result["new_supplier_emails"] = new_email_count

        # Add supplier LLM call logs to eval trace
        if "supplier_llm_calls" in overnight_result:
            result["supplier_llm_calls"] = overnight_result["supplier_llm_calls"]

        return result

    # =========================================================================
    # Research Tool
    # =========================================================================

    def research_product(self, query: str) -> Dict[str, Any]:
        """
        Research product information (mock implementation).

        Args:
            query: Research query

        Returns:
            Dict with research results
        """
        self.env.message_count += 1

        # Search mock database
        results = []
        query_lower = query.lower()

        for key, info in MOCK_RESEARCH_DB.items():
            # Simple keyword matching
            if any(word in query_lower for word in key.split()):
                results.append({
                    "topic": key,
                    "information": info
                })

        if not results:
            results = [{
                "topic": "general",
                "information": "No specific information found. Consider checking product catalog or supplier communications."
            }]

        return {
            "success": True,
            "query": query,
            "results": results[:3],  # Top 3 results
            "message": f"Found {len(results)} relevant results"
        }

    # =========================================================================
    # Memory Tools (Scratchpad)
    # =========================================================================

    def scratchpad_write(self, key: str, content: str) -> Dict[str, Any]:
        """
        Write a note to your scratchpad for future reference.

        Use this to record observations, plans, insights, or anything
        you want to remember across days. Good for:
        - Noting price experiments and results
        - Recording seasonal patterns you observe
        - Keeping track of strategies that work/don't work
        - Storing reminders for future actions

        Args:
            key: A descriptive name for this note (e.g., "coffee_pricing_experiment")
            content: The text content to store

        Returns:
            Dict with operation result
        """
        self.env.message_count += 1

        if not key or not key.strip():
            return {
                "success": False,
                "error": "Key cannot be empty"
            }

        key = key.strip()
        is_update = key in self.scratchpad
        self.scratchpad[key] = content

        return {
            "success": True,
            "key": key,
            "action": "updated" if is_update else "created",
            "message": f"{'Updated' if is_update else 'Created'} scratchpad note '{key}'"
        }

    def scratchpad_read(self, key: str) -> Dict[str, Any]:
        """
        Read a note from your scratchpad.

        Args:
            key: The key of the note to read

        Returns:
            Dict with note content or error if not found
        """
        self.env.message_count += 1

        if key not in self.scratchpad:
            return {
                "success": False,
                "error": f"No note found with key '{key}'",
                "available_keys": list(self.scratchpad.keys())
            }

        return {
            "success": True,
            "key": key,
            "content": self.scratchpad[key]
        }

    def scratchpad_list(self) -> Dict[str, Any]:
        """
        List all keys in your scratchpad.

        Returns:
            Dict with list of all scratchpad keys
        """
        self.env.message_count += 1

        return {
            "success": True,
            "keys": list(self.scratchpad.keys()),
            "count": len(self.scratchpad),
            "message": f"You have {len(self.scratchpad)} notes in your scratchpad"
        }

    def scratchpad_delete(self, key: str) -> Dict[str, Any]:
        """
        Delete a note from your scratchpad.

        Args:
            key: The key of the note to delete

        Returns:
            Dict with operation result
        """
        self.env.message_count += 1

        if key not in self.scratchpad:
            return {
                "success": False,
                "error": f"No note found with key '{key}'"
            }

        del self.scratchpad[key]
        return {
            "success": True,
            "key": key,
            "message": f"Deleted note '{key}'"
        }

    # =========================================================================
    # Memory Tools (Key-Value Store)
    # =========================================================================

    def kv_store_write(self, key: str, value: Any) -> Dict[str, Any]:
        """
        Store structured data in your key-value store.

        Use this for structured data like numbers, lists, or dictionaries.
        Good for:
        - Storing price history: {"day": 5, "coffee_price": 3.50, "sales": 12}
        - Tracking metrics over time
        - Saving configuration/strategy parameters
        - Recording experiment results in structured format

        Args:
            key: A descriptive key for this data
            value: Any JSON-serializable value (number, string, list, dict)

        Returns:
            Dict with operation result
        """
        self.env.message_count += 1

        if not key or not key.strip():
            return {
                "success": False,
                "error": "Key cannot be empty"
            }

        key = key.strip()
        is_update = key in self.kv_store
        self.kv_store[key] = value

        return {
            "success": True,
            "key": key,
            "action": "updated" if is_update else "created",
            "value_type": type(value).__name__,
            "message": f"{'Updated' if is_update else 'Stored'} data at key '{key}'"
        }

    def kv_store_read(self, key: str) -> Dict[str, Any]:
        """
        Read structured data from your key-value store.

        Args:
            key: The key to read

        Returns:
            Dict with stored value or error if not found
        """
        self.env.message_count += 1

        if key not in self.kv_store:
            return {
                "success": False,
                "error": f"No data found at key '{key}'",
                "available_keys": list(self.kv_store.keys())
            }

        return {
            "success": True,
            "key": key,
            "value": self.kv_store[key],
            "value_type": type(self.kv_store[key]).__name__
        }

    def kv_store_list(self) -> Dict[str, Any]:
        """
        List all keys in your key-value store.

        Returns:
            Dict with list of all keys and their value types
        """
        self.env.message_count += 1

        key_info = {
            key: type(value).__name__
            for key, value in self.kv_store.items()
        }

        return {
            "success": True,
            "keys": key_info,
            "count": len(self.kv_store),
            "message": f"You have {len(self.kv_store)} entries in your key-value store"
        }

    def kv_store_delete(self, key: str) -> Dict[str, Any]:
        """
        Delete data from your key-value store.

        Args:
            key: The key to delete

        Returns:
            Dict with operation result
        """
        self.env.message_count += 1

        if key not in self.kv_store:
            return {
                "success": False,
                "error": f"No data found at key '{key}'"
            }

        del self.kv_store[key]
        return {
            "success": True,
            "key": key,
            "message": f"Deleted data at key '{key}'"
        }

    # =========================================================================
    # Tool Registration for inspect_ai
    # =========================================================================

    def get_tool_list(self) -> List[Dict[str, Any]]:
        """
        Get list of available tools for inspect_ai.

        Returns:
            List of tool definitions
        """
        return [
            # === TIME CONTROL (Most Important!) ===
            {
                "name": "wait_for_next_day",
                "description": "CRITICAL: End your day and sleep until tomorrow. Customers buy from your machine overnight. You'll receive a sales report when you wake up. Use this when you're done with today's activities.",
                "parameters": {}
            },
            # === INVENTORY MANAGEMENT ===
            {
                "name": "check_storage_inventory",
                "description": "View items in your storage facility (not yet in the vending machine). Shows quantity, value, and expiration dates.",
                "parameters": {}
            },
            {
                "name": "get_machine_inventory",
                "description": "View items currently in the vending machine and their prices. Customers can only buy what's in the machine!",
                "parameters": {}
            },
            {
                "name": "stock_machine",
                "description": "Move items from storage to vending machine. IMPORTANT: Items must be in the machine to be sold to customers!",
                "parameters": {
                    "product": "Product name (coffee, chocolate, chips, soda)",
                    "quantity": "Number of units to move from storage to machine"
                }
            },
            {
                "name": "order_inventory",
                "description": "Order products from supplier. IMPORTANT: Delivery takes 3 days! Plan ahead. Products arrive in storage, then use stock_machine to put them in the vending machine.",
                "parameters": {
                    "product": "Product name (coffee, chocolate, chips, soda)",
                    "quantity": "Number of units to order"
                }
            },
            {
                "name": "check_pending_orders",
                "description": "Check status of orders in transit. Shows what you've ordered and when it will arrive.",
                "parameters": {}
            },
            # === PRICING ===
            {
                "name": "set_price",
                "description": "Set retail price for a product. Higher prices = higher margins but lower sales volume. Lower prices = more sales but less profit per item.",
                "parameters": {
                    "product": "Product name",
                    "price": "New retail price in dollars"
                }
            },
            # === FINANCIAL ===
            {
                "name": "check_balance",
                "description": "Get current cash balance and net worth estimate",
                "parameters": {}
            },
            {
                "name": "collect_cash",
                "description": "Collect and acknowledge revenue from vending machine sales",
                "parameters": {}
            },
            # === COMMUNICATION ===
            {
                "name": "list_emails",
                "description": "List all emails in inbox",
                "parameters": {}
            },
            {
                "name": "read_email",
                "description": "Read a specific email",
                "parameters": {
                    "email_id": "Email ID to read"
                }
            },
            {
                "name": "write_email",
                "description": "Send an email to supplier or customer",
                "parameters": {
                    "to": "Recipient email address",
                    "subject": "Email subject",
                    "body": "Email body"
                }
            },
            # === RESEARCH ===
            {
                "name": "research_product",
                "description": "Research product information, pricing, demand, and market trends",
                "parameters": {
                    "query": "Research query"
                }
            },
            # === MEMORY: SCRATCHPAD (Free-form notes) ===
            {
                "name": "scratchpad_write",
                "description": "Write a note to your scratchpad for future reference. Use for observations, plans, strategies, and reminders you want to remember across days.",
                "parameters": {
                    "key": "A descriptive name for this note (e.g., 'coffee_pricing_experiment', 'weekly_strategy')",
                    "content": "The text content to store"
                }
            },
            {
                "name": "scratchpad_read",
                "description": "Read a note from your scratchpad",
                "parameters": {
                    "key": "The key of the note to read"
                }
            },
            {
                "name": "scratchpad_list",
                "description": "List all notes in your scratchpad",
                "parameters": {}
            },
            {
                "name": "scratchpad_delete",
                "description": "Delete a note from your scratchpad",
                "parameters": {
                    "key": "The key of the note to delete"
                }
            },
            # === MEMORY: KEY-VALUE STORE (Structured data) ===
            {
                "name": "kv_store_write",
                "description": "Store structured data (numbers, lists, dicts) in your key-value store. Use for metrics, price history, experiment results, and configuration.",
                "parameters": {
                    "key": "A descriptive key for this data",
                    "value": "Any JSON-serializable value (number, string, list, dict)"
                }
            },
            {
                "name": "kv_store_read",
                "description": "Read structured data from your key-value store",
                "parameters": {
                    "key": "The key to read"
                }
            },
            {
                "name": "kv_store_list",
                "description": "List all keys in your key-value store",
                "parameters": {}
            },
            {
                "name": "kv_store_delete",
                "description": "Delete data from your key-value store",
                "parameters": {
                    "key": "The key to delete"
                }
            }
        ]

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about memory usage for analysis.

        Returns:
            Dict with memory usage statistics
        """
        return {
            "scratchpad": {
                "num_entries": len(self.scratchpad),
                "keys": list(self.scratchpad.keys()),
                "total_chars": sum(len(v) for v in self.scratchpad.values())
            },
            "kv_store": {
                "num_entries": len(self.kv_store),
                "keys": list(self.kv_store.keys()),
                "value_types": {k: type(v).__name__ for k, v in self.kv_store.items()}
            }
        }

    def get_email_mode_tools(self) -> List[Dict[str, Any]]:
        """
        Get tool definitions for EMAIL MODE (VendingBench 2 supplier negotiation).

        In this mode:
        - order_inventory is NOT available
        - Agent must use search_suppliers, send_supplier_email, send_payment

        Returns:
            List of tool definitions for email mode
        """
        return [
            # === TIME CONTROL ===
            {
                "name": "wait_for_next_day",
                "description": "CRITICAL: End your day and sleep until tomorrow. Customers buy from your machine overnight, and supplier emails arrive. Use this when you're done with today's activities.",
                "parameters": {}
            },
            # === SUPPLIER/EMAIL TOOLS (EMAIL MODE ONLY) ===
            {
                "name": "search_suppliers",
                "description": "Search for wholesale suppliers. Returns list of supplier names and email addresses. Contact them via email to get prices.",
                "parameters": {
                    "query": "(Optional) Search query like 'wholesale snacks'"
                }
            },
            {
                "name": "send_supplier_email",
                "description": "Send email to a supplier to inquire about products, prices, or negotiate. Supplier responds within 1-2 days.",
                "parameters": {
                    "to": "Supplier's email address",
                    "subject": "Email subject",
                    "body": "Your message"
                }
            },
            {
                "name": "list_supplier_emails",
                "description": "List emails in your inbox from suppliers. Check after wait_for_next_day() to see responses.",
                "parameters": {
                    "unread_only": "(Optional) If true, only show unread emails"
                }
            },
            {
                "name": "read_supplier_email",
                "description": "Read a specific email from a supplier",
                "parameters": {
                    "email_id": "Email ID to read"
                }
            },
            {
                "name": "send_payment",
                "description": "Send payment to supplier after agreeing on terms via email. This places your order.",
                "parameters": {
                    "to": "Supplier's email address",
                    "amount": "Total payment amount",
                    "products": "Dict of products e.g. {\"coffee\": 50, \"chips\": 30}",
                    "description": "(Optional) Order description"
                }
            },
            # === INVENTORY MANAGEMENT ===
            {
                "name": "check_storage_inventory",
                "description": "View items in storage (not yet in vending machine)",
                "parameters": {}
            },
            {
                "name": "get_machine_inventory",
                "description": "View items in vending machine and their prices",
                "parameters": {}
            },
            {
                "name": "stock_machine",
                "description": "Move items from storage to vending machine",
                "parameters": {
                    "product": "Product name (coffee, chocolate, chips, soda)",
                    "quantity": "Units to move"
                }
            },
            {
                "name": "check_pending_orders",
                "description": "Check status of orders in transit",
                "parameters": {}
            },
            # === PRICING ===
            {
                "name": "set_price",
                "description": "Set retail price for a product",
                "parameters": {
                    "product": "Product name",
                    "price": "New price in dollars"
                }
            },
            # === FINANCIAL ===
            {
                "name": "check_balance",
                "description": "Get current cash balance",
                "parameters": {}
            },
            # === RESEARCH ===
            {
                "name": "research_product",
                "description": "Research product information and market trends",
                "parameters": {
                    "query": "Research query"
                }
            },
            # === MEMORY ===
            {
                "name": "scratchpad_write",
                "description": "Write notes for future reference",
                "parameters": {
                    "key": "Note name",
                    "content": "Content"
                }
            },
            {
                "name": "scratchpad_read",
                "description": "Read a note",
                "parameters": {
                    "key": "Note name"
                }
            },
            {
                "name": "scratchpad_list",
                "description": "List all notes",
                "parameters": {}
            },
            {
                "name": "scratchpad_delete",
                "description": "Delete a note from your scratchpad",
                "parameters": {
                    "key": "Note name to delete"
                }
            },
            # === KEY-VALUE STORE ===
            {
                "name": "kv_store_write",
                "description": "Store structured data (numbers, lists, dicts)",
                "parameters": {
                    "key": "Data key",
                    "value": "Value to store"
                }
            },
            {
                "name": "kv_store_read",
                "description": "Read structured data",
                "parameters": {
                    "key": "Key to read"
                }
            },
            {
                "name": "kv_store_list",
                "description": "List all stored keys",
                "parameters": {}
            },
            {
                "name": "kv_store_delete",
                "description": "Delete stored data",
                "parameters": {
                    "key": "Key to delete"
                }
            },
        ]

    def get_direct_mode_tools(self) -> List[Dict[str, Any]]:
        """
        Get tool definitions for DIRECT MODE (current behavior, no email negotiation).

        In this mode:
        - order_inventory IS available at fixed catalog prices
        - No supplier email tools

        Returns:
            List of tool definitions for direct mode
        """
        # Return the existing tool list (which includes order_inventory)
        return self.get_tool_list()

    def get_open_product_search_tools(self) -> List[Dict[str, Any]]:
        """
        Get tool definitions for OPEN PRODUCT SEARCH mode.

        In this mode:
        - search_internet tool is available for discovering suppliers/products
        - Expanded product universe with 40+ products
        - Discoverable suppliers (10+) instead of fixed 4
        - Email-based ordering (no order_inventory)

        Returns:
            List of tool definitions for open product search mode
        """
        return [
            # === TIME CONTROL ===
            {
                "name": "wait_for_next_day",
                "description": "CRITICAL: End your day and sleep until tomorrow. Customers buy from your machine overnight, and supplier emails arrive. Use this when you're done with today's activities.",
                "parameters": {}
            },
            # === INTERNET SEARCH (OPEN SEARCH MODE ONLY) ===
            {
                "name": "search_internet",
                "description": "Search the internet for vending machine suppliers, products, or market information. Returns relevant suppliers, product info, and market trends.",
                "parameters": {
                    "query": "Search query (e.g., 'vending suppliers san francisco', 'energy drink wholesale')"
                }
            },
            # === SUPPLIER/EMAIL TOOLS ===
            {
                "name": "search_suppliers",
                "description": "Search for wholesale suppliers. Returns list of supplier names and contacts. Use this if search_internet doesn't give you enough supplier options.",
                "parameters": {
                    "query": "(Optional) Search query like 'wholesale snacks'"
                }
            },
            {
                "name": "send_supplier_email",
                "description": "Send email to a supplier to inquire about products, prices, or negotiate. Suppliers offer various products - ask what they have available!",
                "parameters": {
                    "to": "Supplier's email address",
                    "subject": "Email subject",
                    "body": "Your message"
                }
            },
            {
                "name": "list_supplier_emails",
                "description": "List emails in your inbox from suppliers. Check after wait_for_next_day() to see responses.",
                "parameters": {
                    "unread_only": "(Optional) If true, only show unread emails"
                }
            },
            {
                "name": "read_supplier_email",
                "description": "Read a specific email from a supplier",
                "parameters": {
                    "email_id": "Email ID to read"
                }
            },
            {
                "name": "send_payment",
                "description": "Send payment to supplier after agreeing on terms via email. This places your order. Products can include sodas, chips, candy, energy drinks, protein bars, electronics, and more!",
                "parameters": {
                    "to": "Supplier's email address",
                    "amount": "Total payment amount",
                    "products": "Dict of products with quantities (use product IDs from supplier emails)",
                    "description": "(Optional) Order description"
                }
            },
            # === INVENTORY MANAGEMENT ===
            {
                "name": "check_storage_inventory",
                "description": "View items in storage (not yet in vending machine)",
                "parameters": {}
            },
            {
                "name": "get_machine_inventory",
                "description": "View items in vending machine and their prices",
                "parameters": {}
            },
            {
                "name": "stock_machine",
                "description": "Move items from storage to vending machine. Works with any product you've purchased.",
                "parameters": {
                    "product": "Product ID (from your inventory)",
                    "quantity": "Units to move"
                }
            },
            {
                "name": "check_pending_orders",
                "description": "Check status of orders in transit",
                "parameters": {}
            },
            # === PRICING ===
            {
                "name": "set_price",
                "description": "Set retail price for any product in your machine",
                "parameters": {
                    "product": "Product ID",
                    "price": "New price in dollars"
                }
            },
            # === FINANCIAL ===
            {
                "name": "check_balance",
                "description": "Get current cash balance",
                "parameters": {}
            },
            # === RESEARCH ===
            {
                "name": "research_product",
                "description": "Research product information and market trends (basic info - use search_internet for more)",
                "parameters": {
                    "query": "Research query"
                }
            },
            # === MEMORY ===
            {
                "name": "scratchpad_write",
                "description": "Write notes for future reference",
                "parameters": {
                    "key": "Note name",
                    "content": "Content"
                }
            },
            {
                "name": "scratchpad_read",
                "description": "Read a note",
                "parameters": {
                    "key": "Note name"
                }
            },
            {
                "name": "scratchpad_list",
                "description": "List all notes",
                "parameters": {}
            },
            {
                "name": "scratchpad_delete",
                "description": "Delete a note",
                "parameters": {
                    "key": "Note name"
                }
            },
            # === KEY-VALUE STORE ===
            {
                "name": "kv_store_write",
                "description": "Store structured data",
                "parameters": {
                    "key": "Data key",
                    "value": "Value"
                }
            },
            {
                "name": "kv_store_read",
                "description": "Read structured data",
                "parameters": {
                    "key": "Key"
                }
            },
            {
                "name": "kv_store_list",
                "description": "List all keys",
                "parameters": {}
            },
            {
                "name": "kv_store_delete",
                "description": "Delete data",
                "parameters": {
                    "key": "Key"
                }
            },
        ]
