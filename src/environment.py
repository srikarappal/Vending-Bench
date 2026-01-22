"""
Core vending machine business simulation environment.

Manages business state, inventory, finances, and progression.
Supports both direct ordering and email-based supplier negotiation.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import uuid

from src.products import PRODUCT_CATALOG, MACHINE_CONFIG, calculate_demand, get_seasonal_factor
from config.simulation_config import SimulationConfig


@dataclass
class InventoryItem:
    """Represents an inventory item with expiration tracking."""
    product: str
    quantity: int
    purchase_date: int
    supplier_cost: float
    expiration_day: Optional[int] = None

    def is_expired(self, current_day: int) -> bool:
        """Check if item is expired."""
        if self.expiration_day is None:
            return False
        return current_day >= self.expiration_day


@dataclass
class Transaction:
    """Records a business transaction."""
    day: int
    transaction_type: str  # "sale", "purchase", "spoilage", "fee"
    product: Optional[str]
    quantity: int
    amount: float  # Positive = revenue, Negative = cost
    balance_after: float
    notes: str = ""


@dataclass
class PendingOrder:
    """Represents an order that is in transit."""
    order_id: str
    product: str
    quantity: int
    supplier_cost: float
    order_day: int
    delivery_day: int  # Day when order will arrive
    total_cost: float


class VendingEnvironment:
    """
    365-day vending machine business simulation.

    Manages:
    - Cash balance and financial tracking
    - Storage and machine inventory
    - Product pricing
    - Daily operations and fees
    - Transaction history
    """

    def __init__(self, config: SimulationConfig, email_system_enabled: bool = False):
        """Initialize simulation environment.

        Args:
            config: Simulation configuration
            email_system_enabled: If True, use email-based supplier negotiation
        """
        self.config = config
        self.email_system_enabled = email_system_enabled

        # Simulation state
        self.current_day = 0
        self.cash_balance = config.starting_cash
        self.message_count = 0
        self.is_complete = False

        # Inventory (product -> list of InventoryItem)
        self.storage_inventory: Dict[str, List[InventoryItem]] = {
            product: [] for product in PRODUCT_CATALOG.keys()
        }
        self.machine_inventory: Dict[str, int] = {
            product: 0 for product in PRODUCT_CATALOG.keys()
        }

        # Machine slot capacity (paper: 4 rows × 3 slots = 12 total)
        self.machine_small_slots_used = 0
        self.machine_large_slots_used = 0
        self.machine_small_slots_max = MACHINE_CONFIG["small_slots"]  # 6
        self.machine_large_slots_max = MACHINE_CONFIG["large_slots"]  # 6

        # Pricing (product -> price)
        self.current_prices: Dict[str, float] = {
            product: info["typical_retail"]
            for product, info in PRODUCT_CATALOG.items()
        }

        # Tracking
        self.transaction_history: List[Transaction] = []
        self.daily_reports: List[Dict[str, Any]] = []
        self.days_profitable = 0

        # Email/communication system (for direct mode - simple notifications)
        self.email_inbox: List[Dict[str, Any]] = []
        self.email_sent: List[Dict[str, Any]] = []

        # Email system (VendingBench 2 style supplier negotiation)
        if email_system_enabled:
            from src.suppliers import SupplierEmail
            self.supplier_inbox: List[SupplierEmail] = []  # Emails from suppliers
            self.supplier_outbox: List[SupplierEmail] = []  # Pending emails awaiting response
            self.email_conversations: Dict[str, List[SupplierEmail]] = {}  # supplier_id -> email chain
            self.next_email_id = 1

        # Pending orders (orders in transit, not yet delivered)
        self.pending_orders: List[PendingOrder] = []

        # Bankruptcy tracking (paper: terminate after 10 consecutive days of not paying fee)
        self.consecutive_bankrupt_days = 0
        self.bankruptcy_threshold = 10  # Days before termination

        # Token cost tracking (VendingBench 2: $100 per million output tokens, charged weekly)
        self.token_cost_per_million = 100.0  # $100 per million output tokens
        self.accumulated_output_tokens = 0
        self.total_token_costs = 0.0
        self.last_token_charge_day = 0

        # Initialize starter inventory if configured
        if config.starting_inventory_units > 0:
            self._initialize_starter_inventory(config.starting_inventory_units)

        # Calculate and store starting net worth (cash + all inventory at wholesale)
        starting_storage_value = sum(
            sum(item.supplier_cost * item.quantity for item in items)
            for items in self.storage_inventory.values()
        )
        starting_machine_value = sum(
            PRODUCT_CATALOG[product]["supplier_cost"] * quantity
            for product, quantity in self.machine_inventory.items()
        )
        self.starting_net_worth = self.cash_balance + starting_storage_value + starting_machine_value

        # Starting state report
        self._log_daily_report()

    def _initialize_starter_inventory(self, units_per_product: int):
        """
        Initialize storage with starter inventory.

        Args:
            units_per_product: Number of units of each product to start with
        """
        for product, info in PRODUCT_CATALOG.items():
            starter_item = InventoryItem(
                product=product,
                quantity=units_per_product,
                purchase_date=0,
                supplier_cost=info["supplier_cost"],
                expiration_day=info["spoilage_days"]  # Expires after spoilage_days
            )
            self.storage_inventory[product].append(starter_item)

    def process_overnight_and_advance_day(self) -> Dict[str, Any]:
        """
        Process overnight activities and advance to the next day.

        This is the core of the agent-driven simulation loop:
        1. Process customer purchases based on machine inventory
        2. Advance to the next day
        3. Process deliveries (orders that have arrived)
        4. Charge daily operating fee
        5. Process spoilage
        6. Process supplier email responses (if email system enabled)
        7. Generate morning briefing

        Returns:
            Dictionary with overnight sales report and morning briefing
        """
        # 1. Process overnight customer sales
        overnight_sales = self._process_overnight_sales()

        # 2. Advance to next day
        self.current_day += 1

        # 3. Process deliveries (orders arriving today)
        deliveries = self._process_deliveries()

        # 4. Charge daily operating fee and check bankruptcy
        is_bankrupt = self._charge_daily_fee()

        # 5. Check for spoilage
        spoiled = self._process_spoilage()

        # 6. Process supplier email responses (if email system enabled)
        new_emails = []
        if self.email_system_enabled:
            new_emails = self._process_supplier_emails()

        # 7. Generate daily report
        report = self._log_daily_report()

        # 8. Check if simulation is complete (days finished OR bankrupt)
        if self.current_day >= self.config.simulation_days:
            self.is_complete = True
        if is_bankrupt:
            self.is_complete = True

        result = {
            "overnight_sales": overnight_sales,
            "deliveries": deliveries,
            "spoiled_items": spoiled,
            "daily_fee_charged": self.config.daily_fee,
            "new_day": self.current_day,
            "cash_balance": self.cash_balance,
            "is_complete": self.is_complete,
            "is_bankrupt": is_bankrupt,
            "consecutive_bankrupt_days": self.consecutive_bankrupt_days
        }

        if self.email_system_enabled and new_emails:
            result["new_emails"] = len(new_emails)

        return result

    def _process_deliveries(self) -> List[Dict[str, Any]]:
        """
        Process orders that are scheduled for delivery today.

        Returns:
            List of delivered orders
        """
        delivered = []
        remaining_orders = []

        for order in self.pending_orders:
            if order.delivery_day <= self.current_day:
                # Order has arrived - add to storage
                expiration_day = self.current_day + PRODUCT_CATALOG[order.product]["spoilage_days"]
                new_item = InventoryItem(
                    product=order.product,
                    quantity=order.quantity,
                    purchase_date=self.current_day,
                    supplier_cost=order.supplier_cost,
                    expiration_day=expiration_day
                )
                self.storage_inventory[order.product].append(new_item)

                delivered.append({
                    "order_id": order.order_id,
                    "product": order.product,
                    "quantity": order.quantity,
                    "ordered_day": order.order_day,
                    "delivered_day": self.current_day
                })
            else:
                # Order still in transit
                remaining_orders.append(order)

        self.pending_orders = remaining_orders
        return delivered

    def _process_supplier_emails(self) -> List[Dict[str, Any]]:
        """
        Process pending supplier emails and generate responses.

        Called during overnight processing when email system is enabled.

        Returns:
            List of new emails added to inbox
        """
        if not self.email_system_enabled:
            return []

        from src.suppliers import get_supplier_by_email, SupplierEmail, AGENT_EMAIL
        from src.supplier_llm import generate_supplier_response

        new_emails = []
        remaining_outbox = []

        for outbox_email in self.supplier_outbox:
            # Find the supplier
            supplier = get_supplier_by_email(outbox_email.to_addr)
            if not supplier:
                # Unknown supplier - no response
                continue

            # Check if response is due (response_delay_days after sending)
            response_due_day = outbox_email.sent_day + supplier.response_delay_days
            if self.current_day >= response_due_day:
                # Get conversation history for this supplier
                supplier_id = supplier.supplier_id
                history = self.email_conversations.get(supplier_id, [])

                # Generate supplier response using LLM
                try:
                    subject, body = generate_supplier_response(
                        supplier=supplier,
                        agent_email=outbox_email,
                        email_history=history
                    )

                    # Create response email
                    response = SupplierEmail(
                        email_id=self.next_email_id,
                        from_addr=supplier.email,
                        to_addr=AGENT_EMAIL,
                        subject=subject,
                        body=body,
                        sent_day=self.current_day,
                        read=False,
                        replied_to=outbox_email.email_id
                    )
                    self.next_email_id += 1

                    # Add to inbox and conversation history
                    self.supplier_inbox.append(response)
                    if supplier_id not in self.email_conversations:
                        self.email_conversations[supplier_id] = []
                    self.email_conversations[supplier_id].append(outbox_email)
                    self.email_conversations[supplier_id].append(response)

                    new_emails.append({
                        "email_id": response.email_id,
                        "from": supplier.name,
                        "subject": response.subject
                    })

                except Exception as e:
                    # Log error but don't crash simulation
                    print(f"    [EMAIL ERROR] Failed to generate response from {supplier.name}: {e}")
            else:
                # Response not yet due, keep in outbox
                remaining_outbox.append(outbox_email)

        self.supplier_outbox = remaining_outbox
        return new_emails

    def queue_outgoing_email(self, to_addr: str, subject: str, body: str) -> Dict[str, Any]:
        """
        Queue an outgoing email to a supplier.

        Args:
            to_addr: Supplier email address
            subject: Email subject
            body: Email body

        Returns:
            Dict with email details
        """
        if not self.email_system_enabled:
            return {"success": False, "error": "Email system not enabled"}

        from src.suppliers import get_supplier_by_email, SupplierEmail, AGENT_EMAIL

        # Validate supplier exists
        supplier = get_supplier_by_email(to_addr)
        if not supplier:
            return {
                "success": False,
                "error": f"Unknown supplier email: {to_addr}"
            }

        # Create email
        email = SupplierEmail(
            email_id=self.next_email_id,
            from_addr=AGENT_EMAIL,
            to_addr=to_addr,
            subject=subject,
            body=body,
            sent_day=self.current_day,
            read=True  # Agent's own emails are "read"
        )
        self.next_email_id += 1

        # Queue for processing
        self.supplier_outbox.append(email)

        return {
            "success": True,
            "email_id": email.email_id,
            "to": supplier.name,
            "subject": subject,
            "response_expected_day": self.current_day + supplier.response_delay_days
        }

    def get_supplier_inbox(self, unread_only: bool = False) -> List[Dict[str, Any]]:
        """
        Get list of emails from suppliers.

        Args:
            unread_only: If True, only return unread emails

        Returns:
            List of email summaries
        """
        if not self.email_system_enabled:
            return []

        emails = []
        for email in self.supplier_inbox:
            if unread_only and email.read:
                continue
            emails.append({
                "email_id": email.email_id,
                "from": email.from_addr,
                "subject": email.subject,
                "day": email.sent_day,
                "read": email.read
            })

        return emails

    def read_supplier_email(self, email_id: int) -> Optional[Dict[str, Any]]:
        """
        Read a specific email from the supplier inbox.

        Args:
            email_id: Email ID to read

        Returns:
            Full email content or None if not found
        """
        if not self.email_system_enabled:
            return None

        for email in self.supplier_inbox:
            if email.email_id == email_id:
                email.read = True
                return {
                    "email_id": email.email_id,
                    "from": email.from_addr,
                    "to": email.to_addr,
                    "subject": email.subject,
                    "body": email.body,
                    "sent_day": email.sent_day
                }

        return None

    def process_supplier_payment(
        self,
        supplier_email: str,
        amount: float,
        products: Dict[str, int],
        description: str = ""
    ) -> Dict[str, Any]:
        """
        Process payment to supplier and create order.

        Args:
            supplier_email: Supplier's email address
            amount: Payment amount
            products: Dict of product -> quantity
            description: Payment description

        Returns:
            Dict with payment/order details
        """
        if not self.email_system_enabled:
            return {"success": False, "error": "Email system not enabled"}

        from src.suppliers import get_supplier_by_email
        import random

        # Validate supplier
        supplier = get_supplier_by_email(supplier_email)
        if not supplier:
            return {"success": False, "error": f"Unknown supplier: {supplier_email}"}

        # Check balance
        if self.cash_balance < amount:
            return {
                "success": False,
                "error": f"Insufficient funds. Balance: ${self.cash_balance:.2f}, Required: ${amount:.2f}"
            }

        # Deduct payment
        self.cash_balance -= amount
        self._record_transaction(
            transaction_type="supplier_payment",
            product=None,
            quantity=sum(products.values()),
            amount=-amount,
            notes=f"Payment to {supplier.name}: {description}"
        )

        # Check if supplier will actually deliver (reliability check)
        will_deliver = random.random() < supplier.reliability

        if will_deliver:
            # Create pending orders for each product
            delivery_day = self.current_day + supplier.delivery_days
            order_id = f"ORD-{uuid.uuid4().hex[:8].upper()}"

            for product, quantity in products.items():
                if product in PRODUCT_CATALOG:
                    # Calculate per-unit cost for this order
                    unit_cost = amount / sum(products.values()) if products else 0

                    order = PendingOrder(
                        order_id=f"{order_id}-{product}",
                        product=product,
                        quantity=quantity,
                        supplier_cost=unit_cost,
                        order_day=self.current_day,
                        delivery_day=delivery_day,
                        total_cost=unit_cost * quantity
                    )
                    self.pending_orders.append(order)

            return {
                "success": True,
                "order_id": order_id,
                "amount_paid": amount,
                "products": products,
                "expected_delivery_day": delivery_day,
                "supplier": supplier.name
            }
        else:
            # Scammer - took money but won't deliver
            return {
                "success": True,  # Payment went through
                "order_id": f"ORD-{uuid.uuid4().hex[:8].upper()}",
                "amount_paid": amount,
                "products": products,
                "expected_delivery_day": self.current_day + supplier.delivery_days,
                "supplier": supplier.name,
                # Note: Order won't actually be added to pending_orders
                # Agent will notice when delivery never arrives
            }

    def get_pending_orders(self) -> List[Dict[str, Any]]:
        """Get list of orders currently in transit."""
        return [
            {
                "order_id": order.order_id,
                "product": order.product,
                "quantity": order.quantity,
                "ordered_day": order.order_day,
                "delivery_day": order.delivery_day,
                "days_until_delivery": order.delivery_day - self.current_day
            }
            for order in self.pending_orders
        ]

    def add_output_tokens(self, tokens: int):
        """
        Track output tokens used by the agent.

        Args:
            tokens: Number of output tokens to add
        """
        self.accumulated_output_tokens += tokens

    def process_weekly_token_charge(self) -> Dict[str, Any]:
        """
        Process weekly token cost charge (VendingBench 2: $100 per million output tokens).

        Called at the end of each week (every 7 days).

        Returns:
            Dict with charge details
        """
        # Only charge once per week
        if self.current_day - self.last_token_charge_day < 7:
            return {"charged": False, "reason": "Not yet a full week"}

        # Calculate cost: $100 per million output tokens
        cost = (self.accumulated_output_tokens / 1_000_000) * self.token_cost_per_million

        if cost > 0:
            self.cash_balance -= cost
            self.total_token_costs += cost

            # Record transaction
            self._record_transaction(
                transaction_type="token_cost",
                product=None,
                quantity=self.accumulated_output_tokens,
                amount=-cost,
                notes=f"Weekly token cost: {self.accumulated_output_tokens:,} output tokens @ ${self.token_cost_per_million}/million"
            )

            print(f"    [TOKEN COST] Week ending Day {self.current_day}: {self.accumulated_output_tokens:,} tokens = ${cost:.2f} | Cash: ${self.cash_balance + cost:.2f} → ${self.cash_balance:.2f}", flush=True)

        # Reset for next week
        tokens_charged = self.accumulated_output_tokens
        self.accumulated_output_tokens = 0
        self.last_token_charge_day = self.current_day

        return {
            "charged": True,
            "tokens": tokens_charged,
            "cost": cost,
            "day": self.current_day,
            "new_balance": self.cash_balance
        }

    def _process_overnight_sales(self) -> Dict[str, Any]:
        """
        Process customer purchases overnight based on machine inventory and prices.

        Returns:
            Dictionary with sales details
        """
        from src.products import calculate_demand

        sales = {}
        total_revenue = 0.0
        total_units_sold = 0

        # Get list of products currently in machine (for choice multiplier)
        products_in_machine = [
            product for product, qty in self.machine_inventory.items()
            if qty > 0
        ]

        for product in PRODUCT_CATALOG.keys():
            # Check if product is available in machine
            available = self.machine_inventory[product]
            if available == 0:
                continue

            # Calculate demand with all VendingBench factors
            price = self.current_prices[product]
            demand = calculate_demand(
                product=product,
                price=price,
                day=self.current_day,
                products_in_machine=products_in_machine
            )

            # Actual sales = min(demand, available inventory)
            actual_sales = min(demand, available)

            if actual_sales > 0:
                # Calculate revenue
                revenue = actual_sales * price

                # Record the sale
                sales[product] = {
                    "quantity": actual_sales,
                    "price": price,
                    "revenue": revenue
                }
                total_revenue += revenue
                total_units_sold += actual_sales

                # Update machine inventory and slot usage
                self.remove_from_machine(product, actual_sales)

                # Add to cash balance
                self.cash_balance += revenue

                # Record transaction
                self._record_transaction(
                    transaction_type="sale",
                    product=product,
                    quantity=actual_sales,
                    amount=revenue,
                    notes=f"Overnight sales: {actual_sales} units at ${price:.2f}"
                )

        return {
            "sales_by_product": sales,
            "total_revenue": total_revenue,
            "total_units_sold": total_units_sold
        }

    def advance_day(self) -> Dict[str, Any]:
        """
        Advance simulation by one day (legacy method for backwards compatibility).

        NOTE: For the agent-driven loop, use process_overnight_and_advance_day() instead.

        Returns:
            Dictionary with day summary
        """
        self.current_day += 1

        # Charge daily operating fee
        self._charge_daily_fee()

        # Check for spoilage
        spoiled = self._process_spoilage()

        # Generate daily report
        report = self._log_daily_report()
        report["spoiled_items"] = spoiled

        # Check if simulation is complete
        if self.current_day >= self.config.simulation_days:
            self.is_complete = True

        return report

    def _charge_daily_fee(self) -> bool:
        """
        Charge the daily operating fee and track bankruptcy status.

        Returns:
            True if agent should be terminated (10 consecutive bankrupt days)
        """
        # Check if agent can pay the fee
        can_pay = self.cash_balance >= self.config.daily_fee

        # Charge the fee regardless (can go negative)
        self.cash_balance -= self.config.daily_fee
        self._record_transaction(
            transaction_type="fee",
            product=None,
            quantity=1,
            amount=-self.config.daily_fee,
            notes=f"Day {self.current_day} operating fee"
        )

        # Track consecutive bankrupt days (paper: terminate after 10 consecutive days)
        if not can_pay:
            self.consecutive_bankrupt_days += 1
        else:
            self.consecutive_bankrupt_days = 0

        # Return True if should terminate
        return self.consecutive_bankrupt_days >= self.bankruptcy_threshold

    def _process_spoilage(self) -> List[Dict[str, Any]]:
        """Process spoiled inventory items."""
        spoiled_items = []

        for product, items in self.storage_inventory.items():
            # Filter expired items
            valid_items = []
            for item in items:
                if item.is_expired(self.current_day):
                    spoiled_items.append({
                        "product": product,
                        "quantity": item.quantity,
                        "cost": item.supplier_cost * item.quantity
                    })
                    # Record loss
                    self._record_transaction(
                        transaction_type="spoilage",
                        product=product,
                        quantity=item.quantity,
                        amount=-item.supplier_cost * item.quantity,
                        notes=f"Spoiled on day {self.current_day}"
                    )
                else:
                    valid_items.append(item)

            self.storage_inventory[product] = valid_items

        return spoiled_items

    def _record_transaction(
        self,
        transaction_type: str,
        product: Optional[str],
        quantity: int,
        amount: float,
        notes: str = ""
    ):
        """Record a business transaction."""
        transaction = Transaction(
            day=self.current_day,
            transaction_type=transaction_type,
            product=product,
            quantity=quantity,
            amount=amount,
            balance_after=self.cash_balance,
            notes=notes
        )
        self.transaction_history.append(transaction)

    def _log_daily_report(self) -> Dict[str, Any]:
        """Generate and store daily business report."""
        # Calculate storage inventory value (wholesale price)
        storage_value = sum(
            sum(item.supplier_cost * item.quantity for item in items)
            for items in self.storage_inventory.values()
        )

        # Calculate machine inventory value (wholesale price)
        machine_value = sum(
            PRODUCT_CATALOG[product]["supplier_cost"] * quantity
            for product, quantity in self.machine_inventory.items()
        )

        # Count machine inventory units
        machine_units = sum(self.machine_inventory.values())

        report = {
            "day": self.current_day,
            "cash_balance": self.cash_balance,
            "storage_inventory": {
                product: sum(item.quantity for item in items)
                for product, items in self.storage_inventory.items()
            },
            "machine_inventory": self.machine_inventory.copy(),
            "storage_value": storage_value,
            "machine_value": machine_value,
            "machine_units": machine_units,
            "current_prices": self.current_prices.copy(),
            "net_worth": self.cash_balance + storage_value + machine_value
        }

        self.daily_reports.append(report)
        return report

    def get_state(self) -> Dict[str, Any]:
        """Get current simulation state."""
        return {
            "day": self.current_day,
            "cash_balance": self.cash_balance,
            "storage_inventory": {
                product: sum(item.quantity for item in items)
                for product, items in self.storage_inventory.items()
            },
            "machine_inventory": self.machine_inventory.copy(),
            "prices": self.current_prices.copy(),
            "message_count": self.message_count,
            "is_complete": self.is_complete,
            "days_remaining": self.config.simulation_days - self.current_day,
            "machine_slots": self.get_machine_slot_status()
        }

    def get_machine_slot_status(self) -> Dict[str, Any]:
        """Get current machine slot usage."""
        return {
            "small_slots_used": self.machine_small_slots_used,
            "small_slots_max": self.machine_small_slots_max,
            "small_slots_available": self.machine_small_slots_max - self.machine_small_slots_used,
            "large_slots_used": self.machine_large_slots_used,
            "large_slots_max": self.machine_large_slots_max,
            "large_slots_available": self.machine_large_slots_max - self.machine_large_slots_used,
            "total_used": self.machine_small_slots_used + self.machine_large_slots_used,
            "total_max": self.machine_small_slots_max + self.machine_large_slots_max
        }

    def can_stock_product(self, product: str, quantity: int) -> tuple:
        """
        Check if product can be stocked in machine.

        Args:
            product: Product name
            quantity: Quantity to stock

        Returns:
            Tuple of (can_stock: bool, reason: str, max_stockable: int)
        """
        if product not in PRODUCT_CATALOG:
            return False, f"Unknown product: {product}", 0

        product_size = PRODUCT_CATALOG[product]["size"]

        if product_size == "small":
            available = self.machine_small_slots_max - self.machine_small_slots_used
        else:  # large
            available = self.machine_large_slots_max - self.machine_large_slots_used

        if quantity <= available:
            return True, "OK", quantity
        elif available > 0:
            return False, f"Only {available} {product_size} slots available (requested {quantity})", available
        else:
            return False, f"No {product_size} slots available in machine", 0

    def add_to_machine(self, product: str, quantity: int) -> bool:
        """
        Add items to machine and update slot usage.

        Args:
            product: Product name
            quantity: Quantity to add

        Returns:
            True if successful
        """
        product_size = PRODUCT_CATALOG[product]["size"]

        if product_size == "small":
            self.machine_small_slots_used += quantity
        else:
            self.machine_large_slots_used += quantity

        self.machine_inventory[product] += quantity
        return True

    def remove_from_machine(self, product: str, quantity: int) -> bool:
        """
        Remove items from machine and update slot usage (called on sales).

        Args:
            product: Product name
            quantity: Quantity removed (sold)

        Returns:
            True if successful
        """
        product_size = PRODUCT_CATALOG[product]["size"]

        if product_size == "small":
            self.machine_small_slots_used = max(0, self.machine_small_slots_used - quantity)
        else:
            self.machine_large_slots_used = max(0, self.machine_large_slots_used - quantity)

        self.machine_inventory[product] = max(0, self.machine_inventory[product] - quantity)
        return True

    def calculate_final_metrics(self) -> Dict[str, Any]:
        """Calculate final business metrics.

        VendingBench 2 scoring: CASH BALANCE ONLY after 365 days.
        (Inventory value is tracked but not counted toward final score)
        """
        # Storage inventory value (wholesale price) - for reference only
        storage_value = sum(
            sum(item.supplier_cost * item.quantity for item in items)
            for items in self.storage_inventory.values()
        )

        # Machine inventory value (wholesale price) - for reference only
        machine_value = sum(
            PRODUCT_CATALOG[product]["supplier_cost"] * quantity
            for product, quantity in self.machine_inventory.items()
        )

        # Net worth (for reference, NOT the score)
        final_net_worth = self.cash_balance + storage_value + machine_value

        # Revenue and profit calculations
        total_revenue = sum(
            t.amount for t in self.transaction_history
            if t.transaction_type == "sale" and t.amount > 0
        )
        total_costs = sum(
            abs(t.amount) for t in self.transaction_history
            if t.amount < 0
        )
        total_profit = total_revenue - total_costs

        # Days profitable (cash > starting cash)
        days_profitable = sum(
            1 for report in self.daily_reports
            if report.get("cash_balance", 0) > self.config.starting_cash
        )

        return {
            # PRIMARY SCORE (VendingBench 2): Cash balance only
            "score": self.cash_balance,  # THE OFFICIAL VENDINGBENCH 2 SCORE
            "final_cash_balance": self.cash_balance,
            "starting_cash": self.config.starting_cash,
            "cash_gain_loss": self.cash_balance - self.config.starting_cash,

            # Inventory values (for reference, NOT part of score)
            "storage_value": storage_value,
            "machine_value": machine_value,
            "final_net_worth": final_net_worth,  # Reference only
            "starting_net_worth": self.starting_net_worth,
            "profit_loss": final_net_worth - self.starting_net_worth,

            # Token costs (VendingBench 2: $100 per million output tokens)
            "total_token_costs": self.total_token_costs,
            "total_output_tokens": self.accumulated_output_tokens,  # Remaining uncharged tokens

            # Other metrics
            "total_revenue": total_revenue,
            "total_costs": total_costs,
            "total_profit": total_profit,
            "days_profitable": days_profitable,
            "days_simulated": self.current_day,
            "messages_used": self.message_count
        }

    def export_state(self, filepath: str):
        """Export full simulation state to JSON."""
        state = {
            "config": self.config.to_dict(),
            "current_state": self.get_state(),
            "daily_reports": self.daily_reports,
            "transactions": [
                {
                    "day": t.day,
                    "type": t.transaction_type,
                    "product": t.product,
                    "quantity": t.quantity,
                    "amount": t.amount,
                    "balance": t.balance_after,
                    "notes": t.notes
                }
                for t in self.transaction_history
            ],
            "final_metrics": self.calculate_final_metrics() if self.is_complete else None
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
