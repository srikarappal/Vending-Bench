"""
Core vending machine business simulation environment.

Manages business state, inventory, finances, and progression.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json

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

    def __init__(self, config: SimulationConfig):
        """Initialize simulation environment."""
        self.config = config

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

        # Email/communication system
        self.email_inbox: List[Dict[str, Any]] = []
        self.email_sent: List[Dict[str, Any]] = []

        # Pending orders (orders in transit, not yet delivered)
        self.pending_orders: List[PendingOrder] = []

        # Bankruptcy tracking (paper: terminate after 10 consecutive days of not paying fee)
        self.consecutive_bankrupt_days = 0
        self.bankruptcy_threshold = 10  # Days before termination

        # Initialize starter inventory if configured
        if config.starting_inventory_units > 0:
            self._initialize_starter_inventory(config.starting_inventory_units)

        # Calculate and store starting net worth (cash + inventory value)
        starting_storage_value = sum(
            sum(item.supplier_cost * item.quantity for item in items)
            for items in self.storage_inventory.values()
        )
        self.starting_net_worth = self.cash_balance + starting_storage_value

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
        6. Generate morning briefing

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

        # 6. Generate daily report
        report = self._log_daily_report()

        # 7. Check if simulation is complete (days finished OR bankrupt)
        if self.current_day >= self.config.simulation_days:
            self.is_complete = True
        if is_bankrupt:
            self.is_complete = True

        return {
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
        # Calculate total inventory value
        storage_value = sum(
            sum(item.supplier_cost * item.quantity for item in items)
            for items in self.storage_inventory.values()
        )

        # Count machine inventory
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
            "machine_units": machine_units,
            "current_prices": self.current_prices.copy(),
            "net_worth": self.cash_balance + storage_value
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
        """Calculate final business metrics."""
        # Net worth calculation
        storage_value = sum(
            sum(item.supplier_cost * item.quantity for item in items)
            for items in self.storage_inventory.values()
        )
        final_net_worth = self.cash_balance + storage_value

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

        # Days profitable
        days_profitable = sum(
            1 for report in self.daily_reports
            if report.get("cash_balance", 0) > self.config.starting_cash
        )

        return {
            "final_net_worth": final_net_worth,
            "starting_net_worth": self.starting_net_worth,
            "starting_cash": self.config.starting_cash,
            "final_cash_balance": self.cash_balance,
            "cash_gain_loss": self.cash_balance - self.config.starting_cash,
            "profit_loss": final_net_worth - self.starting_net_worth,
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
