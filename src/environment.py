"""
Core vending machine business simulation environment.

Manages business state, inventory, finances, and progression.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json

from src.products import PRODUCT_CATALOG, calculate_demand, get_seasonal_factor
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

    def advance_day(self) -> Dict[str, Any]:
        """
        Advance simulation by one day.

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

    def _charge_daily_fee(self):
        """Charge the daily operating fee."""
        self.cash_balance -= self.config.daily_fee
        self._record_transaction(
            transaction_type="fee",
            product=None,
            quantity=1,
            amount=-self.config.daily_fee,
            notes=f"Day {self.current_day} operating fee"
        )

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
            "days_remaining": self.config.simulation_days - self.current_day
        }

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
