"""
Business event generation for vending machine simulation.

Generates realistic events like purchases, supplier emails, and random occurrences.
"""

from typing import Dict, List, Any, Optional
import random
import uuid

from src.products import PRODUCT_CATALOG, calculate_demand, get_seasonal_factor
from src.environment import VendingEnvironment, InventoryItem
from config.simulation_config import EventComplexity


class EventGenerator:
    """Generates business events based on simulation state and complexity level."""

    def __init__(self, env: VendingEnvironment, complexity: str = "simple"):
        """
        Initialize event generator.

        Args:
            env: Simulation environment
            complexity: Event complexity level ("simple", "medium", "full")
        """
        self.env = env
        self.event_config = EventComplexity.get_config(complexity)
        self.competitor_appeared = False

    def generate_daily_events(self) -> List[Dict[str, Any]]:
        """
        Generate all events for the current day.

        Returns:
            List of event dictionaries
        """
        events = []

        # 1. Customer purchases (always happens if items in machine)
        if self.event_config["customer_purchases"]:
            purchase_event = self._generate_purchases()
            if purchase_event:
                events.append(purchase_event)

        # 2. Supplier emails (weekly)
        if self.event_config["supplier_emails"] and self.env.current_day % 7 == 0:
            supplier_event = self._generate_supplier_email()
            if supplier_event:
                events.append(supplier_event)

        # 3. Competitor appears (day 100 in full mode)
        if (self.event_config["competitors"] and
            self.env.current_day == 100 and
            not self.competitor_appeared):
            events.append(self._generate_competitor_event())
            self.competitor_appeared = True

        # 4. Random maintenance (5% chance in medium/full)
        if self.event_config["maintenance"] and random.random() < 0.05:
            events.append(self._generate_maintenance_event())

        return events

    def _generate_purchases(self) -> Optional[Dict[str, Any]]:
        """
        Generate customer purchase event based on demand and pricing.

        Returns:
            Purchase event dict or None if no sales
        """
        sales = {}
        total_revenue = 0

        for product in PRODUCT_CATALOG.keys():
            # Check if product is available in machine
            if self.env.machine_inventory[product] == 0:
                continue

            # Calculate demand
            price = self.env.current_prices[product]
            seasonal = get_seasonal_factor(product, self.env.current_day)
            demand = calculate_demand(product, price, self.env.current_day, seasonal)

            # Actual sales = min(demand, available inventory)
            actual_sales = min(demand, self.env.machine_inventory[product])

            if actual_sales > 0:
                # Process sale
                revenue = actual_sales * price
                sales[product] = {
                    "quantity": actual_sales,
                    "price": price,
                    "revenue": revenue
                }
                total_revenue += revenue

                # Update inventory
                self.env.machine_inventory[product] -= actual_sales

                # Record transaction
                self.env._record_transaction(
                    transaction_type="sale",
                    product=product,
                    quantity=actual_sales,
                    amount=revenue,
                    notes=f"Customer purchases on day {self.env.current_day}"
                )

                # Update cash
                self.env.cash_balance += revenue

        if not sales:
            return None

        return {
            "type": "purchase",
            "day": self.env.current_day,
            "sales": sales,
            "total_revenue": total_revenue,
            "description": f"Customer purchases: {sum(s['quantity'] for s in sales.values())} items sold for ${total_revenue:.2f}"
        }

    def _generate_supplier_email(self) -> Dict[str, Any]:
        """
        Generate supplier communication email.

        Returns:
            Email event dict
        """
        suppliers = [
            {
                "name": "Supplier A",
                "email": "orders@supplierA.com",
                "specialization": "beverages"
            },
            {
                "name": "Supplier B",
                "email": "sales@supplierB.com",
                "specialization": "snacks"
            }
        ]

        supplier = random.choice(suppliers)

        # Email types
        email_types = [
            "price_update",
            "new_product",
            "delivery_reminder",
            "promotion"
        ]

        email_type = random.choice(email_types)

        if email_type == "price_update":
            subject = "Price Update Notification"
            body = f"Dear Business Partner,\n\nPlease be advised that due to increased costs, our prices will be adjusted slightly next month. Current prices remain valid for orders placed this week.\n\nBest regards,\n{supplier['name']}"
        elif email_type == "new_product":
            subject = "New Product Available"
            body = f"Hello,\n\nWe're excited to introduce a new product line that might interest your vending business. Contact us for samples and pricing.\n\nBest,\n{supplier['name']}"
        elif email_type == "delivery_reminder":
            subject = "Delivery Schedule"
            body = f"Reminder: We deliver on Tuesdays and Fridays. Please place orders at least 2 days in advance.\n\nThank you,\n{supplier['name']}"
        else:  # promotion
            subject = "Special Promotion"
            body = f"Limited time offer: 10% discount on bulk orders (50+ units) placed this week!\n\nDon't miss out,\n{supplier['name']}"

        email = {
            "id": str(uuid.uuid4())[:8],
            "from": supplier["email"],
            "to": "agent@vendingbusiness.com",
            "subject": subject,
            "body": body,
            "date": self.env.current_day,
            "read": False
        }

        self.env.email_inbox.append(email)

        return {
            "type": "email",
            "day": self.env.current_day,
            "email_id": email["id"],
            "from": supplier["name"],
            "subject": subject,
            "description": f"Email from {supplier['name']}: {subject}"
        }

    def _generate_competitor_event(self) -> Dict[str, Any]:
        """
        Generate competitor appearance event.

        Returns:
            Competitor event dict
        """
        email = {
            "id": str(uuid.uuid4())[:8],
            "from": "local_news@community.com",
            "to": "agent@vendingbusiness.com",
            "subject": "New Vending Machine in Area",
            "body": "FYI - We noticed a new vending machine was installed about 50 feet from your location. They appear to be offering similar products at competitive prices (Coffee $2.50, Chips $1.25, Chocolate $1.75).",
            "date": self.env.current_day,
            "read": False
        }

        self.env.email_inbox.append(email)

        return {
            "type": "competitor",
            "day": self.env.current_day,
            "email_id": email["id"],
            "competitor_prices": {
                "coffee": 2.50,
                "chips": 1.25,
                "chocolate": 1.75
            },
            "description": "Competitor vending machine opened nearby with lower prices"
        }

    def _generate_maintenance_event(self) -> Dict[str, Any]:
        """
        Generate maintenance/issue event.

        Returns:
            Maintenance event dict
        """
        issues = [
            "coin mechanism jammed",
            "display screen flickering",
            "cooling system needs cleaning",
            "product dispenser stuck"
        ]

        issue = random.choice(issues)
        cost = random.uniform(20, 100)

        # Charge maintenance cost
        self.env.cash_balance -= cost
        self.env._record_transaction(
            transaction_type="maintenance",
            product=None,
            quantity=1,
            amount=-cost,
            notes=f"Maintenance: {issue}"
        )

        return {
            "type": "maintenance",
            "day": self.env.current_day,
            "issue": issue,
            "cost": cost,
            "description": f"Maintenance issue: {issue} (Cost: ${cost:.2f})"
        }
