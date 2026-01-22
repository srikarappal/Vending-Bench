"""
Supplier definitions for email-based negotiation system.

Based on VendingBench 2 specification:
- 4 supplier personas: Friendly, Negotiator, Membership Scammer, Price Scammer
- All suppliers sell the same products at different prices
- Suppliers respond via LLM-generated emails
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime


@dataclass
class Supplier:
    """Represents a supplier with specific persona and pricing."""
    supplier_id: str
    name: str
    email: str
    persona: str  # "friendly", "negotiator", "membership_scammer", "price_scammer"
    base_prices: Dict[str, float]  # product -> price per unit
    negotiation_flexibility: float  # 0.0-1.0, how much they'll lower prices
    reliability: float  # 0.0-1.0, chance of successful delivery
    response_delay_days: int = 1  # days to respond to emails
    membership_fee: Optional[float] = None  # for membership scammers
    min_order_quantity: int = 10  # minimum units per order
    delivery_days: int = 3  # days for delivery after payment


@dataclass
class SupplierEmail:
    """Represents an email in the system."""
    email_id: int
    from_addr: str
    to_addr: str
    subject: str
    body: str
    sent_day: int
    read: bool = False
    replied_to: Optional[int] = None  # email_id this is replying to


@dataclass
class PendingNegotiation:
    """Tracks an ongoing negotiation with a supplier."""
    negotiation_id: str
    supplier_id: str
    products: Dict[str, int]  # product -> quantity requested
    agent_proposed_price: Optional[float] = None  # total price agent proposed
    supplier_quoted_price: Optional[float] = None  # supplier's current quote
    status: str = "initial"  # "initial", "quoted", "counter", "accepted", "rejected"
    created_day: int = 0
    last_response_day: int = 0
    email_chain: List[int] = field(default_factory=list)  # email_ids in conversation


@dataclass
class SupplierOrder:
    """Represents an order placed with a supplier after negotiation."""
    order_id: str
    supplier_id: str
    products: Dict[str, int]  # product -> quantity
    total_cost: float
    order_day: int
    expected_delivery_day: int
    delivered: bool = False
    failed: bool = False  # for scammer scenarios


# Agent's email address
AGENT_EMAIL = "charles.paxton@vendingsandstuff.com"


# Supplier Catalog - 4 suppliers with different personas
# All sell the same products (coffee, chocolate, chips, soda) matching our PRODUCT_CATALOG
SUPPLIER_CATALOG = {
    "wholesale_direct": Supplier(
        supplier_id="wholesale_direct",
        name="Wholesale Direct",
        email="orders@wholesaledirect.com",
        persona="friendly",
        base_prices={
            "coffee": 1.00,
            "chocolate": 0.50,
            "chips": 0.35,
            "soda": 0.45,
        },
        negotiation_flexibility=0.1,  # barely negotiates - already good prices
        reliability=0.95,  # very reliable
        response_delay_days=1,
        min_order_quantity=10,
        delivery_days=2,
    ),

    "bulk_suppliers_inc": Supplier(
        supplier_id="bulk_suppliers_inc",
        name="Bulk Suppliers Inc",
        email="sales@bulksuppliers.com",
        persona="negotiator",
        base_prices={
            "coffee": 1.80,   # starts at 1.8x friendly price
            "chocolate": 1.00,  # 2x friendly price
            "chips": 0.80,    # 2.3x friendly price
            "soda": 0.90,     # 2x friendly price
        },
        negotiation_flexibility=0.5,  # will go down to ~50% of initial quote
        reliability=0.90,
        response_delay_days=1,
        min_order_quantity=20,  # requires larger orders
        delivery_days=3,
    ),

    "vending_elite": Supplier(
        supplier_id="vending_elite",
        name="Vending Elite Club",
        email="membership@vendingelite.com",
        persona="membership_scammer",
        base_prices={
            "coffee": 0.80,   # amazing prices (too good to be true)
            "chocolate": 0.40,
            "chips": 0.30,
            "soda": 0.35,
        },
        negotiation_flexibility=0.0,  # doesn't negotiate - just wants membership fee
        reliability=0.2,  # very likely to not deliver
        response_delay_days=1,
        membership_fee=75.0,  # requires upfront membership
        min_order_quantity=5,
        delivery_days=5,
    ),

    "vendmart": Supplier(
        supplier_id="vendmart",
        name="VendMart",
        email="vendmart@vendmart.com",
        persona="price_scammer",
        base_prices={
            "coffee": 2.50,   # near retail prices
            "chocolate": 2.00,
            "chips": 1.50,
            "soda": 2.40,
        },
        negotiation_flexibility=0.1,  # barely budges on price
        reliability=0.85,  # will deliver, just overcharges
        response_delay_days=1,
        min_order_quantity=10,
        delivery_days=2,
    ),
}


def get_supplier_by_email(email: str) -> Optional[Supplier]:
    """Look up supplier by email address."""
    for supplier in SUPPLIER_CATALOG.values():
        if supplier.email.lower() == email.lower():
            return supplier
    return None


def get_supplier_by_id(supplier_id: str) -> Optional[Supplier]:
    """Look up supplier by ID."""
    return SUPPLIER_CATALOG.get(supplier_id)


def list_all_suppliers() -> List[Supplier]:
    """Get all suppliers."""
    return list(SUPPLIER_CATALOG.values())


def calculate_order_total(supplier: Supplier, products: Dict[str, int]) -> float:
    """Calculate total cost for an order at supplier's base prices."""
    total = 0.0
    for product, quantity in products.items():
        if product in supplier.base_prices:
            total += supplier.base_prices[product] * quantity
    return total


def get_negotiated_price(supplier: Supplier, base_total: float, negotiation_rounds: int) -> float:
    """
    Calculate price after negotiation based on supplier flexibility.

    Args:
        supplier: The supplier
        base_total: Initial quoted price
        negotiation_rounds: Number of back-and-forth negotiations

    Returns:
        Negotiated price (may be lower than base for negotiator persona)
    """
    if supplier.persona == "friendly":
        # Friendly suppliers already have good prices, minimal discount
        discount = min(supplier.negotiation_flexibility * negotiation_rounds * 0.02, 0.05)
    elif supplier.persona == "negotiator":
        # Negotiators start high but come down significantly
        # Each round can get ~10% more discount, up to flexibility limit
        discount = min(supplier.negotiation_flexibility * (1 - 0.7 ** negotiation_rounds),
                      supplier.negotiation_flexibility)
    elif supplier.persona == "membership_scammer":
        # Scammers don't negotiate - they want the membership fee
        discount = 0.0
    elif supplier.persona == "price_scammer":
        # Price scammers barely move
        discount = min(supplier.negotiation_flexibility * negotiation_rounds * 0.01, 0.05)
    else:
        discount = 0.0

    return base_total * (1 - discount)


# Product info for suppliers to reference (matches products.py)
SUPPLIER_PRODUCT_INFO = {
    "coffee": {
        "display_name": "Coffee",
        "unit": "cup",
        "typical_retail": 3.00,
    },
    "chocolate": {
        "display_name": "Chocolate Bar",
        "unit": "bar",
        "typical_retail": 2.00,
    },
    "chips": {
        "display_name": "Chips",
        "unit": "bag",
        "typical_retail": 1.50,
    },
    "soda": {
        "display_name": "Soda",
        "unit": "can",
        "typical_retail": 2.50,
    },
}
