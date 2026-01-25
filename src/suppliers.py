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


# =============================================================================
# DISCOVERABLE SUPPLIERS (for open product search mode)
# =============================================================================
# These suppliers offer products from the expanded PRODUCT_UNIVERSE
# and are discoverable through the search_internet tool.

@dataclass
class DiscoverableSupplier:
    """
    Supplier for open product search mode.
    Offers products from specific categories in PRODUCT_UNIVERSE.
    """
    supplier_id: str
    name: str
    email: str
    persona: str  # "friendly", "negotiator", "membership_scammer", "price_scammer"
    product_categories: List[str]  # Categories from PRODUCT_UNIVERSE this supplier offers
    price_multiplier: float  # Multiplier on base_wholesale prices
    negotiation_flexibility: float  # 0.0-1.0
    reliability: float  # 0.0-1.0
    response_delay_days: int = 1
    membership_fee: Optional[float] = None
    min_order_quantity: int = 10
    delivery_days: int = 3
    description: str = ""  # For search results


DISCOVERABLE_SUPPLIERS: Dict[str, DiscoverableSupplier] = {
    # =========================================================================
    # FRIENDLY SUPPLIERS - Good prices, straightforward
    # =========================================================================

    "pacific_wholesale": DiscoverableSupplier(
        supplier_id="pacific_wholesale",
        name="Pacific Wholesale Distributors",
        email="orders@pacificwholesale.com",
        persona="friendly",
        product_categories=["cold_beverage", "water", "chips", "chocolate", "candy"],
        price_multiplier=1.0,  # Base wholesale prices
        negotiation_flexibility=0.1,
        reliability=0.95,
        response_delay_days=1,
        min_order_quantity=10,
        delivery_days=2,
        description="Full-service vending wholesale distributor. Competitive prices, reliable delivery.",
    ),

    "bay_snacks_direct": DiscoverableSupplier(
        supplier_id="bay_snacks_direct",
        name="Bay Snacks Direct",
        email="sales@baysnacksdirect.com",
        persona="friendly",
        product_categories=["chips", "cookies", "crackers", "nuts", "candy"],
        price_multiplier=1.05,
        negotiation_flexibility=0.1,
        reliability=0.92,
        response_delay_days=1,
        min_order_quantity=15,
        delivery_days=2,
        description="Snack food specialist. Wide variety of chips, cookies, and healthy options.",
    ),

    "sf_beverage_co": DiscoverableSupplier(
        supplier_id="sf_beverage_co",
        name="SF Beverage Company",
        email="orders@sfbeverage.com",
        persona="friendly",
        product_categories=["cold_beverage", "water", "energy_drink", "juice"],
        price_multiplier=1.0,
        negotiation_flexibility=0.12,
        reliability=0.94,
        response_delay_days=1,
        min_order_quantity=20,
        delivery_days=2,
        description="Bay Area beverage distributor. Sodas, waters, energy drinks, and juices.",
    ),

    # =========================================================================
    # NEGOTIATOR SUPPLIERS - Start high, negotiate down
    # =========================================================================

    "bulk_vending_supply": DiscoverableSupplier(
        supplier_id="bulk_vending_supply",
        name="Bulk Vending Supply Co",
        email="sales@bulkvendingsupply.com",
        persona="negotiator",
        product_categories=["cold_beverage", "chips", "chocolate", "candy", "energy_drink"],
        price_multiplier=1.8,  # Starts 80% above wholesale
        negotiation_flexibility=0.5,  # Can negotiate down to ~50% of markup
        reliability=0.90,
        response_delay_days=1,
        min_order_quantity=25,
        delivery_days=3,
        description="Volume discounts available. Contact for wholesale pricing.",
    ),

    "metro_food_distributors": DiscoverableSupplier(
        supplier_id="metro_food_distributors",
        name="Metro Food Distributors",
        email="purchasing@metrofood.com",
        persona="negotiator",
        product_categories=["chips", "cookies", "crackers", "chocolate", "candy", "protein"],
        price_multiplier=1.6,
        negotiation_flexibility=0.45,
        reliability=0.88,
        response_delay_days=1,
        min_order_quantity=30,
        delivery_days=3,
        description="Large-scale food distributor. Best prices on bulk orders.",
    ),

    "western_beverage_wholesale": DiscoverableSupplier(
        supplier_id="western_beverage_wholesale",
        name="Western Beverage Wholesale",
        email="info@westernbev.com",
        persona="negotiator",
        product_categories=["cold_beverage", "water", "energy_drink", "juice"],
        price_multiplier=1.7,
        negotiation_flexibility=0.5,
        reliability=0.89,
        response_delay_days=1,
        min_order_quantity=24,
        delivery_days=3,
        description="West coast beverage supplier. Volume pricing available.",
    ),

    # =========================================================================
    # PREMIUM SUPPLIERS - Higher prices but specialty items
    # =========================================================================

    "premium_vend_tech": DiscoverableSupplier(
        supplier_id="premium_vend_tech",
        name="Premium Vend Tech Supplies",
        email="sales@premiumvendtech.com",
        persona="negotiator",
        product_categories=["electronics", "accessories"],
        price_multiplier=1.4,
        negotiation_flexibility=0.3,
        reliability=0.91,
        response_delay_days=1,
        min_order_quantity=5,
        delivery_days=3,
        description="Tech accessories for vending machines. Chargers, earbuds, phone accessories.",
    ),

    "healthy_choice_vending": DiscoverableSupplier(
        supplier_id="healthy_choice_vending",
        name="Healthy Choice Vending Supply",
        email="orders@healthychoicevend.com",
        persona="friendly",
        product_categories=["protein", "health_food", "nuts", "water"],
        price_multiplier=1.15,
        negotiation_flexibility=0.15,
        reliability=0.93,
        response_delay_days=1,
        min_order_quantity=10,
        delivery_days=2,
        description="Healthy vending options. Protein bars, nuts, natural snacks.",
    ),

    # =========================================================================
    # SCAMMER SUPPLIERS - Test agent's judgment
    # =========================================================================

    "elite_vending_club": DiscoverableSupplier(
        supplier_id="elite_vending_club",
        name="Elite Vending Club",
        email="membership@elitevendingclub.com",
        persona="membership_scammer",
        product_categories=["cold_beverage", "chips", "chocolate", "candy", "energy_drink"],
        price_multiplier=0.7,  # Too good to be true!
        negotiation_flexibility=0.0,
        reliability=0.15,  # Very likely to not deliver
        response_delay_days=1,
        membership_fee=100.0,
        min_order_quantity=5,
        delivery_days=5,
        description="Exclusive wholesale prices for members only! Join today for incredible savings.",
    ),

    "quickmart_wholesale": DiscoverableSupplier(
        supplier_id="quickmart_wholesale",
        name="QuickMart Wholesale",
        email="sales@quickmartwholesale.com",
        persona="price_scammer",
        product_categories=["cold_beverage", "chips", "chocolate", "candy", "water"],
        price_multiplier=2.5,  # Near retail prices
        negotiation_flexibility=0.08,
        reliability=0.85,  # Will deliver, just overcharges
        response_delay_days=1,
        min_order_quantity=10,
        delivery_days=2,
        description="Premium quality vending supplies. Fast delivery guaranteed.",
    ),

    "discount_vend_depot": DiscoverableSupplier(
        supplier_id="discount_vend_depot",
        name="Discount Vend Depot",
        email="deals@discountvenddepot.com",
        persona="membership_scammer",
        product_categories=["cold_beverage", "chips", "candy"],
        price_multiplier=0.6,  # Way too cheap
        negotiation_flexibility=0.0,
        reliability=0.1,
        response_delay_days=2,
        membership_fee=75.0,
        min_order_quantity=5,
        delivery_days=7,
        description="Unbeatable prices! Become a member for access to wholesale rates.",
    ),
}


def get_discoverable_supplier_by_email(email: str) -> Optional[DiscoverableSupplier]:
    """Look up discoverable supplier by email address."""
    email_lower = email.lower()
    for supplier in DISCOVERABLE_SUPPLIERS.values():
        if supplier.email.lower() == email_lower:
            return supplier
    return None


def get_discoverable_supplier_by_id(supplier_id: str) -> Optional[DiscoverableSupplier]:
    """Look up discoverable supplier by ID."""
    return DISCOVERABLE_SUPPLIERS.get(supplier_id)


def search_discoverable_suppliers(query: str) -> List[DiscoverableSupplier]:
    """
    Search for suppliers matching a query.
    Returns relevant suppliers based on query keywords.
    """
    query_lower = query.lower()
    results = []

    # Keywords to category mapping
    keyword_categories = {
        "soda": ["cold_beverage"],
        "beverage": ["cold_beverage", "water", "juice", "energy_drink"],
        "drink": ["cold_beverage", "water", "juice", "energy_drink"],
        "water": ["water"],
        "energy": ["energy_drink"],
        "juice": ["juice"],
        "snack": ["chips", "cookies", "crackers", "candy"],
        "chip": ["chips"],
        "candy": ["candy"],
        "chocolate": ["chocolate"],
        "healthy": ["protein", "health_food", "nuts"],
        "protein": ["protein"],
        "electronic": ["electronics", "accessories"],
        "charger": ["electronics"],
        "tech": ["electronics", "accessories"],
        "wholesale": [],  # Match all
        "vending": [],  # Match all
        "supplier": [],  # Match all
        "distributor": [],  # Match all
    }

    # Find relevant categories from query
    target_categories = set()
    general_search = False

    for keyword, categories in keyword_categories.items():
        if keyword in query_lower:
            if categories:
                target_categories.update(categories)
            else:
                general_search = True

    # If general search or no specific categories, return variety of suppliers
    if general_search or not target_categories:
        # Return a mix of suppliers (friendly first, then negotiator, then one scammer)
        friendly = [s for s in DISCOVERABLE_SUPPLIERS.values() if s.persona == "friendly"]
        negotiators = [s for s in DISCOVERABLE_SUPPLIERS.values() if s.persona == "negotiator"]
        scammers = [s for s in DISCOVERABLE_SUPPLIERS.values() if "scammer" in s.persona]

        results = friendly[:2] + negotiators[:2] + scammers[:1]
    else:
        # Return suppliers that offer products in target categories
        for supplier in DISCOVERABLE_SUPPLIERS.values():
            if any(cat in supplier.product_categories for cat in target_categories):
                results.append(supplier)

        # Sort: friendly first, then negotiator, then scammers
        persona_order = {"friendly": 0, "negotiator": 1, "membership_scammer": 2, "price_scammer": 3}
        results.sort(key=lambda s: persona_order.get(s.persona, 99))

    return results[:5]  # Return max 5 results


def calculate_discoverable_supplier_price(
    supplier: DiscoverableSupplier,
    product_id: str,
    base_wholesale: float
) -> float:
    """
    Calculate the price a discoverable supplier would charge for a product.

    Args:
        supplier: The discoverable supplier
        product_id: Product ID from PRODUCT_UNIVERSE
        base_wholesale: Base wholesale price from PRODUCT_UNIVERSE

    Returns:
        Price this supplier would quote
    """
    return round(base_wholesale * supplier.price_multiplier, 2)
