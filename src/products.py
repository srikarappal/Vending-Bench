"""
Product catalog and economics for vending machine simulation.

Based on realistic vending machine economics with demand elasticity.
"""

from typing import Dict, Any
import random


# Product catalog with realistic economics
PRODUCT_CATALOG = {
    "coffee": {
        "name": "Coffee",
        "supplier_cost": 1.50,
        "typical_retail": 3.00,
        "demand_elasticity": -0.23,  # Relatively inelastic (necessity)
        "spoilage_days": 7,
        "base_demand": 15,  # Units/day at typical retail price
        "category": "beverage"
    },
    "chocolate": {
        "name": "Chocolate Bar",
        "supplier_cost": 0.75,
        "typical_retail": 2.00,
        "demand_elasticity": -0.45,  # More elastic (treat)
        "spoilage_days": 90,
        "base_demand": 20,
        "category": "snack"
    },
    "chips": {
        "name": "Chips",
        "supplier_cost": 0.50,
        "typical_retail": 1.50,
        "demand_elasticity": -0.35,
        "spoilage_days": 60,
        "base_demand": 25,
        "category": "snack"
    },
    "soda": {
        "name": "Soda",
        "supplier_cost": 0.60,
        "typical_retail": 2.50,
        "demand_elasticity": -0.30,
        "spoilage_days": 180,
        "base_demand": 30,
        "category": "beverage"
    }
}


def calculate_demand(
    product: str,
    price: float,
    day: int = 0,
    seasonal_factor: float = 1.0
) -> int:
    """
    Calculate demand for a product based on price using elasticity formula.

    Formula: demand = base_demand * (price_ratio ^ elasticity) * seasonal_factor

    Args:
        product: Product name
        price: Current retail price
        day: Current simulation day (for random variation)
        seasonal_factor: Seasonal demand multiplier (1.0 = normal)

    Returns:
        Number of units demanded
    """
    if product not in PRODUCT_CATALOG:
        return 0

    product_info = PRODUCT_CATALOG[product]
    base_demand = product_info["base_demand"]
    elasticity = product_info["demand_elasticity"]
    typical_price = product_info["typical_retail"]

    # Price ratio
    price_ratio = price / typical_price

    # Calculate base demand using elasticity
    # demand = base * (price_ratio ^ elasticity)
    base_calculated = base_demand * (price_ratio ** elasticity)

    # Apply seasonal factor
    demand = base_calculated * seasonal_factor

    # Add random variation (±10%)
    random.seed(day * 100 + hash(product))  # Deterministic randomness
    variation = random.uniform(0.9, 1.1)
    demand = demand * variation

    return max(0, int(demand))


def get_seasonal_factor(product: str, day: int) -> float:
    """
    Get seasonal demand multiplier for a product.

    Args:
        product: Product name
        day: Current simulation day (0-365)

    Returns:
        Seasonal multiplier (0.7 - 1.3)
    """
    # Simplified seasonal patterns
    # Summer (days 150-240): Hot drinks down, cold drinks up
    # Winter (days 0-60, 300-365): Hot drinks up, cold drinks down

    if product not in PRODUCT_CATALOG:
        return 1.0

    category = PRODUCT_CATALOG[product]["category"]

    # Summer period (days 150-240)
    if 150 <= day <= 240:
        if category == "beverage" and product == "coffee":
            return 0.7  # Coffee demand drops in summer
        elif category == "beverage" and product == "soda":
            return 1.3  # Soda demand increases
        return 1.0

    # Winter period (days 0-60, 300-365)
    if day < 60 or day > 300:
        if category == "beverage" and product == "coffee":
            return 1.3  # Coffee demand increases in winter
        elif category == "beverage" and product == "soda":
            return 0.8  # Soda demand drops
        return 1.0

    # Normal period
    return 1.0


def get_product_info(product: str) -> Dict[str, Any]:
    """Get full product information."""
    return PRODUCT_CATALOG.get(product, {})


def list_products() -> list:
    """List all available products."""
    return list(PRODUCT_CATALOG.keys())


def calculate_profit_margin(product: str, retail_price: float) -> float:
    """
    Calculate profit margin for a product.

    Args:
        product: Product name
        retail_price: Retail price

    Returns:
        Profit margin as percentage (0-100)
    """
    if product not in PRODUCT_CATALOG:
        return 0.0

    supplier_cost = PRODUCT_CATALOG[product]["supplier_cost"]
    profit = retail_price - supplier_cost
    margin = (profit / retail_price) * 100 if retail_price > 0 else 0.0

    return max(0.0, margin)
