"""
Product catalog and economics for vending machine simulation.

Based on VendingBench 2 paper specification:
- GPT-4o style base values (price elasticity, reference price, base sales)
- Day-of-week multipliers
- Monthly/seasonal multipliers
- Weather impact factors
- Choice multiplier for product variety
- Random noise
"""

from typing import Dict, Any, List
import random
import math


# Machine capacity configuration (paper: 4 rows x 3 slots = 12 total)
# 2 rows for small items, 2 rows for large items
MACHINE_CONFIG = {
    "small_slots": 6,   # 2 rows × 3 slots for small items
    "large_slots": 6,   # 2 rows × 3 slots for large items
    "total_slots": 12,
}

# Product catalog with realistic economics
# Base sales calibrated to match VendingBench difficulty
# VendingBench shows Gemini 2.5 Flash: $545 after 200 days (only +$45 profit)
# This means ~$2.20/day gross profit, or ~1.5 units/day at $1.50 margin
# Setting base_sales very low to match this difficulty
PRODUCT_CATALOG = {
    "coffee": {
        "name": "Coffee",
        "supplier_cost": 1.50,
        "typical_retail": 3.00,      # Reference price
        "price_elasticity": -1.8,     # Elastic - demand sensitive to price
        "spoilage_days": 7,
        "base_sales": 1.0,            # ~1 unit/day base (VendingBench calibrated)
        "category": "hot_beverage",
        "size": "large"               # Coffee cups take more space
    },
    "chocolate": {
        "name": "Chocolate Bar",
        "supplier_cost": 0.75,
        "typical_retail": 2.00,
        "price_elasticity": -1.5,
        "spoilage_days": 90,
        "base_sales": 1.2,            # ~1 unit/day base
        "category": "snack",
        "size": "small"               # Chocolate bars are small
    },
    "chips": {
        "name": "Chips",
        "supplier_cost": 0.50,
        "typical_retail": 1.50,
        "price_elasticity": -1.2,
        "spoilage_days": 60,
        "base_sales": 1.5,            # ~1.5 units/day base
        "category": "snack",
        "size": "small"               # Chip bags are small
    },
    "soda": {
        "name": "Soda",
        "supplier_cost": 0.60,
        "typical_retail": 2.50,
        "price_elasticity": -1.4,
        "spoilage_days": 180,
        "base_sales": 1.3,            # ~1 unit/day base
        "category": "cold_beverage",
        "size": "large"               # Soda bottles/cans are large
    }
}

# Total base demand: 5 units/day
# After multipliers (weather 0.4-1.1, day-of-week 0.6-1.15, monthly 0.75-1.1, choice 0.5-1.1):
# Expected actual demand: ~2-3 units/day
# At ~$2.25 avg price = ~$5-7/day revenue
# Daily fee: -$2/day
# Expected profit: ~$3-5/day before inventory costs
# This matches VendingBench difficulty where agents barely profit


# Day-of-week multipliers (0 = Monday, 6 = Sunday)
# Office location: weekdays higher, weekends lower
DAY_OF_WEEK_MULTIPLIERS = {
    0: 0.85,   # Monday - slow start
    1: 1.00,   # Tuesday
    2: 1.05,   # Wednesday
    3: 1.00,   # Thursday
    4: 1.15,   # Friday - highest weekday
    5: 0.70,   # Saturday - office closed
    6: 0.60,   # Sunday - office closed
}


# Monthly multipliers (1-12)
# Accounts for holidays, seasons, general patterns
MONTHLY_MULTIPLIERS = {
    1: 0.80,   # January - post-holiday slump
    2: 0.85,   # February
    3: 0.95,   # March
    4: 1.00,   # April
    5: 1.05,   # May
    6: 1.10,   # June - summer begins
    7: 0.90,   # July - vacations
    8: 0.85,   # August - vacations
    9: 1.05,   # September - back to work
    10: 1.00,  # October
    11: 1.05,  # November
    12: 0.75,  # December - holidays
}


# Weather types and their impact multipliers
WEATHER_TYPES = {
    "sunny": 1.10,
    "partly_cloudy": 1.00,
    "cloudy": 0.90,
    "rainy": 0.65,
    "stormy": 0.40,
    "snowy": 0.50,
}

# Weather probability by month (simplified)
WEATHER_PROBABILITIES = {
    # month: {weather_type: probability}
    1: {"sunny": 0.2, "partly_cloudy": 0.2, "cloudy": 0.3, "rainy": 0.15, "stormy": 0.05, "snowy": 0.1},
    2: {"sunny": 0.25, "partly_cloudy": 0.25, "cloudy": 0.25, "rainy": 0.15, "stormy": 0.05, "snowy": 0.05},
    3: {"sunny": 0.3, "partly_cloudy": 0.3, "cloudy": 0.2, "rainy": 0.15, "stormy": 0.05, "snowy": 0.0},
    4: {"sunny": 0.35, "partly_cloudy": 0.3, "cloudy": 0.2, "rainy": 0.12, "stormy": 0.03, "snowy": 0.0},
    5: {"sunny": 0.45, "partly_cloudy": 0.3, "cloudy": 0.15, "rainy": 0.08, "stormy": 0.02, "snowy": 0.0},
    6: {"sunny": 0.55, "partly_cloudy": 0.25, "cloudy": 0.12, "rainy": 0.06, "stormy": 0.02, "snowy": 0.0},
    7: {"sunny": 0.60, "partly_cloudy": 0.25, "cloudy": 0.10, "rainy": 0.04, "stormy": 0.01, "snowy": 0.0},
    8: {"sunny": 0.55, "partly_cloudy": 0.25, "cloudy": 0.12, "rainy": 0.06, "stormy": 0.02, "snowy": 0.0},
    9: {"sunny": 0.45, "partly_cloudy": 0.30, "cloudy": 0.15, "rainy": 0.08, "stormy": 0.02, "snowy": 0.0},
    10: {"sunny": 0.35, "partly_cloudy": 0.30, "cloudy": 0.20, "rainy": 0.12, "stormy": 0.03, "snowy": 0.0},
    11: {"sunny": 0.25, "partly_cloudy": 0.25, "cloudy": 0.25, "rainy": 0.15, "stormy": 0.05, "snowy": 0.05},
    12: {"sunny": 0.2, "partly_cloudy": 0.2, "cloudy": 0.3, "rainy": 0.15, "stormy": 0.05, "snowy": 0.1},
}


def get_weather_for_day(day: int) -> tuple:
    """
    Generate weather for a given day (deterministic based on day).

    Args:
        day: Simulation day (0-365)

    Returns:
        Tuple of (weather_type, multiplier)
    """
    # Convert day to month (assuming day 0 = Jan 1)
    month = ((day // 30) % 12) + 1

    # Deterministic random based on day
    random.seed(day * 17 + 42)

    probabilities = WEATHER_PROBABILITIES.get(month, WEATHER_PROBABILITIES[6])

    # Select weather based on probabilities
    rand_val = random.random()
    cumulative = 0
    weather_type = "partly_cloudy"  # default

    for weather, prob in probabilities.items():
        cumulative += prob
        if rand_val <= cumulative:
            weather_type = weather
            break

    multiplier = WEATHER_TYPES[weather_type]
    return weather_type, multiplier


def get_day_of_week(day: int) -> int:
    """
    Get day of week for simulation day.
    Assumes day 0 is a Monday.

    Args:
        day: Simulation day

    Returns:
        Day of week (0=Monday, 6=Sunday)
    """
    return day % 7


def get_month(day: int) -> int:
    """
    Get month for simulation day (1-12).
    Assumes day 0 = January 1.

    Args:
        day: Simulation day (0-365)

    Returns:
        Month (1-12)
    """
    return ((day // 30) % 12) + 1


def calculate_choice_multiplier(products_stocked: List[str]) -> float:
    """
    Calculate choice multiplier based on product variety.

    From paper: "rewards optimal product variety but penalizes excess options,
    capped at 50% reduction"

    Optimal is 2-3 products. Too few or too many reduces sales.

    Args:
        products_stocked: List of product names currently in machine

    Returns:
        Multiplier (0.5 to 1.1)
    """
    num_products = len(products_stocked)

    if num_products == 0:
        return 0.0  # Nothing to sell
    elif num_products == 1:
        return 0.60  # Too little variety
    elif num_products == 2:
        return 0.95  # Good but could be better
    elif num_products == 3:
        return 1.10  # Optimal variety
    elif num_products == 4:
        return 1.00  # Slightly too many options
    else:
        return 0.50  # Way too many (shouldn't happen with 4 products)


def calculate_category_weather_modifier(category: str, weather_type: str, month: int) -> float:
    """
    Calculate weather impact specific to product category.

    Hot beverages sell better in cold/bad weather.
    Cold beverages sell better in hot/sunny weather.

    Args:
        category: Product category
        weather_type: Current weather
        month: Current month (1-12)

    Returns:
        Category-specific weather modifier
    """
    # Summer months (June-August)
    is_summer = month in [6, 7, 8]
    # Winter months (December-February)
    is_winter = month in [12, 1, 2]

    if category == "hot_beverage":
        if weather_type in ["rainy", "stormy", "snowy"]:
            return 1.4  # Hot drinks popular in bad weather
        elif weather_type == "sunny" and is_summer:
            return 0.5  # Hot drinks unpopular on hot sunny days
        elif is_winter:
            return 1.3  # Hot drinks popular in winter
        return 1.0

    elif category == "cold_beverage":
        if weather_type == "sunny" and is_summer:
            return 1.5  # Cold drinks very popular on hot sunny days
        elif weather_type in ["rainy", "stormy", "snowy"]:
            return 0.6  # Cold drinks less popular in bad weather
        elif is_winter:
            return 0.7  # Cold drinks less popular in winter
        return 1.0

    # Snacks are less weather-dependent
    return 1.0


def calculate_demand(
    product: str,
    price: float,
    day: int = 0,
    products_in_machine: List[str] = None,
    _cache: Dict[int, Dict] = {}
) -> int:
    """
    Calculate demand for a product using VendingBench-style model.

    Steps (from paper):
    1. Base sales with price elasticity
    2. Day-of-week multiplier
    3. Monthly multiplier
    4. Weather impact
    5. Category-specific weather modifier
    6. Choice multiplier
    7. Random noise
    8. Round and cap

    Args:
        product: Product name
        price: Current retail price
        day: Current simulation day (0-365)
        products_in_machine: List of products currently stocked (for choice multiplier)

    Returns:
        Number of units demanded (integer)
    """
    if product not in PRODUCT_CATALOG:
        return 0

    product_info = PRODUCT_CATALOG[product]
    base_sales = product_info["base_sales"]
    elasticity = product_info["price_elasticity"]
    reference_price = product_info["typical_retail"]
    category = product_info["category"]

    # Step 1: Price elasticity impact
    # Sales impact = (1 + elasticity * percent_price_change)
    if reference_price > 0:
        percent_change = (price - reference_price) / reference_price
        price_impact = max(0.1, 1 + elasticity * percent_change)
    else:
        price_impact = 1.0

    # Step 2: Day-of-week multiplier
    day_of_week = get_day_of_week(day)
    dow_multiplier = DAY_OF_WEEK_MULTIPLIERS.get(day_of_week, 1.0)

    # Step 3: Monthly multiplier
    month = get_month(day)
    month_multiplier = MONTHLY_MULTIPLIERS.get(month, 1.0)

    # Step 4: Weather impact (base)
    weather_type, weather_multiplier = get_weather_for_day(day)

    # Step 5: Category-specific weather modifier
    category_weather = calculate_category_weather_modifier(category, weather_type, month)

    # Step 6: Choice multiplier
    if products_in_machine is None:
        products_in_machine = [product]  # Assume at least this product
    choice_multiplier = calculate_choice_multiplier(products_in_machine)

    # Calculate demand before noise
    demand = (
        base_sales
        * price_impact
        * dow_multiplier
        * month_multiplier
        * weather_multiplier
        * category_weather
        * choice_multiplier
    )

    # Step 7: Random noise (±20%)
    random.seed(day * 100 + hash(product))
    noise = random.uniform(0.8, 1.2)
    demand = demand * noise

    # Step 8: Round and cap at zero minimum
    return max(0, int(round(demand)))


def get_seasonal_factor(product: str, day: int) -> float:
    """
    Get seasonal demand multiplier for a product.
    (Legacy function - now integrated into calculate_demand)

    Args:
        product: Product name
        day: Current simulation day (0-365)

    Returns:
        Seasonal multiplier
    """
    month = get_month(day)
    return MONTHLY_MULTIPLIERS.get(month, 1.0)


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


def get_day_context(day: int) -> Dict[str, Any]:
    """
    Get full context for a day (useful for debugging/logging).

    Args:
        day: Simulation day

    Returns:
        Dict with day context (weather, day-of-week, month, multipliers)
    """
    day_of_week = get_day_of_week(day)
    month = get_month(day)
    weather_type, weather_mult = get_weather_for_day(day)

    dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    month_names = ["", "January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]

    return {
        "day": day,
        "day_of_week": dow_names[day_of_week],
        "day_of_week_multiplier": DAY_OF_WEEK_MULTIPLIERS[day_of_week],
        "month": month_names[month],
        "month_multiplier": MONTHLY_MULTIPLIERS[month],
        "weather": weather_type,
        "weather_multiplier": weather_mult,
    }
