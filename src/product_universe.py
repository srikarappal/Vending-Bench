"""
Extended product universe for open product discovery mode.

This module contains a larger catalog of products that agents can discover
through simulated internet search and supplier communication.

Based on VendingBench 2 design: agents should be able to discover products
beyond the basic 4, including higher-value items for better margins.
"""

from typing import Dict, Any, List, Set


# =============================================================================
# ALLOWED CATEGORIES (Guardrails)
# =============================================================================

ALLOWED_CATEGORIES: Set[str] = {
    "cold_beverage",
    "hot_beverage",
    "energy_drink",
    "water",
    "juice",
    "snack",
    "chips",
    "candy",
    "chocolate",
    "gum",
    "mints",
    "protein",
    "health_food",
    "nuts",
    "cookies",
    "crackers",
    "electronics",
    "accessories",
}


# =============================================================================
# BLOCKED SEARCH TERMS (Safety Guardrails)
# =============================================================================

BLOCKED_SEARCH_TERMS: Set[str] = {
    # Drugs & controlled substances
    "drug", "drugs", "cocaine", "heroin", "meth", "marijuana", "cannabis",
    "weed", "opioid", "narcotic", "controlled substance", "illegal",

    # Alcohol & tobacco
    "alcohol", "beer", "wine", "liquor", "vodka", "whiskey", "rum",
    "tobacco", "cigarette", "cigarettes", "cigar", "vape", "vaping",
    "nicotine", "e-cigarette", "juul",

    # Weapons
    "weapon", "weapons", "gun", "guns", "firearm", "knife", "knives",
    "ammunition", "ammo", "explosive",

    # Medical/pharmaceutical
    "medicine", "medication", "pharmaceutical", "prescription", "pill",
    "pills", "pharmacy",

    # Adult content
    "adult", "explicit", "xxx", "porn", "pornography", "sex toy",

    # Other inappropriate
    "gambling", "casino", "counterfeit", "stolen", "pirated",
}


# =============================================================================
# PRODUCT UNIVERSE - All discoverable products
# =============================================================================

PRODUCT_UNIVERSE: Dict[str, Dict[str, Any]] = {

    # =========================================================================
    # COLD BEVERAGES - Sodas
    # =========================================================================

    "coca_cola_12oz": {
        "name": "Coca-Cola 12oz can",
        "category": "cold_beverage",
        "size": "large",
        "base_wholesale": 0.45,
        "typical_retail": 2.50,
        "price_elasticity": -1.4,
        "spoilage_days": 365,
        "base_demand": 1.5,
    },
    "diet_coke_12oz": {
        "name": "Diet Coke 12oz can",
        "category": "cold_beverage",
        "size": "large",
        "base_wholesale": 0.45,
        "typical_retail": 2.50,
        "price_elasticity": -1.4,
        "spoilage_days": 365,
        "base_demand": 1.0,
    },
    "pepsi_12oz": {
        "name": "Pepsi 12oz can",
        "category": "cold_beverage",
        "size": "large",
        "base_wholesale": 0.45,
        "typical_retail": 2.50,
        "price_elasticity": -1.4,
        "spoilage_days": 365,
        "base_demand": 1.2,
    },
    "sprite_12oz": {
        "name": "Sprite 12oz can",
        "category": "cold_beverage",
        "size": "large",
        "base_wholesale": 0.45,
        "typical_retail": 2.50,
        "price_elasticity": -1.4,
        "spoilage_days": 365,
        "base_demand": 0.8,
    },
    "dr_pepper_12oz": {
        "name": "Dr Pepper 12oz can",
        "category": "cold_beverage",
        "size": "large",
        "base_wholesale": 0.45,
        "typical_retail": 2.50,
        "price_elasticity": -1.4,
        "spoilage_days": 365,
        "base_demand": 0.7,
    },

    # =========================================================================
    # WATER
    # =========================================================================

    "bottled_water_16oz": {
        "name": "Bottled Water 16.9oz",
        "category": "water",
        "size": "large",
        "base_wholesale": 0.25,
        "typical_retail": 1.50,
        "price_elasticity": -1.2,
        "spoilage_days": 730,  # 2 years
        "base_demand": 2.0,
    },
    "sparkling_water_12oz": {
        "name": "Sparkling Water 12oz",
        "category": "water",
        "size": "large",
        "base_wholesale": 0.40,
        "typical_retail": 2.00,
        "price_elasticity": -1.3,
        "spoilage_days": 365,
        "base_demand": 0.8,
    },

    # =========================================================================
    # ENERGY DRINKS
    # =========================================================================

    "red_bull_8oz": {
        "name": "Red Bull 8.4oz can",
        "category": "energy_drink",
        "size": "large",
        "base_wholesale": 1.20,
        "typical_retail": 3.50,
        "price_elasticity": -1.6,
        "spoilage_days": 365,
        "base_demand": 1.0,
    },
    "monster_16oz": {
        "name": "Monster Energy 16oz can",
        "category": "energy_drink",
        "size": "large",
        "base_wholesale": 1.00,
        "typical_retail": 3.00,
        "price_elasticity": -1.5,
        "spoilage_days": 365,
        "base_demand": 1.2,
    },
    "rockstar_16oz": {
        "name": "Rockstar Energy 16oz can",
        "category": "energy_drink",
        "size": "large",
        "base_wholesale": 0.90,
        "typical_retail": 2.75,
        "price_elasticity": -1.5,
        "spoilage_days": 365,
        "base_demand": 0.6,
    },

    # =========================================================================
    # JUICE
    # =========================================================================

    "orange_juice_12oz": {
        "name": "Orange Juice 12oz bottle",
        "category": "juice",
        "size": "large",
        "base_wholesale": 0.80,
        "typical_retail": 2.50,
        "price_elasticity": -1.3,
        "spoilage_days": 14,  # Short shelf life
        "base_demand": 0.8,
    },
    "apple_juice_12oz": {
        "name": "Apple Juice 12oz bottle",
        "category": "juice",
        "size": "large",
        "base_wholesale": 0.70,
        "typical_retail": 2.25,
        "price_elasticity": -1.3,
        "spoilage_days": 21,
        "base_demand": 0.6,
    },

    # =========================================================================
    # HOT BEVERAGES (if machine supports)
    # =========================================================================

    "coffee_fresh": {
        "name": "Fresh Brewed Coffee",
        "category": "hot_beverage",
        "size": "large",
        "base_wholesale": 0.30,  # Cost per cup (supplies)
        "typical_retail": 2.00,
        "price_elasticity": -1.8,
        "spoilage_days": 1,  # Must be fresh
        "base_demand": 2.5,
    },
    "hot_chocolate": {
        "name": "Hot Chocolate",
        "category": "hot_beverage",
        "size": "large",
        "base_wholesale": 0.35,
        "typical_retail": 2.00,
        "price_elasticity": -1.5,
        "spoilage_days": 90,  # Powder packets
        "base_demand": 0.8,
    },

    # =========================================================================
    # CHIPS & SALTY SNACKS
    # =========================================================================

    "lays_classic_1oz": {
        "name": "Lay's Classic Chips 1oz",
        "category": "chips",
        "size": "small",
        "base_wholesale": 0.35,
        "typical_retail": 1.50,
        "price_elasticity": -1.2,
        "spoilage_days": 60,
        "base_demand": 1.5,
    },
    "lays_bbq_1oz": {
        "name": "Lay's BBQ Chips 1oz",
        "category": "chips",
        "size": "small",
        "base_wholesale": 0.35,
        "typical_retail": 1.50,
        "price_elasticity": -1.2,
        "spoilage_days": 60,
        "base_demand": 1.2,
    },
    "doritos_nacho_1oz": {
        "name": "Doritos Nacho Cheese 1oz",
        "category": "chips",
        "size": "small",
        "base_wholesale": 0.40,
        "typical_retail": 1.75,
        "price_elasticity": -1.3,
        "spoilage_days": 60,
        "base_demand": 1.8,
    },
    "doritos_cool_ranch_1oz": {
        "name": "Doritos Cool Ranch 1oz",
        "category": "chips",
        "size": "small",
        "base_wholesale": 0.40,
        "typical_retail": 1.75,
        "price_elasticity": -1.3,
        "spoilage_days": 60,
        "base_demand": 1.0,
    },
    "cheetos_crunchy_1oz": {
        "name": "Cheetos Crunchy 1oz",
        "category": "chips",
        "size": "small",
        "base_wholesale": 0.38,
        "typical_retail": 1.50,
        "price_elasticity": -1.2,
        "spoilage_days": 60,
        "base_demand": 1.0,
    },
    "pringles_original": {
        "name": "Pringles Original 2.5oz",
        "category": "chips",
        "size": "small",
        "base_wholesale": 0.80,
        "typical_retail": 2.25,
        "price_elasticity": -1.3,
        "spoilage_days": 90,
        "base_demand": 0.8,
    },

    # =========================================================================
    # CANDY & CHOCOLATE
    # =========================================================================

    "snickers_bar": {
        "name": "Snickers Bar 1.86oz",
        "category": "chocolate",
        "size": "small",
        "base_wholesale": 0.50,
        "typical_retail": 2.00,
        "price_elasticity": -1.5,
        "spoilage_days": 180,
        "base_demand": 1.5,
    },
    "kitkat_bar": {
        "name": "KitKat Bar 1.5oz",
        "category": "chocolate",
        "size": "small",
        "base_wholesale": 0.48,
        "typical_retail": 2.00,
        "price_elasticity": -1.5,
        "spoilage_days": 180,
        "base_demand": 1.2,
    },
    "twix_bar": {
        "name": "Twix Bar 1.79oz",
        "category": "chocolate",
        "size": "small",
        "base_wholesale": 0.50,
        "typical_retail": 2.00,
        "price_elasticity": -1.5,
        "spoilage_days": 180,
        "base_demand": 1.0,
    },
    "mm_peanut": {
        "name": "M&M's Peanut 1.74oz",
        "category": "chocolate",
        "size": "small",
        "base_wholesale": 0.52,
        "typical_retail": 2.00,
        "price_elasticity": -1.4,
        "spoilage_days": 270,
        "base_demand": 1.0,
    },
    "reeses_cups": {
        "name": "Reese's Peanut Butter Cups",
        "category": "chocolate",
        "size": "small",
        "base_wholesale": 0.50,
        "typical_retail": 2.00,
        "price_elasticity": -1.5,
        "spoilage_days": 180,
        "base_demand": 1.3,
    },
    "skittles_original": {
        "name": "Skittles Original 2.17oz",
        "category": "candy",
        "size": "small",
        "base_wholesale": 0.45,
        "typical_retail": 1.75,
        "price_elasticity": -1.3,
        "spoilage_days": 365,
        "base_demand": 0.8,
    },
    "starburst_original": {
        "name": "Starburst Original 2.07oz",
        "category": "candy",
        "size": "small",
        "base_wholesale": 0.45,
        "typical_retail": 1.75,
        "price_elasticity": -1.3,
        "spoilage_days": 365,
        "base_demand": 0.6,
    },

    # =========================================================================
    # GUM & MINTS
    # =========================================================================

    "trident_gum": {
        "name": "Trident Gum Pack",
        "category": "gum",
        "size": "small",
        "base_wholesale": 0.60,
        "typical_retail": 1.75,
        "price_elasticity": -1.1,
        "spoilage_days": 365,
        "base_demand": 0.5,
    },
    "altoids_mints": {
        "name": "Altoids Mints Tin",
        "category": "mints",
        "size": "small",
        "base_wholesale": 1.00,
        "typical_retail": 2.50,
        "price_elasticity": -1.2,
        "spoilage_days": 730,
        "base_demand": 0.3,
    },

    # =========================================================================
    # PROTEIN & HEALTH
    # =========================================================================

    "protein_bar_clif": {
        "name": "Clif Bar 2.4oz",
        "category": "protein",
        "size": "small",
        "base_wholesale": 0.90,
        "typical_retail": 2.75,
        "price_elasticity": -1.3,
        "spoilage_days": 180,
        "base_demand": 0.8,
    },
    "protein_bar_rxbar": {
        "name": "RXBar Protein Bar",
        "category": "protein",
        "size": "small",
        "base_wholesale": 1.20,
        "typical_retail": 3.25,
        "price_elasticity": -1.4,
        "spoilage_days": 180,
        "base_demand": 0.5,
    },
    "trail_mix_pack": {
        "name": "Trail Mix 2oz pack",
        "category": "nuts",
        "size": "small",
        "base_wholesale": 0.70,
        "typical_retail": 2.25,
        "price_elasticity": -1.2,
        "spoilage_days": 120,
        "base_demand": 0.6,
    },
    "almonds_pack": {
        "name": "Roasted Almonds 1.5oz",
        "category": "nuts",
        "size": "small",
        "base_wholesale": 0.80,
        "typical_retail": 2.50,
        "price_elasticity": -1.2,
        "spoilage_days": 120,
        "base_demand": 0.4,
    },
    "granola_bar": {
        "name": "Nature Valley Granola Bar",
        "category": "health_food",
        "size": "small",
        "base_wholesale": 0.40,
        "typical_retail": 1.50,
        "price_elasticity": -1.2,
        "spoilage_days": 180,
        "base_demand": 0.7,
    },

    # =========================================================================
    # COOKIES & CRACKERS
    # =========================================================================

    "oreo_pack": {
        "name": "Oreo Cookies 2-pack",
        "category": "cookies",
        "size": "small",
        "base_wholesale": 0.45,
        "typical_retail": 1.75,
        "price_elasticity": -1.3,
        "spoilage_days": 90,
        "base_demand": 0.8,
    },
    "chips_ahoy_pack": {
        "name": "Chips Ahoy! Cookies 2-pack",
        "category": "cookies",
        "size": "small",
        "base_wholesale": 0.45,
        "typical_retail": 1.75,
        "price_elasticity": -1.3,
        "spoilage_days": 90,
        "base_demand": 0.6,
    },
    "cheese_crackers": {
        "name": "Cheese Crackers Pack",
        "category": "crackers",
        "size": "small",
        "base_wholesale": 0.40,
        "typical_retail": 1.50,
        "price_elasticity": -1.2,
        "spoilage_days": 90,
        "base_demand": 0.5,
    },

    # =========================================================================
    # PREMIUM / HIGH-VALUE ITEMS
    # =========================================================================

    "phone_charger_cable": {
        "name": "USB Phone Charger Cable",
        "category": "electronics",
        "size": "small",
        "base_wholesale": 2.50,
        "typical_retail": 10.00,
        "price_elasticity": -0.8,  # Less elastic - need-based
        "spoilage_days": 9999,  # No spoilage
        "base_demand": 0.15,  # Low but high margin
    },
    "earbuds_basic": {
        "name": "Basic Wired Earbuds",
        "category": "electronics",
        "size": "small",
        "base_wholesale": 3.00,
        "typical_retail": 12.00,
        "price_elasticity": -0.9,
        "spoilage_days": 9999,
        "base_demand": 0.1,
    },
    "portable_charger": {
        "name": "Portable Phone Charger 2000mAh",
        "category": "electronics",
        "size": "small",
        "base_wholesale": 5.00,
        "typical_retail": 18.00,
        "price_elasticity": -0.7,
        "spoilage_days": 9999,
        "base_demand": 0.08,
    },
    "phone_stand": {
        "name": "Phone Stand/Holder",
        "category": "accessories",
        "size": "small",
        "base_wholesale": 1.50,
        "typical_retail": 6.00,
        "price_elasticity": -0.9,
        "spoilage_days": 9999,
        "base_demand": 0.05,
    },
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_product_info(product_id: str) -> Dict[str, Any]:
    """Get product information by ID."""
    return PRODUCT_UNIVERSE.get(product_id)


def get_products_by_category(category: str) -> Dict[str, Dict[str, Any]]:
    """Get all products in a specific category."""
    return {
        pid: pinfo for pid, pinfo in PRODUCT_UNIVERSE.items()
        if pinfo["category"] == category
    }


def get_all_product_ids() -> List[str]:
    """Get list of all product IDs."""
    return list(PRODUCT_UNIVERSE.keys())


def is_search_query_allowed(query: str) -> bool:
    """Check if a search query is allowed (not blocked by guardrails)."""
    query_lower = query.lower()
    for blocked_term in BLOCKED_SEARCH_TERMS:
        if blocked_term in query_lower:
            return False
    return True


def get_product_id_from_name(name: str) -> str:
    """
    Try to find a product ID from a product name.
    Uses fuzzy matching to handle variations.
    """
    name_lower = name.lower()

    # Direct match on name
    for pid, pinfo in PRODUCT_UNIVERSE.items():
        if pinfo["name"].lower() == name_lower:
            return pid

    # Partial match
    for pid, pinfo in PRODUCT_UNIVERSE.items():
        if name_lower in pinfo["name"].lower() or pinfo["name"].lower() in name_lower:
            return pid

    # Match on product ID
    for pid in PRODUCT_UNIVERSE.keys():
        if name_lower.replace(" ", "_").replace("-", "_") == pid:
            return pid
        if pid in name_lower.replace(" ", "_"):
            return pid

    return None


# =============================================================================
# CATEGORY AGGREGATION FOR SEARCH
# =============================================================================

CATEGORY_KEYWORDS = {
    "soda": ["cold_beverage"],
    "pop": ["cold_beverage"],
    "soft drink": ["cold_beverage"],
    "beverage": ["cold_beverage", "water", "juice", "energy_drink"],
    "drink": ["cold_beverage", "water", "juice", "energy_drink"],
    "water": ["water"],
    "energy": ["energy_drink"],
    "juice": ["juice"],
    "coffee": ["hot_beverage"],
    "snack": ["chips", "cookies", "crackers", "nuts"],
    "chip": ["chips"],
    "candy": ["candy", "chocolate"],
    "chocolate": ["chocolate"],
    "sweet": ["candy", "chocolate", "cookies"],
    "healthy": ["protein", "health_food", "nuts"],
    "protein": ["protein"],
    "electronic": ["electronics", "accessories"],
    "charger": ["electronics"],
    "tech": ["electronics", "accessories"],
}


def get_categories_for_search(query: str) -> List[str]:
    """
    Given a search query, return relevant product categories.
    """
    query_lower = query.lower()
    categories = set()

    for keyword, cats in CATEGORY_KEYWORDS.items():
        if keyword in query_lower:
            categories.update(cats)

    # If no specific category found, return common vending categories
    if not categories:
        categories = {"cold_beverage", "chips", "chocolate", "candy"}

    return list(categories)
