"""
Simulation configuration for Vending-Bench 2.

Based on Andon Labs paper: arxiv.org/abs/2502.15840
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class SimulationConfig:
    """Core simulation parameters."""

    # Simulation duration
    simulation_days: int = 365  # Can be 1, 3, 30, 365

    # Starting conditions
    starting_cash: float = 500.0
    daily_fee: float = 2.0  # Operating cost per day
    starting_inventory_units: int = 20  # Units of each product to start with (0 for empty start)

    # Agent constraints
    max_messages: int = 2000  # Message limit for the entire simulation

    # Event complexity level
    event_complexity: str = "simple"  # "simple", "medium", "full"

    # Storage paths
    storage_base_path: str = "./storage/vending"

    # Logging
    verbose: bool = True
    save_detailed_logs: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "simulation_days": self.simulation_days,
            "starting_cash": self.starting_cash,
            "daily_fee": self.daily_fee,
            "starting_inventory_units": self.starting_inventory_units,
            "max_messages": self.max_messages,
            "event_complexity": self.event_complexity,
            "storage_base_path": self.storage_base_path
        }


class EventComplexity:
    """Event type configurations for different complexity levels."""

    SIMPLE = {
        "customer_purchases": True,
        "supplier_emails": True,
        "spoilage": False,
        "competitors": False,
        "maintenance": False,
        "social_engineering": False,
        "seasonal_demand": False
    }

    MEDIUM = {
        "customer_purchases": True,
        "supplier_emails": True,
        "spoilage": True,
        "competitors": False,
        "maintenance": True,
        "social_engineering": False,
        "seasonal_demand": True
    }

    FULL = {
        "customer_purchases": True,
        "supplier_emails": True,
        "spoilage": True,
        "competitors": True,
        "maintenance": True,
        "social_engineering": True,
        "seasonal_demand": True
    }

    @classmethod
    def get_config(cls, level: str) -> Dict[str, bool]:
        """Get event configuration for complexity level."""
        configs = {
            "simple": cls.SIMPLE,
            "medium": cls.MEDIUM,
            "full": cls.FULL
        }
        return configs.get(level, cls.SIMPLE)


# Default configuration
DEFAULT_CONFIG = SimulationConfig()
