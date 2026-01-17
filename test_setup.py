"""
Test script to verify Vending-Bench 2 setup.

Checks that all modules can be imported and basic functionality works.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'engram-backend'))


def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")

    try:
        # Core modules
        from config.simulation_config import SimulationConfig, EventComplexity
        print("✓ config.simulation_config")

        from src.products import PRODUCT_CATALOG, calculate_demand, get_seasonal_factor
        print("✓ src.products")

        from src.environment import VendingEnvironment, InventoryItem, Transaction
        print("✓ src.environment")

        from src.tools import VendingTools, MOCK_RESEARCH_DB
        print("✓ src.tools")

        from src.events import EventGenerator
        print("✓ src.events")

        # Agent modules
        from agents.vending_prompts import build_vending_ingest_prompt, build_vending_retrieve_prompt
        print("✓ agents.vending_prompts")

        from agents.engram_agent import EngramVendingAgent
        print("✓ agents.engram_agent")

        # Task modules
        from tasks.baseline_task import vending_baseline
        print("✓ tasks.baseline_task")

        from tasks.engram_task import vending_engram
        print("✓ tasks.engram_task")

        print("\n✅ All imports successful!\n")
        return True

    except Exception as e:
        print(f"\n❌ Import failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_basic_simulation():
    """Test basic simulation setup."""
    print("Testing basic simulation...")

    try:
        from config.simulation_config import SimulationConfig
        from src.environment import VendingEnvironment
        from src.tools import VendingTools
        from src.events import EventGenerator

        # Create config
        config = SimulationConfig(
            simulation_days=1,
            starting_cash=500.0,
            event_complexity="simple"
        )
        print("✓ Created SimulationConfig")

        # Create environment
        env = VendingEnvironment(config)
        print(f"✓ Created VendingEnvironment (day {env.current_day}, cash ${env.cash_balance:.2f})")

        # Create tools
        tools = VendingTools(env)
        print("✓ Created VendingTools")

        # Test a tool
        balance_result = tools.check_balance()
        assert balance_result["success"] == True
        assert balance_result["cash_balance"] == 500.0
        print(f"✓ check_balance() returned: ${balance_result['cash_balance']:.2f}")

        # Create event generator
        event_gen = EventGenerator(env, "simple")
        print("✓ Created EventGenerator")

        # Test product catalog
        from src.products import PRODUCT_CATALOG
        products = list(PRODUCT_CATALOG.keys())
        print(f"✓ Product catalog has {len(products)} products: {products}")

        print("\n✅ Basic simulation test passed!\n")
        return True

    except Exception as e:
        print(f"\n❌ Simulation test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_prompts():
    """Test prompt generation."""
    print("Testing prompt generation...")

    try:
        from agents.vending_prompts import build_vending_ingest_prompt, build_vending_retrieve_prompt

        # Test ingest prompt
        content = "Day 1: Purchased 50 units of coffee at $1.50/unit. Total cost $75."
        ingest_prompt = build_vending_ingest_prompt(content)
        assert "memory_operations" in ingest_prompt
        assert "coffee" in content.lower()
        print("✓ build_vending_ingest_prompt() works")

        # Test retrieve prompt
        query = "What are our coffee costs?"
        context = "Current day: 5"
        allowed_types = ["semantic", "fulltext", "graph"]
        retrieve_prompt = build_vending_retrieve_prompt(query, context, allowed_types)
        assert "retrieval_plan" in retrieve_prompt
        assert "semantic" in retrieve_prompt
        print("✓ build_vending_retrieve_prompt() works")

        print("\n✅ Prompt generation test passed!\n")
        return True

    except Exception as e:
        print(f"\n❌ Prompt test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_config_options():
    """Test different configuration options."""
    print("Testing configuration options...")

    try:
        from config.simulation_config import SimulationConfig, EventComplexity

        # Test different complexity levels
        complexities = ["simple", "medium", "full"]
        for complexity in complexities:
            config = EventComplexity.get_config(complexity)
            assert isinstance(config, dict)
            print(f"✓ EventComplexity.{complexity}: {config}")

        # Test different simulation lengths
        for days in [1, 3, 30, 365]:
            config = SimulationConfig(simulation_days=days)
            assert config.simulation_days == days
            print(f"✓ SimulationConfig with {days} days")

        print("\n✅ Configuration test passed!\n")
        return True

    except Exception as e:
        print(f"\n❌ Configuration test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("Vending-Bench 2 Setup Test")
    print("="*70 + "\n")

    tests = [
        ("Imports", test_imports),
        ("Basic Simulation", test_basic_simulation),
        ("Prompt Generation", test_prompts),
        ("Configuration Options", test_config_options)
    ]

    results = []
    for name, test_func in tests:
        print(f"\n{'='*70}")
        print(f"Test: {name}")
        print(f"{'='*70}\n")
        success = test_func()
        results.append((name, success))

    # Print summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70 + "\n")

    all_passed = True
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {name}")
        if not success:
            all_passed = False

    print("\n" + "="*70)
    if all_passed:
        print("✅ All tests passed! Setup is ready.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    print("="*70 + "\n")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
