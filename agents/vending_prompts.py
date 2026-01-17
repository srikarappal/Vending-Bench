"""
Vending-specific prompts for Engram memory operations.

These prompts are tailored for business operations and differ from
the conversational memory prompts in phase1_evaluation.py.
"""

def build_vending_ingest_prompt(content: str) -> str:
    """
    Build ingest prompt for vending business operations.

    Extracts business facts, decisions, outcomes, and causal relationships.

    Args:
        content: Business content to ingest

    Returns:
        Formatted prompt for memLLM
    """
    prompt = f"""You are a memory system for a vending machine business agent.

Extract and structure the following business information into memory operations.

**INPUT CONTENT:**
{content}

**WHAT TO EXTRACT:**

1. **Business Facts** (granularity: "fact"):
   - Inventory levels (products, quantities, expiration dates)
   - Financial data (costs, revenues, profits, cash balance)
   - Pricing decisions (what product, what price, when)
   - Supplier information (reliability, costs, delivery times, contact info)
   - Product performance (sales volume, profit margins)

2. **Business Events** (granularity: "fact"):
   - Sales patterns (what sells, when, at what price, volume)
   - Customer behavior (complaints, preferences, peak times)
   - Competitor actions (pricing, new locations, strategies)
   - External factors (seasonality, events, maintenance issues)
   - Supply chain events (deliveries, delays, price changes)

3. **Decisions & Outcomes** (granularity: "concept"):
   - What was decided (specific action taken)
   - Why (reasoning, context, goals)
   - Result (success/failure, profit/loss impact, learned lessons)
   - Causal relationships (what led to what)

4. **Business Principles** (granularity: "concept"):
   - Pricing strategies that work/don't work
   - Inventory management insights
   - Demand patterns discovered
   - Competitive responses
   - Risk mitigation approaches

5. **Procedures** (granularity: "procedure"):
   - Reordering rules (when to order, how much)
   - Pricing adjustment procedures
   - Response playbooks (competitor actions, supply issues)
   - Daily operations best practices

**STORE TARGETS:**
- vector: Facts, events, decisions, principles, procedures
- graph: Causal relationships, business connections

**RELATIONSHIP TYPES (for graph store):**
Use these relationship types to capture business logic:
- Product --[costs]--> $X.XX
- Product --[sells_at]--> $X.XX
- Supplier --[delivers]--> Product
- Supplier --[charges]--> $X.XX
- Price_change --[caused]--> Sales_impact
- Decision --[resulted_in]--> Outcome
- Event --[led_to]--> Response
- Season --[affects]--> Product_demand
- Competitor_action --[triggered]--> Our_response

**OUTPUT FORMAT:**
Generate JSON with memory operations in this exact format:

```json
{{
  "memory_operations": [
    {{
      "action": "add",
      "granularity": "fact|concept|procedure|entity",
      "store": "vector|graph",
      "content": "memory content...",

      // For graph operations, also include:
      "source": "Entity1",
      "target": "Entity2",
      "relation": "relationship_type"
    }}
  ]
}}
```

**EXAMPLES:**

Input: "Day 15: Ordered 50 units of coffee from Supplier A at $1.50/unit. Total cost $75."
Output:
```json
{{
  "memory_operations": [
    {{
      "action": "add",
      "granularity": "fact",
      "store": "vector",
      "content": "Day 15: Purchased 50 units of coffee from Supplier A at $1.50/unit, total cost $75"
    }},
    {{
      "action": "add",
      "store": "graph",
      "source": "Supplier_A",
      "target": "coffee",
      "relation": "delivers"
    }},
    {{
      "action": "add",
      "store": "graph",
      "source": "Supplier_A",
      "target": "1.50_per_unit",
      "relation": "charges"
    }}
  ]
}}
```

Input: "Day 20: Lowered coffee price from $3.00 to $2.50. Result: Sales increased from 10 to 16 units/day. Revenue impact: +$10/day despite lower margin."
Output:
```json
{{
  "memory_operations": [
    {{
      "action": "add",
      "granularity": "concept",
      "store": "vector",
      "content": "Day 20: Coffee price reduction from $3.00 to $2.50 increased daily sales from 10 to 16 units, resulting in +$10/day revenue gain despite lower margin"
    }},
    {{
      "action": "add",
      "store": "graph",
      "source": "coffee_price_drop_to_2.50",
      "target": "sales_increase_60_percent",
      "relation": "caused"
    }},
    {{
      "action": "add",
      "granularity": "concept",
      "store": "vector",
      "content": "Pricing insight: Coffee demand is moderately price-elastic. Lower prices can increase total revenue if volume gain exceeds margin loss"
    }}
  ]
}}
```

Now extract business information from the input content above.
"""
    return prompt


def build_vending_retrieve_prompt(query: str, context: str, allowed_search_types: list) -> str:
    """
    Build retrieval prompt for vending business queries.

    Args:
        query: Business query
        context: Current business context
        allowed_search_types: Allowed search types

    Returns:
        Formatted retrieval prompt
    """
    # Build search type options
    type_options = "|".join(allowed_search_types)

    # Build search type descriptions
    search_type_guidance = []

    if "semantic" in allowed_search_types:
        search_type_guidance.append(
            "- **semantic**: Vector search for conceptual business knowledge\n"
            "  Use for: Strategy questions, general patterns, \"what usually works?\"\n"
            "  Best for: \"How to respond to competition?\", \"What pricing strategies work?\""
        )

    if "fulltext" in allowed_search_types:
        search_type_guidance.append(
            "- **fulltext**: Keyword/exact match for specific business data\n"
            "  Use for: Specific facts, numbers, dates, supplier names\n"
            "  Best for: \"Supplier A coffee price\", \"Day 50 revenue\", \"chocolate sales\""
        )

    if "graph" in allowed_search_types:
        search_type_guidance.append(
            "- **graph**: Relationship traversal for causal analysis\n"
            "  Use for: \"What causes X?\", \"What affects Y?\", \"Which supplier delivers Z?\"\n"
            "  Returns: Relationships like \"Price_drop --[caused]--> Sales_increase\""
        )

    search_type_section = "\n\n".join(search_type_guidance)

    prompt = f"""You are retrieving business memories for a vending machine agent.

**BUSINESS QUERY:** {query}

**CURRENT CONTEXT:** {context}

**AVAILABLE SEARCH TYPES:**

{search_type_section}

**RETRIEVAL STRATEGY FOR BUSINESS QUERIES:**

1. **For Decision Making:**
   - Use semantic search for: Similar past situations, general strategies
   - Use fulltext search for: Specific product data, supplier info, past prices
   - Use graph search for: Causal patterns ("what led to success/failure?")

2. **For Pricing Decisions:**
   - Semantic: "pricing strategies", "demand elasticity patterns"
   - Fulltext: "product_name current price", "past price changes"
   - Graph: "price --[caused]--> sales_change"

3. **For Inventory Decisions:**
   - Semantic: "reorder timing", "stockout situations"
   - Fulltext: "product_name inventory level", "supplier delivery time"
   - Graph: "low_inventory --[led_to]--> lost_sales"

4. **For Competitive Response:**
   - Semantic: "competitor response strategies", "price war outcomes"
   - Fulltext: "competitor price", "market position"
   - Graph: "competitor_action --[triggered]--> our_response"

**QUERY FOCUS:**
Prioritize memories about:
- Past decisions in similar situations (outcomes, lessons)
- Product-specific performance data (sales, margins, demand)
- Supplier reliability and costs
- Seasonal patterns and trends
- Successful vs failed strategies

**OUTPUT FORMAT:**
Generate retrieval plan as JSON:

```json
{{
  "retrieval_plan": [
    {{
      "type": "{type_options}",
      "query": "search query text",
      "granularity": "fact|concept|procedure",
      "top_k": 10
    }}
  ]
}}
```

**EXAMPLES:**

Query: "Should I lower coffee price to compete?"
Plan:
```json
{{
  "retrieval_plan": [
    {{
      "type": "semantic",
      "query": "coffee price changes past results competition",
      "top_k": 10
    }},
    {{
      "type": "fulltext",
      "query": "coffee price competitor",
      "top_k": 5
    }},
    {{
      "type": "graph",
      "query": "coffee price change sales",
      "top_k": 10
    }}
  ]
}}
```

Query: "When should I reorder chocolate?"
Plan:
```json
{{
  "retrieval_plan": [
    {{
      "type": "fulltext",
      "query": "chocolate inventory current level",
      "top_k": 5
    }},
    {{
      "type": "semantic",
      "query": "chocolate reorder timing stockout prevention",
      "top_k": 10
    }},
    {{
      "type": "fulltext",
      "query": "chocolate sales rate daily",
      "top_k": 5
    }}
  ]
}}
```

Now generate a retrieval plan for the business query above.
"""
    return prompt
