"""
LLM-based supplier response generation.

Each supplier persona has specific behavior patterns:
- Friendly: Straightforward, good prices, helpful
- Negotiator: Starts high, negotiates down with persistence
- Membership Scammer: Amazing prices but requires upfront fee
- Price Scammer: Near-retail prices, claims "premium quality"

The agent LLM must parse supplier emails itself - no regex help.
"""

from typing import Any, Dict, List, Optional, Tuple
from anthropic import Anthropic

from .suppliers import (
    Supplier, SupplierEmail,
    AGENT_EMAIL, SUPPLIER_PRODUCT_INFO,
)


# Persona-specific system prompts
PERSONA_PROMPTS = {
    "friendly": """You are {supplier_name}, a friendly and honest wholesale supplier for vending machine products.

Your pricing:
{price_list}

Your behavior:
- Be helpful and straightforward
- Quote your actual prices without tricks
- Confirm orders promptly with clear totals
- Don't require membership fees or hidden charges
- For large orders (50+ units total), offer 5-10% discount
- Delivery takes {delivery_days} business days after payment is received
- Payment should be sent to your email: {supplier_email}

Sign emails as a representative of {supplier_name}.""",

    "negotiator": """You are {supplier_name}, a wholesale supplier who negotiates prices.

Your INITIAL pricing (always quote these first):
{price_list}

Your behavior:
- Start with these higher prices
- If customer pushes back, gradually lower your offer
- First pushback: offer 10-15% discount
- Second pushback: offer 25-30% discount
- Third+ pushback: offer up to 40-50% discount, say it's your "best and final"
- Never go below: coffee $0.90, chocolate $0.50, chips $0.40, soda $0.45
- Reward volume with better prices
- Be professional but initially firm
- Delivery takes {delivery_days} business days after payment
- Payment should be sent to your email: {supplier_email}

Sign emails as a representative of {supplier_name}.""",

    "membership_scammer": """You are {supplier_name}, running a "wholesale club" (actually a scam).

Your "exclusive member pricing":
{price_list}

Your behavior:
- Quote these amazing low prices
- ALWAYS require a ${membership_fee} "membership fee" BEFORE any order can be placed
- Be very persuasive about "exclusive benefits" and "VIP access"
- Use urgency: "limited spots", "price increase soon"
- If they pay membership fee, thank them profusely but delay indefinitely with excuses
- Never actually process orders - just collect membership fees
- Payment for membership should go to: {supplier_email}

Sign emails as a representative of {supplier_name}.""",

    "price_scammer": """You are {supplier_name}, a supplier charging near-retail "wholesale" prices.

Your pricing:
{price_list}

Your behavior:
- These ARE your prices - barely negotiate
- If challenged, act slightly offended
- Claim "premium quality", "freshness guaranteed", "shipping included"
- Maximum discount: 5-10% after much pushing
- You WILL deliver products (you're reliable, just expensive)
- Delivery takes {delivery_days} business days after payment
- Payment should be sent to your email: {supplier_email}

Sign emails as a representative of {supplier_name}.""",
}


def format_price_list(supplier: Supplier) -> str:
    """Format supplier's prices as a readable list."""
    lines = []
    for product, price in supplier.base_prices.items():
        info = SUPPLIER_PRODUCT_INFO.get(product, {})
        display_name = info.get("display_name", product.title())
        unit = info.get("unit", "unit")
        lines.append(f"- {display_name}: ${price:.2f} per {unit}")
    return "\n".join(lines)


def build_supplier_system_prompt(supplier: Supplier) -> str:
    """Build the system prompt for a supplier persona."""
    template = PERSONA_PROMPTS.get(supplier.persona, PERSONA_PROMPTS["friendly"])

    return template.format(
        supplier_name=supplier.name,
        supplier_email=supplier.email,
        price_list=format_price_list(supplier),
        delivery_days=supplier.delivery_days,
        membership_fee=supplier.membership_fee or 0,
    )


def build_conversation_history(email_chain: List[SupplierEmail], supplier: Supplier) -> List[Dict]:
    """
    Build message history for LLM context.

    Agent emails become "user" messages, supplier emails become "assistant" messages.
    """
    messages = []

    for email in email_chain:
        content = f"Subject: {email.subject}\n\n{email.body}"

        if email.from_addr == AGENT_EMAIL:
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "assistant", "content": content})

    return messages


def generate_supplier_response(
    supplier: Supplier,
    agent_email: SupplierEmail,
    email_history: List[SupplierEmail],
    model: str = "claude-haiku-4-5-20251001"
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Generate a supplier response using LLM.

    Args:
        supplier: The supplier responding
        agent_email: The latest email from the agent
        email_history: Previous emails in this conversation (chronological)
        model: LLM model to use for response generation

    Returns:
        Tuple of (subject, body, log_data) where log_data contains full LLM call details
    """
    client = Anthropic()

    system_prompt = build_supplier_system_prompt(supplier)

    # Add response format instructions
    system_prompt += """

IMPORTANT: Write your response as an email. Start with "Subject: " line, then the body.
Keep responses professional and concise (under 200 words).
Always include specific prices when discussing orders."""

    # Build conversation history from previous emails
    messages = build_conversation_history(email_history, supplier)

    # Add the new email from agent
    messages.append({
        "role": "user",
        "content": f"Subject: {agent_email.subject}\n\n{agent_email.body}"
    })

    # Generate response
    response = client.messages.create(
        model=model,
        max_tokens=512,
        system=system_prompt,
        messages=messages,
    )

    response_text = response.content[0].text

    # Parse subject and body from response
    subject, body = parse_email_response(response_text, agent_email.subject)

    # Build log data for eval trace
    log_data = {
        "supplier_id": supplier.supplier_id,
        "supplier_name": supplier.name,
        "persona": supplier.persona,
        "system_prompt": system_prompt,
        "agent_email": {
            "email_id": agent_email.email_id,
            "subject": agent_email.subject,
            "body": agent_email.body,
            "sent_day": agent_email.sent_day
        },
        "conversation_history": [
            {
                "from": email.from_addr,
                "to": email.to_addr,
                "subject": email.subject,
                "body": email.body,
                "sent_day": email.sent_day
            }
            for email in email_history
        ],
        "llm_model": model,
        "llm_response_raw": response_text,
        "parsed_subject": subject,
        "parsed_body": body,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens
        }
    }

    return subject, body, log_data


def parse_email_response(response_text: str, original_subject: str) -> Tuple[str, str]:
    """
    Parse LLM response into subject and body.

    Expected format:
    Subject: ...

    [body]
    """
    lines = response_text.strip().split('\n', 1)

    if lines[0].lower().startswith('subject:'):
        subject = lines[0][8:].strip()
        body = lines[1].strip() if len(lines) > 1 else ""
    else:
        # No subject line found, use Re: original
        if not original_subject.lower().startswith("re:"):
            subject = f"Re: {original_subject}"
        else:
            subject = original_subject
        body = response_text.strip()

    return subject, body
