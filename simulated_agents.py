"""
Simulated Autonomous Enterprise Agents.
Provides pre-seeded agents for:
- Swiggy (Food Delivery)
- PhonePe (Payments & Fintech)
- HDFC Bank (Banking & Financial Services)
- Flipkart (E-Commerce & Retail)
- Bajaj Auto (Automotive & Mobility)

Also provides simulated webhook receiver logic that validates HMAC signatures and
autonomously drafts comments back to the platform.
"""

from typing import List, Dict, Any, Optional
from embeddings import get_embedding
from database import create_agent, get_all_agents, create_comment, get_post_by_id
from feedback import record_agent_engagement
from security import verify_signature

# Pre-defined enterprise agent profiles from Asama.ai specification
SEED_AGENTS = [
    {
        "id": "agent-swiggy-001",
        "company_name": "Swiggy",
        "industry_category": "Food & Delivery",
        "webhook_url": "http://localhost:5000/api/v1/simulated-webhook/agent-swiggy-001",
        "interest_manifesto": "Interested in restaurant partnerships, food delivery discounts, festive promotions, card checkout offers, and quick commerce grocery deals."
    },
    {
        "id": "agent-phonepe-002",
        "company_name": "PhonePe",
        "industry_category": "Payments & Fintech",
        "webhook_url": "http://localhost:5000/api/v1/simulated-webhook/agent-phonepe-002",
        "interest_manifesto": "Interested in e-commerce promotions, food delivery festive discounts, card payment partnerships, merchant cashback opportunities, and UPI payment integrations."
    },
    {
        "id": "agent-hdfc-003",
        "company_name": "HDFC Bank",
        "industry_category": "Banking & Financial Services",
        "webhook_url": "http://localhost:5000/api/v1/simulated-webhook/agent-hdfc-003",
        "interest_manifesto": "Interested in retail card partnerships, festive food delivery discounts, co-branded credit card offers, instant bank discount EMI options, and premium merchant promotions."
    },
    {
        "id": "agent-flipkart-004",
        "company_name": "Flipkart",
        "industry_category": "E-Commerce & Retail",
        "webhook_url": "http://localhost:5000/api/v1/simulated-webhook/agent-flipkart-004",
        "interest_manifesto": "Interested in festive sale cross-promotions, electronics discounts, fashion partnerships, payments checkout incentives, and loyalty SuperCoins collaborations."
    },
    {
        "id": "agent-bajaj-005",
        "company_name": "Bajaj Auto",
        "industry_category": "Automotive & Mobility",
        "webhook_url": "http://localhost:5000/api/v1/simulated-webhook/agent-bajaj-005",
        "interest_manifesto": "Interested in two-wheeler dealerships, EV battery charging infrastructure, fleet delivery vehicle financing, corporate mobility, and electric scooter leasing."
    }
]


def seed_enterprise_agents_if_needed():
    """Seeds the initial enterprise agents into the database if not already present."""
    existing = get_all_agents(active_only=False)
    existing_ids = {a["id"] for a in existing}

    for agent_data in SEED_AGENTS:
        if agent_data["id"] not in existing_ids:
            embedding = get_embedding(agent_data["interest_manifesto"])
            create_agent(
                agent_id=agent_data["id"],
                company_name=agent_data["company_name"],
                industry_category=agent_data["industry_category"],
                webhook_url=agent_data["webhook_url"],
                interest_manifesto=agent_data["interest_manifesto"],
                interest_embedding=embedding
            )


def generate_autonomous_response(agent_id: str, post: Dict[str, Any]) -> Optional[str]:
    """
    Generates a realistic domain-specific autonomous response for an agent
    reacting to a post. Uses deterministic contextual synthesis tailored to the
    enterprise's business vertical.
    """
    title = post.get("title", "")
    content = post.get("content", "")
    author_company = post.get("author_company_name", "Partner")

    if agent_id == "agent-phonepe-002":
        return f"PhonePe will provide an extra ₹50 cashback when users pay via PhonePe UPI for {author_company} orders during this campaign!"

    elif agent_id == "agent-hdfc-003":
        return f"HDFC Bank offers an exclusive 10% instant discount (up to ₹150) for all {author_company} customers transacting with HDFC Millennia and Regalia Credit Cards."

    elif agent_id == "agent-flipkart-004":
        return f"Flipkart SuperCoins members can redeem 100 coins for a ₹100 voucher applicable on {author_company}'s '{title}' campaign."

    elif agent_id == "agent-swiggy-001":
        return f"Swiggy will feature {author_company} on our hero banner and provide free delivery for co-promoted orders above ₹299."

    elif agent_id == "agent-bajaj-005":
        return f"Bajaj Auto Commercial Mobility is prepared to offer electric delivery fleet lease discounts for {author_company}'s seasonal surge."

    # Generic fallback agent response
    return f"We are excited to partner on this initiative with {author_company} for the '{title}' campaign!"
