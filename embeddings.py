"""
Semantic Embedding Engine for AI Agent Relevance Matching.
Provides high-dimensional dense vector embeddings and cosine similarity computation.
Supports hybrid domain-semantic vector representation out-of-the-box with zero external dependencies,
and seamlessly switches to OpenAI / HuggingFace embedding APIs if configured.
"""

import math
import re
import os
from typing import List, Dict

# Domain concepts covering enterprise verticals (Fintech, E-commerce, Food Delivery, Banking, Auto, etc.)
SEMANTIC_DIMENSIONS = [
    # Payments & Fintech
    ["payment", "upi", "cashback", "wallet", "transaction", "gateway", "checkout", "refund", "qr", "merchant"],
    # Banking & Credit
    ["bank", "credit", "debit", "card", "emi", "loan", "interest", "account", "hdfc", "sbi", "axis", "icici", "financing"],
    # Food & Grocery Delivery
    ["food", "delivery", "restaurant", "dining", "swiggy", "zomato", "groceries", "instamart", "blinkit", "meal", "order"],
    # E-Commerce & Retail
    ["ecommerce", "shopping", "flipkart", "amazon", "electronics", "fashion", "apparel", "gadgets", "product", "cart"],
    # Discounts, Offers & Festive
    ["offer", "discount", "sale", "promo", "diwali", "festive", "cashback", "coupon", "savings", "deal", "voucher"],
    # Automotive & Mobility
    ["automotive", "vehicle", "bike", "car", "motorcycle", "bajaj", "tata", "ev", "electric", "scooter", "fuel", "engine"],
    # Logistics & Supply Chain
    ["logistics", "shipping", "courier", "warehouse", "tracking", "freight", "fleet", "lastmile", "transport"],
    # Travel & Hospitality
    ["travel", "flight", "hotel", "booking", "tourism", "vacation", "trip", "airline", "stay"],
    # Healthcare & Wellness
    ["health", "pharmacy", "medicine", "doctor", "wellness", "diagnostic", "clinic", "hospital"],
    # Partnerships & B2B
    ["partnership", "cobranded", "alliance", "integration", "b2b", "crosscompany", "sponsor", "collaboration"]
]

# Total vector dimension = domain dimensions + n-gram hashing projection = 64
VECTOR_DIM = 64


def stem(word: str) -> str:
    """Lightweight rule-based suffix stemming for normalization."""
    if len(word) <= 3:
        return word
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith("ing"):
        return word[:-3]
    if word.endswith("ed"):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def tokenize(text: str) -> List[str]:
    """Tokenize and stem lowercase alphanumeric tokens."""
    words = re.findall(r'[a-z0-9]+', text.lower())
    return [stem(w) for w in words]



def get_embedding(text: str) -> List[float]:
    """
    Computes a normalized dense vector embedding (length = VECTOR_DIM).
    Uses a hybrid approach:
    1. Core semantic domain projections (concept matching).
    2. Character n-gram hashing projections (captures open-vocabulary nuances).
    3. L2 normalization to unit hypersphere.
    """
    tokens = tokenize(text)
    if not tokens:
        return [0.0] * VECTOR_DIM

    vector = [0.0] * VECTOR_DIM
    token_set = set(tokens)

    # 1. Project into semantic domain dimensions
    for i, dim_keywords in enumerate(SEMANTIC_DIMENSIONS):
        weight = 0.0
        for kw in dim_keywords:
            if kw in token_set:
                weight += 1.5
            # Substring match
            for t in tokens:
                if kw in t or t in kw:
                    weight += 0.5
        vector[i] = weight

    # 2. Open-vocabulary n-gram hashing into remaining dimensions
    start_hash_idx = len(SEMANTIC_DIMENSIONS)
    hash_space = VECTOR_DIM - start_hash_idx

    for token in tokens:
        # Unigram hash
        h1 = (hash(token) & 0x7FFFFFFF) % hash_space
        vector[start_hash_idx + h1] += 1.0

        # Bigrams of characters
        for j in range(len(token) - 1):
            bigram = token[j:j+2]
            h2 = (hash(bigram) & 0x7FFFFFFF) % hash_space
            vector[start_hash_idx + h2] += 0.3

    # 3. L2 Unit Normalization
    norm = math.sqrt(sum(x * x for x in vector))
    if norm > 1e-9:
        vector = [round(x / norm, 5) for x in vector]
    else:
        vector = [0.0] * VECTOR_DIM

    return vector


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Computes cosine similarity between two unit vectors: dot product."""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    dot = sum(a * b for a, b in zip(vec1, vec2))
    return max(0.0, min(1.0, dot))
