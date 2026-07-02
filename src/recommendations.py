from __future__ import annotations


def score_customer_health(probability: float) -> dict[str, int]:
    """Return customer-facing health, confidence, and retention scores."""
    health_score = round((1 - probability) * 100)
    retention_score = round((1 - min(probability * 0.85, 1)) * 100)
    confidence_score = round((0.5 + abs(probability - 0.5)) * 100)
    return {
        "health_score": int(health_score),
        "retention_score": int(retention_score),
        "confidence_score": int(confidence_score),
    }


def build_recommendations(customer: dict, probability: float) -> dict[str, list[str] | str]:
    """Generate simple business recommendations from model output and inputs."""
    risk_factors: list[str] = []
    positive_factors: list[str] = []

    if customer.get("Contract") == "Month-to-month":
        risk_factors.append("Month-to-month contract increases churn exposure.")
    else:
        positive_factors.append("Longer contract commitment supports retention.")

    if customer.get("InternetService") == "Fiber optic":
        risk_factors.append("Fiber optic customers showed elevated churn in EDA.")
    elif customer.get("InternetService") == "No":
        positive_factors.append("No-internet customers had comparatively low churn.")

    if customer.get("PaymentMethod") == "Electronic check":
        risk_factors.append("Electronic check users are a high-risk payment segment.")
    elif "automatic" in str(customer.get("PaymentMethod", "")).lower():
        positive_factors.append("Automatic payment reduces billing-friction risk.")

    tenure = float(customer.get("tenure", 0) or 0)
    if tenure <= 12:
        risk_factors.append("Low tenure indicates the customer is still early in the lifecycle.")
    elif tenure >= 24:
        positive_factors.append("Longer tenure suggests stronger customer loyalty.")

    monthly = float(customer.get("MonthlyCharges", 0) or 0)
    if monthly >= 80:
        risk_factors.append("High monthly charges may increase price sensitivity.")
    elif monthly <= 45:
        positive_factors.append("Lower monthly charges may reduce cancellation pressure.")

    if not risk_factors:
        risk_factors.append("No major business risk factors detected from the submitted fields.")
    if not positive_factors:
        positive_factors.append("Few stabilizing signals detected; monitor engagement closely.")

    if probability >= 0.65:
        strategy = "High-touch save plan"
        actions = [
            "Assign a retention specialist within 24 hours.",
            "Offer a contract upgrade incentive or loyalty discount.",
            "Review service quality concerns before proposing a renewal.",
            "Bundle support, security, or backup services to increase stickiness.",
            "Follow up after seven days and track response in CRM.",
        ]
    elif probability >= 0.35:
        strategy = "Targeted engagement plan"
        actions = [
            "Send a personalized check-in campaign.",
            "Offer a plan review focused on value and usage fit.",
            "Promote automatic payment or longer-term contract options.",
            "Monitor support tickets and billing complaints for the next month.",
        ]
    else:
        strategy = "Relationship maintenance plan"
        actions = [
            "Keep customer on standard nurture campaigns.",
            "Offer loyalty recognition instead of heavy discounts.",
            "Watch for future changes in tenure, price, or support activity.",
        ]

    return {
        "strategy": strategy,
        "risk_factors": risk_factors,
        "positive_factors": positive_factors,
        "actions": actions,
    }
