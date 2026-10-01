"""Formatting for human-facing numeric values."""


def format_amount(amount: float, currency: str = "") -> str:
    formatted_amount = f"{amount:,.2f}".replace(",", " ")
    return f"{formatted_amount} {currency}".strip()
