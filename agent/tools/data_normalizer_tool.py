import re
from typing import List, Dict
from .currency_converter import get_exchange_rate


def normalize_offer_data(text_data: str) -> List[Dict]:
    """
    Normalizes offer data by extracting product information, identifying Norwegian text,
    and converting prices from NOK to USD.

    Args:
        text_data: The raw text data from the offer.

    Returns:
        A list of dictionaries, where each dictionary represents a normalized product or service.
    """
    normalized_data = []
    lines = text_data.split("\n")

    # Regular expression to find product lines and extract relevant information
    # This is a simplified regex and might need adjustment based on actual data variations
    product_pattern = re.compile(
        r"^(?P<description>[^\d]+?)\s*(?P<quantity>\d+)?\s*(?P<unit>per\s*stykk|totalpris|per\s*time|per\s*år)?\s*(?P<price>[\d\.,]+)?"
    )

    current_section = None
    for line in lines:
        line = line.strip()
        if not line:
            continue

        if "Hovedprodukt" in line:
            current_section = "Hovedprodukt"
            continue
        elif "Opplæring" in line:
            current_section = "Opplæring"
            continue
        elif "Opsjon: servicenivå" in line:
            current_section = "Servicenivå"
            continue
        elif "Opsjon: vare/produkt/utstyr/kurs" in line:
            current_section = "Opsjon"
            continue
        elif "Sum Hovedprodukt" in line:
            current_section = None
            continue

        if current_section:
            match = product_pattern.match(line)
            if match:
                description = match.group("description").strip()
                quantity = match.group("quantity")
                unit = match.group("unit")
                price_str = match.group("price")

                price_nok = None
                if price_str:
                    # Handle comma as decimal separator and remove thousand separators
                    price_str = price_str.replace(".", "").replace(",", ".")
                    try:
                        price_nok = float(price_str)
                    except ValueError:
                        price_nok = None

                price_usd = None
                if price_nok is not None:
                    # Assuming get_exchange_rate can be called directly or mocked for testing
                    # In a real scenario, this would involve calling the external API
                    # For now, we'll simulate the conversion or use a placeholder
                    # exchange_rate = get_exchange_rate("NOK", "USD") # This would print to stdout, not return
                    # For now, let's use a fixed exchange rate for demonstration
                    exchange_rate_nok_to_usd = (
                        0.095  # Example rate, needs to be dynamic
                    )
                    price_usd = round(price_nok * exchange_rate_nok_to_usd, 2)

                normalized_data.append(
                    {
                        "original_description_norwegian": description,
                        "quantity": int(quantity) if quantity else 1,
                        "unit": unit if unit else "unknown",
                        "price_nok": price_nok,
                        "price_usd": price_usd,
                        "section": current_section,
                    }
                )
    return normalized_data
