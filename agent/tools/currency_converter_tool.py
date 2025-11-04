from langchain.tools import tool
import json
import urllib.request
import re


@tool("currency_converter")
def currency_converter(data: str) -> str:
    """
    Fetches the exchange rate from source_currency to target_currency and optionally converts an amount.

    Input should be a JSON string like:
    {
        "source_currency": "EUR",
        "target_currency": "USD",
        "amount": 1000
    }

    Returns the exchange rate or the converted amount as a string.
    """
    try:
        # Parse input data
        data_dict = json.loads(data)
        source_currency = data_dict.get("source_currency", "USD")
        target_currency = data_dict.get("target_currency", "USD")
        amount = data_dict.get("amount")

        # Extract currency code
        currency_code = re.split(r"[\s,]", source_currency)[0].lower()
        target_currency_code = target_currency.lower()

        print(f"Converting {source_currency} to {target_currency}, amount: {amount}")

        # Try multiple API endpoints
        urls = [
            f"https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/{currency_code}.json",
            f"https://latest.currency-api.pages.dev/v1/currencies/{currency_code}.json",
        ]

        for url in urls:
            try:
                with urllib.request.urlopen(url, timeout=10) as response:
                    if response.status == 200:
                        api_data = json.loads(response.read().decode("utf-8"))
                        rate = api_data[currency_code].get(target_currency_code)

                        if rate:
                            if amount:
                                converted = float(rate) * float(amount)
                                return f"{converted:.2f} {target_currency.upper()}"
                            else:
                                return f"1 {source_currency.upper()} = {rate:.4f} {target_currency.upper()}"
                        else:
                            raise Exception(f"Rate for {target_currency.upper()} not found in response.")
            except Exception as e:
                print(f"URL {url} failed: {e}")
                continue

        raise Exception("All API endpoints failed.")

    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {str(e)}"
    except Exception as e:
        return f"Error fetching exchange rate: {str(e)}"