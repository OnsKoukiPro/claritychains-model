
import sys
import json
import urllib.request

def get_exchange_rate(source_currency: str, target_currency: str = "USD"):
    """
    Fetches the exchange rate from source_currency to target_currency.
    Prints the rate to stdout.
    """
    try:
        url = f"https://api.frankfurter.app/latest?from={source_currency.upper()}&to={target_currency.upper()}"
        with urllib.request.urlopen(url, timeout=10) as response:
            if response.status == 200:
                data = json.loads(response.read().decode('utf-8'))
                rate = data.get('rates', {}).get(target_currency.upper())
                if rate:
                    print(rate)
                else:
                    print(f"Error: Rate for {target_currency.upper()} not found in response.")
            else:
                print(f"Error: API request failed with status {response.status}.")
    except Exception as e:
        print(f"Error fetching exchange rate: {e}")

if __name__ == "__main__":
    if len(sys.argv) not in [2, 3]:
        print("Usage: python currency_converter.py <SOURCE_CURRENCY> [TARGET_CURRENCY]")
        print("If TARGET_CURRENCY is not provided, it defaults to USD.")
        sys.exit(1)

    source = sys.argv[1]
    target = sys.argv[2] if len(sys.argv) == 3 else "USD"
    get_exchange_rate(source, target)
