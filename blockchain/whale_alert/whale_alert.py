import random
from datetime import datetime
WHALE_THRESHOLD = 500
def get_mock_transactions():
    return [{'hash': hex(random.getrandbits(256)), 'value': round(random.uniform(10, 1500), 2)} for _ in range(5)]
def monitor():
    print(f"[{datetime.now()}] Monitoring > {WHALE_THRESHOLD} ETH")
    for tx in get_mock_transactions():
        if tx['value'] >= WHALE_THRESHOLD:
            print(f"🚨 WHALE ALERT: {tx['value']} ETH | {tx['hash'][:18]}...")
if __name__ == "__main__": monitor()
