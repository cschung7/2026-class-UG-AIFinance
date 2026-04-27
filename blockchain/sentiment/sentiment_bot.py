from datetime import datetime
def analyze(text):
    score = 0
    if 'bullish' in text.lower() or 'surge' in text.lower(): score = 0.5
    if 'bearish' in text.lower() or 'hack' in text.lower(): score = -0.5
    return score
def run():
    news = ["Bitcoin adoption surges", "Exchange hack causes panic"]
    print(f"[{datetime.now()}] Sentiment Bot")
    for n in news:
        s = analyze(n)
        print(f"{n:<30} | {s:<5} | {'BUY' if s > 0 else 'SELL'}")
if __name__ == "__main__": run()
