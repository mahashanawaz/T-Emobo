import pandas as pd
import time
import random

def stream_feedback(csv_path):
    df = pd.read_csv(csv_path)
    while True:
        row = df.sample(1).iloc[0]
        yield row['FeedbackText']
        time.sleep(random.uniform(2, 5))  # wait 2–5 sec before sending next feedback

if __name__ == "__main__":
    print("Starting feedback stream...")
    for feedback in stream_feedback("feedback_labeled.csv"):
        print("📩 New Feedback:", feedback)
