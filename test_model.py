"""
test_model.py
--------------
Quick sanity tests: loads the trained model and checks it predicts the
expected emotion on a handful of sentences it has never seen during
training (i.e. not drawn from the template generator's exact phrasing).

Run:
    python test_model.py
"""

import joblib

MODEL_PATH = "model/emotion_model.joblib"

CASES = [
    ("I just found out I got the internship offer, I could not stop smiling", "joy"),
    ("My teammate ignored my messages all day and I am so frustrated", "anger"),
    ("I have a viva tomorrow and I cannot stop worrying", "fear"),
    ("I found an old photo hidden in a drawer and could not believe it", "surprise"),
    ("My sister cooked dinner together after a long time and it felt so warm", "love"),
    ("I lost the match in the final over and felt heartbroken", "sadness"),
]


def main():
    model = joblib.load(MODEL_PATH)
    passed = 0
    for text, expected in CASES:
        pred = model.predict([text])[0]
        ok = pred == expected
        passed += ok
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] expected={expected:<9} got={pred:<9} | {text}")
    print(f"\n{passed}/{len(CASES)} passed")


if __name__ == "__main__":
    main()
