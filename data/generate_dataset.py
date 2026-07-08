"""
generate_dataset.py
--------------------
Builds a labeled text-emotion training dataset using template-based data
augmentation. Public emotion datasets (e.g. Kaggle's "Emotion Dataset") need
an internet download and an account, so instead we generate our own diverse
dataset by combining emotion-specific sentence templates with swappable
subjects/objects/reasons. This is a legitimate, commonly-used technique for
bootstrapping a text classifier when you don't have a large labeled corpus.

Run:
    python generate_dataset.py
Produces:
    emotion_dataset.csv  (columns: text, emotion)
"""

import csv
import itertools
import random

random.seed(42)

SUBJECTS = ["I", "My friend", "She", "He", "We", "My family", "My teammate", "My sister", "My colleague", "They"]

REASONS = {
    "joy": [
        "got promoted at work",
        "passed the final exam",
        "won the cricket match",
        "got selected for the internship",
        "finished the project on time",
        "got a surprise gift",
        "met an old friend after years",
        "got into the college of my dreams",
        "cooked a perfect meal",
        "solved a really hard bug",
    ],
    "sadness": [
        "lost the match in the final over",
        "failed the exam despite studying hard",
        "missed the train home",
        "had to say goodbye to a close friend",
        "didn't get the internship offer",
        "lost an important file before submission",
        "spent the weekend alone and unwell",
        "watched the team lose in the last minute",
        "couldn't attend the family function",
        "received disappointing feedback on the project",
    ],
    "anger": [
        "got blamed for someone else's mistake",
        "waited two hours for a delayed flight",
        "found the code deleted with no backup",
        "got cut off in traffic again",
        "was ignored during the meeting",
        "saw the project credit given to someone else",
        "kept getting spam calls all day",
        "was accused of cheating unfairly",
        "had the internet go down during the exam",
        "got a rude reply from support",
    ],
    "fear": [
        "have to give a presentation tomorrow",
        "heard strange noises outside at night",
        "might fail the upcoming interview",
        "lost the way on a dark road",
        "saw the server crash right before the demo",
        "have a final viva next week",
        "felt the ground shake during the earthquake drill",
        "are waiting for risky medical test results",
        "have to walk home alone late at night",
        "might lose the job after the merger",
    ],
    "surprise": [
        "found an old photo hidden in a drawer",
        "got an unexpected call from an old friend",
        "saw a surprise party waiting at home",
        "received an unexpected job offer",
        "found out the exam was cancelled",
        "saw a shooting star for the first time",
        "got a random bonus from work",
        "opened the box to find a puppy inside",
        "learned the flight got upgraded for free",
        "bumped into a celebrity at the mall",
    ],
    "love": [
        "spent the whole evening with family",
        "got a heartfelt letter from a close friend",
        "watched the sunset with someone special",
        "received a handmade gift from a sibling",
        "cooked dinner together after a long time",
        "reunited with childhood friends",
        "adopted a puppy from the shelter",
        "celebrated an anniversary with loved ones",
        "got a warm hug after a rough week",
        "planned a surprise trip for the family",
    ],
}

TEMPLATES = [
    "{subj} {reason} and {feeling}.",
    "{subj} {reason}, {feeling_clause}.",
    "Honestly, {subj_l} {reason} and {feeling}.",
    "{subj} just {reason} — {feeling_clause}.",
    "Today {subj_l} {reason}, {feeling_clause}.",
]

FEELING = {
    "joy": ["felt absolutely thrilled", "was overjoyed", "couldn't stop smiling", "felt on top of the world", "felt so happy"],
    "sadness": ["felt heartbroken", "was devastated", "felt like crying", "felt really down", "felt so low"],
    "anger": ["was furious", "felt so frustrated", "was fuming", "couldn't control the anger", "was really irritated"],
    "fear": ["was terrified", "felt so anxious", "was shaking with worry", "felt a wave of panic", "was really scared"],
    "surprise": ["was completely shocked", "couldn't believe it", "was stunned", "was totally caught off guard", "was amazed"],
    "love": ["felt so much warmth", "felt truly grateful", "felt so loved", "cherished every moment", "felt deeply connected"],
}

FEELING_CLAUSE = {
    "joy": ["it made my whole day", "it felt amazing", "I was grinning the entire time", "it was such a happy moment"],
    "sadness": ["it was a really hard day", "it hurt more than expected", "I just wanted to be alone", "it left me in tears"],
    "anger": ["it really wasn't fair", "I couldn't calm down for hours", "it ruined my mood completely", "it was so frustrating"],
    "fear": ["I couldn't stop worrying", "my hands were shaking", "I barely slept that night", "it kept me on edge"],
    "surprise": ["nobody saw it coming", "it caught everyone off guard", "I had to sit down for a second", "it was unbelievable"],
    "love": ["it meant the world to me", "it was such a warm feeling", "I felt truly at peace", "it filled my heart"],
}


def build_rows():
    rows = []
    for emotion, reasons in REASONS.items():
        for subj, reason in itertools.product(SUBJECTS, reasons):
            template = random.choice(TEMPLATES)
            feeling = random.choice(FEELING[emotion])
            feeling_clause = random.choice(FEELING_CLAUSE[emotion])
            text = template.format(
                subj=subj,
                subj_l=subj[0].lower() + subj[1:] if subj != "I" else "I",
                reason=reason,
                feeling=feeling,
                feeling_clause=feeling_clause,
            )
            rows.append((text, emotion))
    random.shuffle(rows)
    return rows


def main():
    rows = build_rows()
    with open("emotion_dataset.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "emotion"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to emotion_dataset.csv")
    from collections import Counter
    print(Counter(r[1] for r in rows))


if __name__ == "__main__":
    main()
