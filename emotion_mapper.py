"""
emotion_mapper.py
------------------
Maps predicted emotion labels to a representative emoji + short caption.
Kept separate from the model code so the mapping can be edited/extended
without touching the ML pipeline.
"""

EMOTION_EMOJI = {
    "joy": {"emoji": "😄", "label": "Joy"},
    "sadness": {"emoji": "😢", "label": "Sadness"},
    "anger": {"emoji": "😡", "label": "Anger"},
    "fear": {"emoji": "😨", "label": "Fear"},
    "surprise": {"emoji": "😲", "label": "Surprise"},
    "love": {"emoji": "❤️", "label": "Love"},
}


def map_emotion(emotion: str) -> dict:
    """Return the emoji/label info for a predicted emotion string."""
    return EMOTION_EMOJI.get(emotion, {"emoji": "❓", "label": "Unknown"})
