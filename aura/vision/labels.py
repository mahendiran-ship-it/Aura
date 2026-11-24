"""Object vocabulary of the default detector (COCO, 80 classes) and spoken aliases.

IMPORTANT, honest limitation: the default COCO model has NO class for
doors, stairs, keys, wallets, glasses or signs. AURA says so instead of
guessing. See docs/ROADMAP.md for how to plug in a model that has them.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

COCO_LABELS: List[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

# Words a user may say -> detector labels that satisfy it (first = preferred).
ALIASES = {
    "phone": ["cell phone"], "mobile": ["cell phone"], "cellphone": ["cell phone"],
    "smartphone": ["cell phone"], "cell phone": ["cell phone"],
    "bag": ["backpack", "handbag", "suitcase"], "backpack": ["backpack"],
    "handbag": ["handbag"], "purse": ["handbag"], "suitcase": ["suitcase"],
    "luggage": ["suitcase"],
    "sofa": ["couch"], "couch": ["couch"],
    "table": ["dining table"], "dining table": ["dining table"], "desk": ["dining table"],
    "tv": ["tv"], "television": ["tv"], "monitor": ["tv"],
    "plant": ["potted plant"], "cup": ["cup"], "mug": ["cup"], "glass": ["wine glass", "cup"],
    "bottle": ["bottle"], "water bottle": ["bottle"],
    "chair": ["chair"], "bench": ["bench"], "bed": ["bed"], "person": ["person"],
    "man": ["person"], "woman": ["person"], "people": ["person"],
    "laptop": ["laptop"], "computer": ["laptop"], "remote": ["remote"], "book": ["book"],
    "umbrella": ["umbrella"], "bicycle": ["bicycle"], "bike": ["bicycle"],
    "car": ["car"], "bus": ["bus"], "truck": ["truck"], "motorcycle": ["motorcycle"],
    "traffic light": ["traffic light"], "dog": ["dog"], "cat": ["cat"],
    "sink": ["sink"], "toilet": ["toilet"], "fridge": ["refrigerator"],
    "refrigerator": ["refrigerator"], "microwave": ["microwave"], "oven": ["oven"],
    "clock": ["clock"], "bowl": ["bowl"], "knife": ["knife"], "fork": ["fork"],
    "spoon": ["spoon"], "keyboard": ["keyboard"], "mouse": ["mouse"],
    "scissors": ["scissors"], "vase": ["vase"], "toothbrush": ["toothbrush"],
}

# Things people ask for that the default model cannot recognise.
UNSUPPORTED = {
    "keys", "key", "wallet", "glasses", "spectacles", "door", "doorway", "stairs",
    "staircase", "steps", "sign", "signboard", "pen", "charger", "wheelchair",
    "elevator", "lift", "crosswalk", "curb", "kerb", "pothole",
}


def resolve_target(name: str) -> Tuple[List[str], Optional[str]]:
    """Map what the user said to detector labels.

    Returns ``(labels, problem)``. ``problem`` is None when supported, else a
    short spoken explanation of why AURA cannot look for it.
    """
    n = name.lower().strip()
    if n in UNSUPPORTED:
        return [], (f"My current object model cannot recognize {n}. "
                    "I can check whether I remember seeing it, but I cannot search for it live.")
    if n in ALIASES:
        return list(ALIASES[n]), None
    if n.endswith("s") and n[:-1] in ALIASES:
        return list(ALIASES[n[:-1]]), None
    if n in COCO_LABELS:
        return [n], None
    return [], f"I don't know how to look for {n} with my current object model."
