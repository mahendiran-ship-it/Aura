import pytest

from aura.audio.command_parser import Intent, parse_command


@pytest.mark.parametrize("text,intent,target", [
    ("AURA, describe.", Intent.DESCRIBE, None),
    ("describe", Intent.DESCRIBE, None),
    ("What is in front of me?", Intent.WHAT_IN_FRONT, None),
    ("Find my phone.", Intent.FIND, "phone"),
    ("Find my backpack.", Intent.FIND, "backpack"),
    ("Search for a bottle.", Intent.FIND, "bottle"),
    ("Look for my keys.", Intent.FIND, "keys"),
    ("Where is the chair?", Intent.WHERE_IS, "chair"),
    ("Where is the door?", Intent.WHERE_IS, "door"),
    ("Where did I last see my bag?", Intent.LAST_SEEN, "bag"),
    ("where was my phone last seen", Intent.LAST_SEEN, "phone"),
    ("Remember this location.", Intent.REMEMBER_LOCATION, None),
    ("remember this location as kitchen", Intent.REMEMBER_LOCATION, "kitchen"),
    ("I am in the kitchen", Intent.SET_ROOM, "kitchen"),
    ("Repeat.", Intent.REPEAT, None),
    ("Stop talking.", Intent.STOP_TALKING, None),
    ("Cancel.", Intent.CANCEL, None),
    ("Help.", Intent.HELP, None),
    ("stop assistance", Intent.PAUSE, None),
    ("resume", Intent.RESUME, None),
    ("guide me toward it", Intent.GUIDE, None),
    ("forget my bag", Intent.FORGET, "bag"),
    ("clear all memory", Intent.CLEAR_MEMORY, None),
    ("navigate to central park", Intent.NAVIGATE, "central park"),
    ("where am I", Intent.WHERE_AM_I, None),
    ("emergency", Intent.EMERGENCY, None),
    ("   ", Intent.EMPTY, None),
])
def test_known_commands(text, intent, target):
    cmd = parse_command(text)
    assert cmd.intent == intent
    assert cmd.target == target


@pytest.mark.parametrize("text", ["Read this.", "read this", "what does this say"])
def test_read(text):
    assert parse_command(text).intent == Intent.READ


def test_read_sign_and_important():
    assert parse_command("Read the sign.").extra == "sign"
    cmd = parse_command("Read only the important information.")
    assert cmd.intent == Intent.READ and cmd.extra == "important"


@pytest.mark.parametrize("text", [
    "Is it safe to cross?", "can I cross the road", "is the road clear to cross now",
    "should I cross", "Is the crossing safe",
])
def test_crossing_questions_are_always_recognised(text):
    assert parse_command(text).intent == Intent.CROSSING


def test_free_question_goes_to_ai_path():
    assert parse_command("What is the difference between these two objects?").intent == Intent.ASK


def test_unrecognised_short_text_is_not_a_command():
    assert parse_command("hmm").intent == Intent.EMPTY


def test_none_input_is_safe():
    assert parse_command(None).intent == Intent.EMPTY  # type: ignore[arg-type]


def test_confirm_location():
    cmd = parse_command("remember that my keys are on the shelf")
    assert cmd.intent == Intent.CONFIRM_LOCATION
    assert cmd.target == "keys" and cmd.extra == "on the shelf"
