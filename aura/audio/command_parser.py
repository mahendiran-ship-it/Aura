"""Rule-based voice/text command parser. Works with no AI model at all.

Order matters: more specific patterns are tried before general ones
(e.g. "where did I last see my bag" before "where is the bag").
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional, Tuple


class Intent(str, Enum):
    DESCRIBE = "describe"
    WHAT_IN_FRONT = "what_in_front"
    FIND = "find"
    WHERE_IS = "where_is"
    GUIDE = "guide"
    READ = "read"
    REMEMBER_LOCATION = "remember_location"
    SET_ROOM = "set_room"
    LAST_SEEN = "last_seen"
    CONFIRM_LOCATION = "confirm_location"
    MEMORY_HISTORY = "memory_history"
    FORGET = "forget"
    CLEAR_MEMORY = "clear_memory"
    NAVIGATE = "navigate"
    WHERE_AM_I = "where_am_i"
    CROSSING = "crossing"
    REPEAT = "repeat"
    STOP_TALKING = "stop_talking"
    CANCEL = "cancel"
    PAUSE = "pause"
    RESUME = "resume"
    STATUS = "status"
    EMERGENCY = "emergency"
    HELP = "help"
    QUIT = "quit"
    ASK = "ask"  # free-form question -> optional AI
    EMPTY = "empty"


@dataclass
class Command:
    intent: Intent
    target: Optional[str] = None  # object / place / room name
    extra: Optional[str] = None  # second argument (e.g. read mode, position)
    raw: str = ""


_FILLER = r"^(?:(?:hey|ok|okay)\s+)?(?:aura|aurora)\b[\s,.:!-]*"
_ARTICLES = r"^(?:my|the|a|an|some|any|our|your)\s+"


def clean_target(text: str) -> str:
    """'my red bottle please' -> 'red bottle' (strips articles/politeness)."""
    t = text.strip().strip("?.!,").lower()
    t = re.sub(r"\b(please|for me|now|around here|nearby)\b", "", t).strip()
    while re.match(_ARTICLES, t):
        t = re.sub(_ARTICLES, "", t).strip()
    t = re.sub(r"\s+", " ", t)
    # reduce "red bottle" -> keep full phrase; resolve_target handles aliases / last word
    return t


def normalize(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"[^\w\s'?.,-]", " ", t)
    t = re.sub(_FILLER, "", t).strip()
    t = re.sub(r"\s+", " ", t)
    return t.strip(" .!,?")


Rule = Tuple[str, Callable[[re.Match], Command]]


def _rules() -> List[Rule]:
    I = Intent

    def c(intent: Intent, target_group: Optional[int] = None, extra_group: Optional[int] = None):
        def build(m: re.Match) -> Command:
            raw_t = m.group(target_group) if target_group else None
            raw_e = m.group(extra_group) if extra_group else None
            tgt = clean_target(raw_t) if raw_t else None
            ext = raw_e.strip() if raw_e else None
            return Command(intent, tgt or None, ext)
        return build

    return [
        # --- immediate controls ---
        (r"^(?:stop talking|be quiet|quiet|silence|shut up|hush|stop speaking)$", c(I.STOP_TALKING)),
        (r"^(?:cancel|never ?mind|stop searching|stop looking|abort)$", c(I.CANCEL)),
        (r"^(?:stop|pause)(?: assistance| aura| listening)?$", c(I.PAUSE)),
        (r"^(?:resume|start|continue|start assistance|resume assistance|wake up)$", c(I.RESUME)),
        (r"^(?:repeat|say that again|repeat that|what did you say|come again)$", c(I.REPEAT)),
        (r"^(?:help|what can you do|commands|what can i say)$", c(I.HELP)),
        (r"^(?:quit|exit|shut down|shutdown|goodbye|bye)$", c(I.QUIT)),
        (r"^(?:status|how are you doing|system status|are you working)$", c(I.STATUS)),
        (r"^(?:emergency|call for help|i need help now|help me now|sos)$", c(I.EMERGENCY)),
        # --- road crossing: always refuse to judge ---
        (r"^.*\b(?:(?:safe|ok|okay|clear)\b.*\bcross(?:ing)?|cross(?:ing)?\b.*\b(?:safe|ok|okay|clear)|can i cross|should i cross|may i cross|is it safe to (?:go|walk))\b.*$", c(I.CROSSING)),
        # --- memory ---
        (r"^(?:where (?:did|have) i (?:last )?(?:see|saw|put|leave|left|seen|place|placed)|where was|where were|when did i last see|when was) (.+?)(?: last (?:seen|observed))?$", c(I.LAST_SEEN, 1)),
        (r"^(?:where(?:'s| is) )?(?:my )?(.+?) (?:was|were) last (?:seen|observed)$", c(I.LAST_SEEN, 1)),
        (r"^(?:last seen|last known location of|where did i leave) (.+)$", c(I.LAST_SEEN, 1)),
        (r"^(?:remember|save|mark) (?:this|the|my current) (?:location|place|spot)(?: as (.+))?$", c(I.REMEMBER_LOCATION, 1)),
        (r"^(?:i am|i'm|we are|we're) (?:now )?(?:in|at) (?:the |my |a )?(.+)$", c(I.SET_ROOM, 1)),
        (r"^(?:this is|set room(?: to)?|room is|current room is) (?:the |my )?(.+)$", c(I.SET_ROOM, 1)),
        (r"^remember (?:that )?(?:my |the )?(.+?) (?:is|are) (.+)$", c(I.CONFIRM_LOCATION, 1, 2)),
        (r"^(?:show |view |read )?(?:memory|memories)(?: history)?$|^memory history$", c(I.MEMORY_HISTORY)),
        (r"^(?:what do you remember(?: about (.+))?|search memory(?: for (.+))?)$", lambda m: Command(I.MEMORY_HISTORY, clean_target(m.group(1) or m.group(2) or "") or None)),
        (r"^(?:forget|delete|remove) (?:memory of |memories of |memory about |everything about |about )?(?:my |the )?(.+)$", c(I.FORGET, 1)),
        (r"^(?:clear|delete|erase|wipe) (?:all )?(?:my )?(?:memory|memories|stored data|everything)$", c(I.CLEAR_MEMORY)),
        # --- reading ---
        (r"^(?:read|scan) (?:only )?(?:the )?(?:important|key) (?:information|info|parts?|text)$|^read only the important information$", lambda m: Command(I.READ, None, "important")),
        (r"^(?:read|scan) (?:the )?(sign|signboard|label|menu|document|paper|text|screen|display|this|that|it)(?: for me)?$", c(I.READ, None, 1)),
        (r"^(?:read|scan)$|^read (?:this|that|it|text) ?(?:for me|out loud|aloud)?$|^what does (?:this|it|that) say$", c(I.READ)),
        # --- searching ---
        (r"^(?:guide me|take me|lead me|walk me|help me reach|direct me)(?: to| toward| towards)?(?: it| that| the .+| my .+)?$|^(?:where is it now|how do i get to it)$", c(I.GUIDE)),
        (r"^(?:find|look for|search for|locate|search|seek|spot) (.+)$", c(I.FIND, 1)),
        (r"^(?:where(?:'s| is| are)) (.+)$", c(I.WHERE_IS, 1)),
        (r"^(?:is there|are there|do you see|can you see|do i have) (?:a |an |any |my |the )?(.+?) (?:near|nearby|around|here|in front|ahead|in view)?$", c(I.WHERE_IS, 1)),
        # --- describing ---
        (r"^(?:what(?:'s| is) )?(?:in front of me|ahead of me|ahead|in front)$|^what(?:'s| is) (?:right )?(?:in front|ahead)(?: of me)?$", c(I.WHAT_IN_FRONT)),
        (r"^(?:describe|describe (?:the |my )?(?:scene|surroundings|room|area|environment)|what(?:'s| is) (?:around me|around|in the room|here|in this room|near me)|what do you see|look around|tell me what you see|what is this place|surroundings|scene)$", c(I.DESCRIBE)),
        # --- navigation ---
        (r"^(?:navigate|take me|directions|route|walk) (?:to|toward|towards) (.+)$|^(?:navigate to|directions to) (.+)$", lambda m: Command(I.NAVIGATE, clean_target(m.group(1) or m.group(2) or ""))),
        (r"^(?:where am i|what(?:'s| is) my location|my location|what(?:'s| is) near me outdoors)$", c(I.WHERE_AM_I)),
    ]


_RULES = [(re.compile(p), fn) for p, fn in _rules()]

_QUESTION_START = re.compile(r"^(?:what|which|who|how|why|is|are|does|do|can|could|tell|explain|describe|compare)\b")


def parse_command(text: str) -> Command:
    """Parse one utterance into a :class:`Command`. Never raises."""
    if text is None or not text.strip():
        return Command(Intent.EMPTY, raw=text or "")
    norm = normalize(text)
    if not norm:
        return Command(Intent.EMPTY, raw=text)
    for pattern, build in _RULES:
        m = pattern.match(norm)
        if m:
            cmd = build(m)
            cmd.raw = text
            if cmd.intent in (Intent.FIND, Intent.WHERE_IS, Intent.LAST_SEEN, Intent.FORGET,
                              Intent.CONFIRM_LOCATION, Intent.SET_ROOM, Intent.NAVIGATE) and not cmd.target:
                continue
            return cmd
    if _QUESTION_START.match(norm):
        return Command(Intent.ASK, target=norm, raw=text)
    return Command(Intent.ASK if len(norm.split()) > 3 else Intent.EMPTY, target=norm, raw=text)


HELP_TEXT = (
    "You can say: describe; what is in front of me; find my phone, bottle or backpack; "
    "where is the chair; guide me; read this; read only the important information; "
    "remember this location; I am in the kitchen; where did I last see my bag; "
    "forget my bag; clear all memory; repeat; stop talking; cancel; stop assistance; "
    "status; navigate to a place; and emergency. "
    "I am an aid only and never replace your cane, guide dog or mobility training."
)
