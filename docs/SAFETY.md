# Safety

## What AURA is and is not

AURA is an **assistive aid**. It is **not** a mobility device, not a replacement for a white cane, guide dog, or orientation-and-mobility (O&M) training, and not validated for use as a safety system. It has not been tested with blind users or evaluated by any regulator. **Do not rely on it where a mistake could injure you** — roads, stairs, platforms, construction, water, traffic.

## Hard rules built into the software

1. **No "safe" claims.** AURA never says a road, crossing, staircase, path or area is safe or clear. The absence of detections is never reported as safety. Empty-scene answer: *"I don't recognize any objects in view right now. That does not mean the area is clear."*
2. **Crossing questions are always refused**, by a fixed rule independent of any AI: *"I cannot reliably determine whether it is safe to cross. Please stop and use your established crossing method."* There is no autonomous road-crossing logic.
3. **Safety decisions never use an LLM.** Obstacle alerts come from `aura/safety/` (plain, deterministic, configurable).
4. **Uncertainty is spoken.** Distances are "appears close / medium / far", never metres, unless a real depth source is plugged in. High-level alerts add "Distance is approximate."
5. **Failures are announced.** Camera lost, model missing/crashed, or poor image quality → AURA says its hazard warnings are limited or unavailable.
6. **Pausing is explicit.** "Stop assistance" says AURA will no longer warn about obstacles until "resume".
7. **Memory is dated.** Remembered locations carry age and freshness and are never stated as current.
8. **OCR is gated.** Low-confidence text is withheld rather than guessed.
9. **No fabricated data.** No simulated detections or sensor readings in the runtime; routes come from a real routing service or not at all.

These rules are covered by automated tests (`tests/test_safety.py`, `tests/test_system.py`).

## Risk scoring

```
score = class_weight × (0.4 + 0.9 × proximity) × position × movement × reliability
position  : in walking zone 1.0, side 0.55, far side 0.3
movement  : approaching 1.35, stationary 1.0, moving away 0.6
reliability: confidence × (0.5 + 0.5 × persistence)
score → INFO < LOW (0.25) < MEDIUM (0.5) < HIGH (0.8) < CRITICAL (1.15)
```

All numbers live in `config/safety.yaml`. **They are engineering heuristics chosen for plausible behaviour, not derived from injury or near-miss data.** They need field calibration with O&M professionals before anyone relies on them. Turning alerts down reduces protection.

Alert policy: ignore weak (<0.5 confidence) or short-lived (<3 frames) detections; one spoken alert at a time; same-object cooldowns (LOW 20 s … CRITICAL 2 s); a worse level overrides the cooldown; 1.5 s global gap except CRITICAL.

## Known failure modes

| Failure | Effect | Mitigation today |
|---|---|---|
| Object not in the 80 COCO classes (curb, stairs, pole, glass door, hole, wet floor) | **Not detected, silently** | Documented; AURA refuses to search for such classes. Needs a better model. |
| Narrow camera view, camera pointed wrongly, hand/clothing over lens | Hazards outside view missed | Blur/dark checks catch some covered-lens cases only |
| Fast vehicles, low frame rate on weak hardware | Warning arrives too late | `HIGH_PERFORMANCE` mode; still not suitable for traffic |
| Monocular distance error (big/small objects, camera tilt) | Wrong "near/far" | Hedged wording; plug in a real depth sensor |
| Night / glare / rain | Missed detections | Quality warning (partial) |
| Misdetection (false alarm) | Annoying alerts | Persistence + cooldown; may still occur |
| GPS error (3–10 m typical, worse near buildings) | Wrong turn timing | Guidance is advisory; arrival text says exact entrance can't be confirmed |
| OpenStreetMap pedestrian data gaps | Poor or unsafe routes | Route summary says crossings and obstacles are the user's responsibility |
| Cloud/LLM wrong answers | Misleading chat | Facts-only prompt, but LLMs can still err; not used for safety |

## Privacy and security

No image/audio storage or upload by default; memory is structured text with retention and deletion; the local API requires a token and listens on localhost; API keys come only from the environment. Details in the README.

## Reporting problems

If you find a case where AURA gave a dangerous or falsely reassuring message, please open an issue with the exact wording (no photos of other people). Treat such reports as top priority.
