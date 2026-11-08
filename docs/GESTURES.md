# Gesture Reference

This page describes the implemented recognition rules, not just the intended poses. All distances and landmark coordinates are normalized MediaPipe image coordinates. Recognition quality depends on framing, light, tracker confidence, hand labels, and the model's blendshape scores.

## Catalog

Priority is used only when more than one candidate is eligible at once. Higher values win. Duration, cooldown, and debounce are seconds.

| Name | Media | Priority | Duration / cooldown / debounce | Recognition rule |
| --- | --- | ---: | --- | --- |
| `hands_on_face` | `hands_on_face.jpg` | 90 | 2 / 3 / 0.18 | Both hands are near the nose; either palm centers are close or middle fingertips meet, the palm midpoint is centered over the lower face, and both middle fingers point upward. The pose must match for 3 qualifying evaluations. |
| `head_grab` | `headgrab.jpeg` | 89 | 2 / 3 / 0.05 | Both palms are above the eye-line plus a face-relative margin and sit on opposite sides of the nose. |
| `hands_on_cheeks` | `cheeks.jpeg` | 87 | 2 / 3 / 0.05 | Separated palms sit in opposite cheek zones, below the eye line and no lower than the chin. |
| `t_sign` | `T_sign.jpg` | 85 | 2 / 3 / 0.15 | Two nearby open hands form perpendicular horizontal and vertical axes. Each hand must have at least three extended fingers. |
| `explaining_hands` | `hands_talking_to_wall.gif` | 80 | 5 / 5 / 0.20 | Both hands are present, neither is near the nose, no recent face-adjacent hand pose is held, and their combined wrist path length reaches the motion threshold. |
| `left_nose_right_wave` | `dance_fox_hands_forward_wave_with_nose_touch.gif` | 60 | 5 / 5 / 0.20 | The physical left hand is near the nose while the physical right wrist makes a sufficiently long vertical movement. |
| `mouth_round` | `mouth_round.jpg` | 55 | 2 / 3 / 0.15 | Face blendshapes `mouthPucker` >= 0.40 and `jawOpen` >= 0.10. |
| `horse_smirk` | `horse_smirk.jpg` | 50 | 2 / 3 / 0.15 | Either mouth-smile corner score is >= 0.40 and the two smile-corner scores differ by >= 0.20. |
| `eyes_closed` | `master_piece_squiting_too_much.jpg` | 45 | 2 / 3 / 0.15 | Both `eyeBlinkLeft` and `eyeBlinkRight` blendshapes are >= 0.60. |
| `tongue_out` | `dog_tongue_out.jpg` | 40 | 2 / 3 / 0.10 | Vertical distance between face landmarks 13 and 14 is > 0.03. Despite its name, this is a mouth-opening proxy, not direct tongue classification. |
| `smile` | `smile_dog.jpg` | 35 | 2 / 3 / 0.15 | Both mouth-smile corner scores are >= 0.35 and differ by < 0.20. |
| `lips_down` | `beluga.jpg` | 30 | 2 / 3 / 0.12 | Both `mouthFrownLeft` and `mouthFrownRight` blendshapes are >= 0.28. |
| `open_palm` | `hold_up_hand_forward.jpg` | 10 | 2 / 2 / 0.10 | MediaPipe's top recognized gesture category is exactly `Open_Palm`; no tracked hand is near the nose and the hands do not form a T-sign. |

## Candidate Ordering And Exclusions

Facial candidates are evaluated independently from hand candidates, so a facial expression can compete with a hand pose through priority.

Hand poses use this branch order after both hands are present:

1. `hands_on_face`
2. `head_grab`
3. `hands_on_cheeks`
4. `t_sign`
5. `left_nose_right_wave`
6. `explaining_hands`

Only the first matching branch contributes a candidate. This deliberately prevents ambiguous hand positions from creating several low-level pose candidates. `t_sign` also suppresses `open_palm`. `open_palm` and `explaining_hands` are suppressed whenever a hand is near the nose.

For 0.35 seconds after `head_grab`, `hands_on_cheeks`, or a qualifying hands-on-face pose, `explaining_hands` remains suppressed. This avoids motion recognition treating face-adjacent poses as hand talking.

## Recognition Details

### Nose Proximity

A hand is "near" the nose when any of its landmarks is closer to the nose than:

```text
max(face_width * 0.25, 0.05)
```

`face_width` is the horizontal distance between face landmarks 234 and 454. The minimum prevents the threshold becoming too small for distant faces.

### Hands On Face

The two palm landmarks (index 9) must be closer than `face_width * 0.6`, or their middle-finger tips (index 12) must be closer than `face_width * 0.35`. This permits palms placed across the cheeks when fingertips meet over the nose or mouth. The palm midpoint must be horizontally within `face_width * 0.35` of the nose and vertically between `face[168].y - 0.05` and the chin landmark `face[152].y`. Both middle-finger tips must be above their wrists. The condition must survive three qualifying detections before it becomes a candidate.

### Head Grab And Cheeks

For head-grab, each palm must be above the average eye y-coordinate plus `face_width * 0.30`; one palm must be left of the nose by more than `face_width * 0.1`, and the other must be right by the same margin.

For cheeks, each palm must be in an opposite side zone. The vertical zone begins at the eye line plus `face_width * 0.32` and ends at the chin. The horizontal zone reaches from `face_width * 0.50` outside the face to `face_width * 0.10` from the nose. The palms must also be farther apart than `face_width * 0.30`.

### T-Sign

Each hand is open when at least three of four fingertip-to-wrist distances exceed its corresponding PIP-to-wrist distance. The palm separation must be less than twice the average wrist-to-palm size. One hand's middle-finger direction must be sufficiently horizontal and the other's sufficiently vertical, using a direction ratio of 0.75.

### Motion Gestures

Wrist samples are retained for 0.5 seconds. `left_nose_right_wave` needs at least three right-wrist samples that span at least 0.15 seconds and have a y-range of at least 0.065. It does not count discrete wave cycles; it measures vertical range.

`explaining_hands` sums the Euclidean path lengths between consecutive wrist samples for both hands. The sum must be at least 0.20 over the retained time window.

## Why A Matching Gesture May Not Activate

- A higher-priority gesture is selected instead.
- The gesture has not been selected continuously for its debounce duration.
- Its prior activation is still visible.
- The gesture is still consumed because it never cleared after the prior activation.
- Its cooldown is active.
- A required face or hand track is missing.
- A hand-pose branch earlier in the ordering matched first.

Use `uv run main.py --debug` to see the measured values and conditions. See [Troubleshooting](TROUBLESHOOTING.md#gesture-problems) for practical calibration guidance.
