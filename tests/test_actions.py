import unittest
from types import SimpleNamespace as Point

from rem.actions import detect_candidates, hands_on_cheeks, hands_on_head, t_sign_metrics, update_active_media
from rem.models import RuntimeState


def point(x, y):
    return Point(x=x, y=y)


def make_face():
    face = [point(0.5, 0.5) for _ in range(455)]
    face[1] = point(0.5, 0.5)
    face[13] = point(0.5, 0.6)
    face[14] = point(0.5, 0.6)
    face[33] = point(0.4, 0.4)
    face[263] = point(0.6, 0.4)
    face[152] = point(0.5, 0.8)
    face[168] = point(0.5, 0.45)
    face[234] = point(0.3, 0.55)
    face[454] = point(0.7, 0.55)
    return face


def make_hand(x, y):
    return [point(x, y) for _ in range(21)]


class GestureActionsTest(unittest.TestCase):
    def setUp(self):
        self.face = make_face()
        self.result = Point(gestures=[])

    def test_head_grab_and_cheeks_are_distinct(self):
        head_left, head_right = make_hand(0.35, 0.50), make_hand(0.65, 0.50)
        self.assertTrue(hands_on_head(head_left, head_right, self.face))
        self.assertFalse(hands_on_cheeks(head_left, head_right, self.face))
        candidates = detect_candidates(
            self.face, [], {"Left": head_left, "Right": head_right}, self.result, RuntimeState(), 0.0
        )
        self.assertEqual(candidates, {"head_grab"})

        cheek_left, cheek_right = make_hand(0.32, 0.60), make_hand(0.68, 0.60)
        self.assertTrue(hands_on_cheeks(cheek_left, cheek_right, self.face))
        self.assertFalse(hands_on_head(cheek_left, cheek_right, self.face))
        candidates = detect_candidates(
            self.face, [], {"Left": cheek_left, "Right": cheek_right}, self.result, RuntimeState(), 0.0
        )
        self.assertEqual(candidates, {"hands_on_cheeks"})

    def test_hands_on_face_confirms_after_three_frames(self):
        left, right = make_hand(0.46, 0.7), make_hand(0.54, 0.7)
        left[9], right[9] = point(0.46, 0.55), point(0.54, 0.55)
        left[12], right[12] = point(0.46, 0.3), point(0.54, 0.3)
        state = RuntimeState()
        for _ in range(2):
            self.assertNotIn(
                "hands_on_face",
                detect_candidates(self.face, [], {"Left": left, "Right": right}, self.result, state, 0.0),
            )
        candidates = detect_candidates(self.face, [], {"Left": left, "Right": right}, self.result, state, 0.0)
        self.assertEqual(candidates, {"hands_on_face"})

    def test_fingertips_together_on_face_suppresses_cheeks(self):
        left, right = make_hand(0.36, 0.70), make_hand(0.64, 0.70)
        left[9], right[9] = point(0.36, 0.60), point(0.64, 0.60)
        left[12], right[12] = point(0.48, 0.45), point(0.52, 0.45)
        self.assertTrue(hands_on_cheeks(left, right, self.face))
        state = RuntimeState()
        for _ in range(2):
            self.assertNotIn(
                "hands_on_cheeks",
                detect_candidates(self.face, [], {"Left": left, "Right": right}, self.result, state, 0.0),
            )
        candidates = detect_candidates(self.face, [], {"Left": left, "Right": right}, self.result, state, 0.0)
        self.assertEqual(candidates, {"hands_on_face"})

    def test_t_sign_suppresses_open_palm_and_explaining(self):
        left, right = make_hand(0.4, 0.5), make_hand(0.5, 0.7)
        left[9], right[9] = point(0.5, 0.5), point(0.5, 0.6)
        for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
            left[tip], left[pip] = point(0.6, 0.5), point(0.55, 0.5)
            right[tip], right[pip] = point(0.5, 0.4), point(0.5, 0.55)
        self.assertTrue(t_sign_metrics(left, right).matches)
        result = Point(gestures=[[Point(category_name="Open_Palm")]])
        candidates = detect_candidates(self.face, [], {"Left": left, "Right": right}, result, RuntimeState(), 0.0)
        self.assertEqual(candidates, {"t_sign"})

    def test_head_grab_activates_after_its_short_debounce(self):
        state = RuntimeState()
        update_active_media(state, {"head_grab"}, True, 0.0)
        update_active_media(state, {"head_grab"}, True, 0.06)
        self.assertEqual(state.active.name, "head_grab")


if __name__ == "__main__":
    unittest.main()
