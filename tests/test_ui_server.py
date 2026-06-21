import unittest

from catchup.components import PLAYER_ONE, PLAYER_TWO
from catchup.game import FINISH, GameState
from catchup.ui_server import PLAYER_NAMES, GameSession, state_payload


class UiServerTest(unittest.TestCase):
    def test_state_payload_includes_component_sizes_for_both_players(self) -> None:
        state = GameState.new()
        state = state.apply_action(0)
        state = state.apply_action(1)
        state = state.apply_action(2)

        payload = state_payload(state)
        players = {player["id"]: player for player in payload["players"]}

        self.assertEqual(players[PLAYER_ONE]["name"], PLAYER_NAMES[PLAYER_ONE])
        self.assertEqual(players[PLAYER_ONE]["group_sizes"], (1,))
        self.assertEqual(players[PLAYER_TWO]["name"], PLAYER_NAMES[PLAYER_TWO])
        self.assertEqual(players[PLAYER_TWO]["group_sizes"], (2,))
        self.assertEqual(players[PLAYER_TWO]["largest_group"], 2)

    def test_state_payload_includes_empty_component_boundaries(self) -> None:
        state = GameState.new().apply_action(30)

        payload = state_payload(state)

        self.assertEqual(len(payload["empty_components"]), 1)
        region = payload["empty_components"][0]
        self.assertEqual(region["size"], 60)
        self.assertNotIn(30, region["cells"])
        self.assertEqual(region["blue"], [{"root": 30, "size": 1}])
        self.assertEqual(region["white"], [])

    def test_session_action_reset_and_undo(self) -> None:
        session = GameSession()

        after_claim = session.apply_action(0)
        self.assertEqual(after_claim["players"][0]["group_sizes"], (1,))
        self.assertEqual(after_claim["current_player"], PLAYER_TWO)

        undone = session.undo()
        self.assertEqual(undone["players"][0]["group_sizes"], ())
        self.assertEqual(undone["current_player"], PLAYER_ONE)

        reset = session.reset()
        self.assertEqual(reset["finish_action"], FINISH)
        self.assertEqual(reset["empty_count"], 61)

    def test_session_allows_human_out_of_order_multi_cell_turn(self) -> None:
        session = GameSession()
        session.apply_action(22)
        after_first_white_cell = session.apply_action(21)

        self.assertIn(0, after_first_white_cell["legal_actions"])

        after_second_white_cell = session.apply_action(0)
        players = {player["id"]: player for player in after_second_white_cell["players"]}

        self.assertEqual(after_second_white_cell["current_player"], PLAYER_ONE)
        self.assertEqual(after_second_white_cell["completed_turns"], 2)
        self.assertEqual(players[PLAYER_TWO]["group_sizes"], (1, 1))


if __name__ == "__main__":
    unittest.main()
