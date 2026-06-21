import unittest

from catchup.components import PLAYER_ONE, PLAYER_TWO
from catchup.game import FINISH, GameState
from catchup.ui_server import PLAYER_NAMES, GameSession, action_description, state_payload


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

    def test_session_suggest_action_does_not_apply_action(self) -> None:
        session = GameSession()

        payload = session.suggest_action(simulations=4)
        suggestion = payload["suggestion"]
        choices = payload["choices"]
        state = payload["state"]

        self.assertIn(suggestion["action"], GameState.new().legal_actions())
        self.assertEqual(suggestion["player"], PLAYER_ONE)
        self.assertEqual(suggestion["player_name"], PLAYER_NAMES[PLAYER_ONE])
        self.assertEqual(suggestion["simulations"], 4)
        self.assertIn("Claim #", suggestion["label"])
        self.assertEqual(suggestion["action"], choices[0]["action"])
        self.assertEqual(sum(choice["visits"] for choice in choices), 4)
        self.assertEqual(
            [choice["visits"] for choice in choices],
            sorted((choice["visits"] for choice in choices), reverse=True),
        )
        self.assertEqual(state["current_player"], PLAYER_ONE)
        self.assertEqual(state["empty_count"], 61)

    def test_action_description_formats_finish_and_claim(self) -> None:
        state = GameState.new()

        claim = action_description(state, 30)
        finish = action_description(state, FINISH)

        self.assertEqual(claim["kind"], "claim")
        self.assertEqual(claim["label"], "Claim #30 (0,0)")
        self.assertEqual(finish["kind"], "finish")
        self.assertEqual(finish["label"], "Finish turn")


if __name__ == "__main__":
    unittest.main()
