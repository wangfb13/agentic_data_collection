"""Director protocol transport tests; no network, OAuth, robot or inference."""
import json
import unittest
from unittest.mock import patch

from bottle_scan.agentic_provider import AgentProvider, ProviderError
from tests import test_agentic_provider as fixtures

stream = fixtures.stream


class DirectorProviderTests(unittest.TestCase):
    make = fixtures.ProviderTests.make

    def test_director_request_is_distinct_and_never_has_robot_tools(self):
        from bottle_scan.skill_director import DIRECTOR_SYSTEM
        context = {"allowed_actions": ["train", "pause"],
                   "valid_evidence_refs": ["A:1"], "target_definition": {"id": "state-set-v2"}}
        decision = {"action": "train", "reason": "根据已复核样本联合学习三个组件。",
                    "evidence_refs": ["A:1"], "plan": None}
        instance, client = self.make(stream(json.dumps(decision)))
        answer = instance.direct(context)
        self.assertEqual(answer["action"], "train")
        self.assertEqual(answer["_meta"]["planning_mode"], "director")
        payload = client.post.call_args.kwargs["json"]
        self.assertEqual(payload["instructions"], DIRECTOR_SYSTEM)
        self.assertEqual(json.loads(payload["input"][0]["content"]), context)
        self.assertFalse(payload["store"])
        self.assertNotIn("tools", payload)

    def test_director_never_executes_code_or_changes_truth(self):
        base = {"action": "train", "reason": "evidence", "evidence_refs": ["A:1"], "plan": None}
        for value in (dict(base, action="run_shell"), dict(base, labels={"1": True}),
                      dict(base, _meta={"token": "secret"}), dict(base, plan={"command": "move"})):
            instance, _ = self.make(stream(json.dumps(value)))
            with self.subTest(value=value), self.assertRaises(ProviderError):
                instance.direct({})

    def test_duplicate_json_and_incomplete_stream_fail_without_fallback(self):
        for events in (stream('{"action":"pause","action":"complete"}'),
                       stream("{}", event="response.incomplete", status="incomplete")):
            instance, _ = self.make(events)
            with self.assertRaises(ProviderError):
                instance.direct({})

    def test_mock_only_pauses_without_network(self):
        with patch("bottle_scan.agentic_provider.new_session", side_effect=AssertionError("network")):
            answer = AgentProvider({"backend": "mock"}).direct({})
        self.assertEqual(answer["action"], "pause")
        self.assertEqual(answer["_meta"]["backend"], "mock")

    def test_invalid_context_never_reaches_transport(self):
        instance, client = self.make()
        with self.assertRaises(ProviderError):
            instance.direct({"x": float("nan")})
        client.post.assert_not_called()
