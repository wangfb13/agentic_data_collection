"""Offline RSI proposal protocol tests: no model inference or robot access."""
import copy
import hashlib
import json
import unittest
from unittest.mock import Mock, patch

from bottle_scan import rsi_proposer as proposer
from bottle_scan.agentic_provider import ProviderError
from bottle_scan.rsi_methods import baseline_method


def context():
    baseline = baseline_method()
    return {
        "protocol_id": "fixed_protocol_test",
        "valid_evidence_refs": ["episode:problem_1", "experience:failed_duplicates"],
        "references": {"episode:problem_1": {"duplicate_fraction": 0.8},
                       "experience:failed_duplicates": {"status": "not_effective"}},
        "method_library": [{"method": baseline, "status": "baseline"}],
        "baseline_method_id": baseline["method_id"],
        "reusable_method_ids": [baseline["method_id"]],
        "available_features": ["failure_rate", "uncertainty", "novelty", "action_conflict", "coverage_gap"],
        "method_experiences": [{"method_id": baseline["method_id"],
                                "problem": "重复采样没有改善缺少历史的观测歧义", "outcome": "not_effective"}],
    }


def decision():
    return {"action": "propose_method", "reason": "用固定预算比较去重复与失败优先，不能由一次成功率上升认定有效。",
            "method_id": None,
            "method": {
                "name": "冲突与覆盖联合选样",
                "parent_id": baseline_method()["method_id"],
                "hypothesis": "相似观测下动作分歧可能提示遗漏的历史；去重复补采可能提高数据效率。",
                "applicability": "只适用于已提供动作分歧统计的一维接触任务；尚未证明跨问题迁移。",
                "score": {"op": "add", "args": [
                    {"op": "mul", "args": [{"op": "feature", "name": "action_conflict"},
                                              {"op": "feature", "name": "novelty"}]},
                    {"op": "feature", "name": "coverage_gap"}]},
                "sampling": {"step_mm": 1.5, "extra_offsets_mm": [-3.5, 2.],
                             "max_reps_per_cell": 4, "diversity_mm": 1.},
                "evidence_refs": ["episode:problem_1", "experience:failed_duplicates"],
                "expected_outcome": "相同起点和预算下若独立评估未优于旧法，则不支持此方法。",
            }}


class FakeProvider:
    def __init__(self, values):
        self.values = list(values)
        self.calls = []
        self.closed = False

    def _infer(self, instructions, prompt):
        self.calls.append((instructions, prompt))
        value = self.values.pop(0)
        if isinstance(value, Exception):
            raise value
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        return text, {"backend": "chatgpt", "model": "offline-fake", "completed": True,
                      "response_id": "resp_test", "usage": {"input_tokens": 12, "output_tokens": 8},
                      "access_token": "never-copy-this-secret", "raw_headers": "Bearer never-copy"}

    def close(self):
        self.closed = True


class MethodProposerTests(unittest.TestCase):
    def test_new_method_has_provenance_snapshot_and_reuses_existing_transport(self):
        fake = FakeProvider([decision()])
        ctx = context()
        original = copy.deepcopy(ctx)
        result = proposer.propose_method({"backend": "chatgpt"}, ctx, provider=fake)
        self.assertEqual(result["action"], "propose_method")
        self.assertEqual(result["method"]["schema"], "rsi_method_v1")
        self.assertEqual(len(result["method"]["method_id"]), 64)
        self.assertEqual(ctx, original)
        self.assertFalse(fake.closed)  # caller owns injected provider
        self.assertEqual(fake.calls[0][0], proposer.RSI_SYSTEM)
        self.assertEqual(json.loads(fake.calls[0][1]), ctx)
        audit = result["audit"]
        self.assertEqual(audit["system_prompt"], proposer.RSI_SYSTEM)
        self.assertEqual(audit["system_prompt_sha256"], hashlib.sha256(proposer.RSI_SYSTEM.encode()).hexdigest())
        self.assertEqual(audit["attempts"][0]["input_json"], fake.calls[0][1])
        self.assertEqual(audit["attempts"][0]["input_sha256"], hashlib.sha256(fake.calls[0][1].encode()).hexdigest())
        self.assertEqual(audit["outcome"], "accepted_for_evaluation")
        self.assertTrue(result["_meta"]["live_inference"])
        self.assertNotIn("never-copy", json.dumps(result))
        self.assertNotIn("raw_headers", json.dumps(result))

    def test_schema_repair_is_bounded_and_preserves_failed_attempt(self):
        first = decision()
        first["method"]["sampling"]["extra_offsets_mm"] = [26.]
        second = decision()
        second["method"]["evidence_refs"] = ["invented-evidence"]
        fake = FakeProvider([first, second, decision()])
        result = proposer.propose_method({"backend": "api"}, context(), provider=fake)
        self.assertEqual(len(fake.calls), 3)
        attempts = result["audit"]["attempts"]
        self.assertEqual(attempts[0]["rejection"]["code"], "method_schema")
        self.assertEqual(attempts[1]["rejection"]["code"], "unknown_evidence")
        self.assertTrue(attempts[2]["accepted"])
        revision = json.loads(fake.calls[2][1])["_rsi_revision"]
        self.assertEqual(revision["attempt"], 3)
        self.assertEqual(len(revision["rejections"]), 2)
        self.assertEqual(revision["previous_proposal"], second)

    def test_only_three_invalid_proposals_then_pause_with_audit(self):
        fake = FakeProvider([{}, {}, {}, decision()])
        with self.assertRaises(proposer.RSIProposalError) as caught:
            proposer.propose_method({"backend": "chatgpt"}, context(), provider=fake)
        self.assertEqual(caught.exception.code, "invalid_proposals_exhausted")
        self.assertEqual(len(fake.calls), 3)
        self.assertEqual(len(caught.exception.audit["attempts"]), 3)

    def test_transport_failure_is_immediate_sanitized_and_never_falls_back(self):
        fake = FakeProvider([ProviderError("Bearer secret-client-token and private proxy password"), decision()])
        with self.assertRaises(proposer.RSIProposalError) as caught:
            proposer.propose_method({"backend": "chatgpt"}, context(), provider=fake)
        self.assertEqual(caught.exception.code, "provider_failure")
        self.assertEqual(len(fake.calls), 1)
        self.assertNotIn("secret-client", str(caught.exception))
        self.assertNotIn("secret-client", json.dumps(caught.exception.audit))
        self.assertFalse(caught.exception.audit["live_inference"])
        self.assertEqual(caught.exception.audit["attempts"][0]["rejection"]["code"], "provider_failure")

    def test_invalid_json_duplicate_keys_nonfinite_and_markdown_do_not_pass(self):
        responses = [
            '{"action":"pause","action":"reuse_method","reason":"x","method":null,"method_id":null}',
            '{"action":"pause","reason":"x","method":NaN,"method_id":null}',
            '{"action":"pause","reason":"x","method":1e999,"method_id":null}',
            '```json\n{"action":"pause","reason":"x","method":null,"method_id":null}\n```',
        ]
        for text in responses:
            with self.subTest(text=text):
                fake = FakeProvider([text])
                with self.assertRaises(proposer.RSIProposalError) as caught:
                    proposer.propose_method({"backend": "api", "max_proposal_attempts": 1}, context(), provider=fake)
                entry = caught.exception.audit["attempts"][0]
                self.assertEqual(entry["rejection"]["code"], "invalid_json")
                self.assertIsNone(entry["proposal"])
                self.assertEqual(entry["response_sha256"], hashlib.sha256(text.encode()).hexdigest())

    def test_arbitrary_code_and_safety_fields_are_rejected(self):
        mutations = [
            lambda x: x.update(action="run_shell"),
            lambda x: x.update(robot_command="move"),
            lambda x: x["method"].update(force_max=100),
            lambda x: x["method"].update(schema="rsi_method_v1"),
            lambda x: x["method"].update(score={"op": "eval", "expression": "import os"}),
            lambda x: x["method"]["sampling"].update(max_reps_per_cell=True),
            lambda x: x["method"].update(parent_id="unknown-parent"),
        ]
        for mutation in mutations:
            value = decision()
            mutation(value)
            with self.subTest(value=value), self.assertRaises(proposer.RSIProposalError):
                proposer.propose_method({"backend": "api", "max_proposal_attempts": 1}, context(),
                                        provider=FakeProvider([value]))

    def test_reuse_needs_explicit_permission_not_just_library_membership(self):
        ctx = context()
        ctx["method_library"].append({"method_id": "unvalidated-candidate", "status": "proposed"})
        value = {"action": "reuse_method", "reason": "复用", "method": None, "method_id": "unvalidated-candidate"}
        with self.assertRaises(proposer.RSIProposalError) as caught:
            proposer.propose_method({"backend": "api", "max_proposal_attempts": 1}, ctx,
                                    provider=FakeProvider([value]))
        self.assertEqual(caught.exception.audit["attempts"][0]["rejection"]["code"], "reuse_not_allowed")
        value["method_id"] = ctx["baseline_method_id"]
        result = proposer.propose_method({"backend": "api"}, ctx, provider=FakeProvider([value]))
        self.assertEqual(result["action"], "reuse_method")

    def test_unavailable_feature_is_not_silently_zero_filled(self):
        ctx = context()
        ctx["available_features"].remove("action_conflict")
        with self.assertRaises(proposer.RSIProposalError) as caught:
            proposer.propose_method({"backend": "api", "max_proposal_attempts": 1}, ctx,
                                    provider=FakeProvider([decision()]))
        self.assertEqual(caught.exception.audit["attempts"][0]["rejection"]["code"], "feature_unavailable")

    def test_explicit_mock_and_deterministic_do_not_infer_or_invent(self):
        for backend in ("mock", "deterministic"):
            with self.subTest(backend=backend), patch.object(proposer, "AgentProvider") as make:
                fake = Mock()
                result = proposer.propose_method({"backend": backend}, context(), provider=fake)
                make.assert_not_called()
                fake._infer.assert_not_called()
                self.assertEqual(result["action"], "reuse_method")
                self.assertIsNone(result["method"])
                self.assertEqual(result["method_id"], context()["baseline_method_id"])
                self.assertFalse(result["_meta"]["live_inference"])
                self.assertEqual(result["_meta"]["backend"], backend)
                self.assertEqual(result["audit"]["attempts"], [])
                self.assertIn("未调用 LLM", result["reason"])
        result = proposer.propose_method({"backend": "mock"}, {})
        self.assertEqual(result["action"], "pause")

    def test_pause_carries_no_hidden_method_and_records_complete_prompt(self):
        value = {"action": "pause", "reason": "当前单帧观测无法判别；需要先验证输入能力。", "method": None, "method_id": None}
        result = proposer.propose_method({"backend": "api"}, context(), provider=FakeProvider([value]))
        self.assertEqual(result["audit"]["outcome"], "pause")
        value["method"] = decision()["method"]
        with self.assertRaises(proposer.RSIProposalError):
            proposer.propose_method({"backend": "api", "max_proposal_attempts": 1}, context(), provider=FakeProvider([value]))

    def test_owned_provider_uses_same_auth_transport_and_closes_on_success_or_error(self):
        for value in (decision(), RuntimeError("secret exception")):
            fake = FakeProvider([value])
            settings = {"backend": "api", "model": "test-model", "proxy_url": "http://localhost:1234"}
            with patch.object(proposer, "AgentProvider", return_value=fake) as factory:
                if isinstance(value, Exception):
                    with self.assertRaises(proposer.RSIProposalError):
                        proposer.propose_method(settings, context())
                else:
                    proposer.propose_method(settings, context())
                factory.assert_called_once_with(settings)
                self.assertTrue(fake.closed)

    def test_credentials_never_enter_prompt_and_untrusted_output_keys_are_redacted(self):
        ctx = context()
        ctx["nested"] = {"access_token": "private-secret-value"}
        fake = FakeProvider([])
        with self.assertRaises(proposer.RSIProposalError) as caught:
            proposer.propose_method({"backend": "api"}, ctx, provider=fake)
        self.assertEqual(fake.calls, [])
        self.assertNotIn("private-secret", json.dumps(caught.exception.audit))
        response = decision()
        response["access_token"] = "private-secret-value"
        fake = FakeProvider([response])
        with self.assertRaises(proposer.RSIProposalError) as caught:
            proposer.propose_method({"backend": "api", "max_proposal_attempts": 1}, context(), provider=fake)
        self.assertNotIn("private-secret", json.dumps(caught.exception.audit))
        self.assertEqual(caught.exception.audit["attempts"][0]["proposal"]["access_token"], "[REDACTED]")

    def test_invalid_input_or_retry_budget_never_reaches_provider(self):
        variants = [({"backend": "api"}, {"score": float("nan")}),
                    ({"backend": "api"}, {"_rsi_revision": {}}),
                    ({"backend": "api", "max_proposal_attempts": 4}, context()),
                    ({"backend": "api", "max_proposal_attempts": True}, context()),
                    ({"backend": "unknown"}, context()),
                    ({"backend": "api"}, {"available_features": ["delta_at_runtime"]})]
        for settings, ctx in variants:
            fake = FakeProvider([])
            with self.subTest(settings=settings, ctx=ctx), self.assertRaises(proposer.RSIProposalError):
                proposer.propose_method(settings, ctx, provider=fake)
            self.assertEqual(fake.calls, [])

    def test_prompt_focuses_on_learning_efficiency_history_and_falsification(self):
        for phrase in ("HOW the system acquires", "negative", "counterexamples", "same data/policy checkpoint",
                       "equal", "historical actions", "not a whitelist", "independent", "never",
                       "cannot authorize real robot motion"):
            self.assertIn(phrase, proposer.RSI_SYSTEM)


if __name__ == "__main__":
    unittest.main()
