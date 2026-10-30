"""Offline OAuth/provider tests. No network, ROS, actuator or browser action."""
import copy
import json
import os
from pathlib import Path
import stat
import tempfile
import time
import unittest
from unittest.mock import Mock, patch
import urllib.parse

from bottle_scan import agentic_auth as auth
from bottle_scan import agentic_provider as provider


def response(data=None, status=200, events=None):
    value = Mock(status_code=status, headers={"x-request-id": "req_test"})
    value.json.return_value = data
    value.iter_lines.return_value = iter(events or [])
    return value


def stream(text, *, status="completed", event="response.completed", output=None):
    terminal = {"id": "resp_test", "status": status,
                "usage": {"input_tokens": 20, "output_tokens": 8, "total_tokens": 28,
                          "untrusted": "not logged"}}
    if output is not None:
        terminal["output"] = output
    return [b"event: response.output_text.delta", "data: " + json.dumps(
        {"type": "response.output_text.delta", "delta": text}),
        "", "data: " + json.dumps({"type": event, "response": terminal})]


def context():
    return {"ranked_candidates": [{"id": "a_0_2", "cond": "A", "score": 1.2}],
            "approved_models": [{"model_id": "fit_1"}], "stop": {"eligible": False}}


class ProviderTests(unittest.TestCase):
    def make(self, events=None, status=200):
        instance = provider.AgentProvider({"backend": "api", "model": "test-model"})
        instance._token = Mock(return_value="super-secret-access")
        client = Mock()
        client.post.return_value = response(status=status, events=events)
        instance._client = client
        return instance, client

    def test_complete_response_selects_only_a_candidate_and_has_no_tools(self):
        decision = {"action": "select_batch", "candidate_id": "a_0_2", "reason": "检查零点", "model_id": "fit_1"}
        instance, client = self.make(stream(json.dumps(decision)))
        result = instance.propose(context())
        self.assertEqual(result["candidate_id"], "a_0_2")
        self.assertEqual(result["_meta"]["usage"]["total_tokens"], 28)
        self.assertNotIn("untrusted", result["_meta"]["usage"])
        self.assertNotIn("super-secret", json.dumps(result))
        args, kwargs = client.post.call_args
        self.assertEqual(args[0], auth.API_BASE + "/responses")
        self.assertIs(kwargs["json"]["store"], False)
        self.assertIs(kwargs["json"]["stream"], True)
        self.assertNotIn("tools", kwargs["json"])
        self.assertFalse(kwargs["allow_redirects"])

    def test_stream_failure_partial_json_and_disconnect_never_fallback(self):
        for events in (["data: " + json.dumps({"type": "response.output_text.delta", "delta": "{}"})],
                       stream("{}", event="response.failed", status="failed"),
                       stream("{}", event="response.incomplete", status="incomplete")):
            instance, _ = self.make(events)
            with self.subTest(events=events), self.assertRaises(provider.ProviderError):
                instance.propose(context())

    def test_strict_schema_rejects_injected_motion_unknown_candidate_and_fit(self):
        original = {"action": "select_batch", "candidate_id": "a_0_2", "reason": "x"}
        variants = [dict(original, move_tcp=[0, 0, 1]), dict(original, candidate_id="invented"),
                    dict(original, model_id="unapproved"), dict(original, action="run_shell"),
                    dict(original, action="pause"), dict(original, reason="")]
        for value in variants:
            instance, _ = self.make(stream(json.dumps(value)))
            with self.subTest(value=value), self.assertRaises(provider.ProviderError):
                instance.propose(context())

    def test_duplicate_json_keys_nan_and_markdown_are_rejected(self):
        for text in ('{"action":"pause","action":"freeze","candidate_id":"","reason":"x"}',
                     '{"action":"pause","candidate_id":"","reason":NaN}', '```json\n{}\n```'):
            instance, _ = self.make(stream(text))
            with self.subTest(text=text), self.assertRaises(provider.ProviderError):
                instance.propose(context())

    def test_response_tool_or_refusal_is_rejected(self):
        for output in ([{"type": "function_call", "name": "move"}],
                       [{"type": "message", "content": [{"type": "refusal", "refusal": "x"}]}]):
            instance, _ = self.make(stream('{"action":"pause","candidate_id":"","reason":"x"}', output=output))
            with self.subTest(output=output), self.assertRaises(provider.ProviderError):
                instance.propose(context())

    def test_statuses_do_not_leak_body_or_credential(self):
        for status in (401, 403, 429, 500, 302):
            instance, client = self.make(status=status)
            client.post.return_value.text = "super-secret-body"
            with self.subTest(status=status), self.assertRaises(provider.ProviderError) as caught:
                instance.propose(context())
            self.assertNotIn("secret", str(caught.exception))
            self.assertEqual(client.post.call_count, 1)

    def test_transport_exception_is_sanitized(self):
        instance, client = self.make()
        client.post.side_effect = RuntimeError("Bearer super-secret-access and proxy password")
        with self.assertRaises(provider.ProviderError) as caught:
            instance.propose(context())
        self.assertNotIn("secret", str(caught.exception))

    def test_subscription_catalog_resolves_model_without_claiming_inference(self):
        instance = provider.AgentProvider({"backend": "chatgpt", "model": None})
        client = Mock()
        client.get.return_value = response({"models": [
            {"slug": "hidden-model", "visibility": "hidden"},
            {"slug": "gpt-other", "visibility": "list"},
            {"slug": "gpt-sol", "visibility": "list", "display_name": "Sol"}]})
        instance._client, instance._token = client, Mock(return_value="secret")
        self.assertEqual(instance._model(), "gpt-sol")
        self.assertEqual(instance._model(), "gpt-sol")
        self.assertEqual(client.get.call_count, 1)
        client.post.assert_not_called()

    def test_connection_requires_completed_inference(self):
        instance, client = self.make(stream('{"ok":true}'))
        self.assertTrue(instance.test_connection()["live_inference"])
        client.post.return_value = response(events=["data: " + json.dumps({"type": "response.created"})])
        with self.assertRaises(provider.ProviderError):
            instance.test_connection()

    def test_mock_is_explicit_and_never_claims_live_inference(self):
        with patch.object(provider, "new_session") as client:
            instance = provider.AgentProvider({"backend": "mock"})
            self.assertEqual(instance.propose(context())["_meta"]["backend"], "mock")
            self.assertFalse(instance.test_connection()["live_inference"])
            client.assert_not_called()
        with self.assertRaises(provider.ProviderError):
            provider.AgentProvider({"backend": "guess"})

    def test_api_secret_is_only_used_when_api_selected(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "secret"}):
            instance = provider.AgentProvider({"backend": "chatgpt"})
            with patch.object(provider, "access_token", side_effect=auth.AuthError("sign in required")):
                with self.assertRaises(provider.ProviderError):
                    instance._token()
            instance.close()

    def test_no_nonofficial_api_endpoint(self):
        for url in ("http://api.openai.com/v1", "https://chatgpt.com/backend-api", "https://example.com/v1"):
            with self.subTest(url=url), self.assertRaises(provider.ProviderError):
                provider.AgentProvider({"base_url": url})


def agent_context():
    return {"planning_mode": "agent", "suggestions": [
                {"cond": "A", "cells": [{"offset_mm": 0., "reps": 2}, {"offset_mm": 4., "reps": 2}]}],
            "approved_models": [{"model_id": "fit_1"}],
            "references": {"level:zero": {"offset_mm": 0., "bias_mm": 1.2},
                           "level:four": {"offset_mm": 4., "bias_mm": 2.1}},
            "valid_evidence_refs": ["level:zero", "level:four"],
            "stop": {"eligible": False}, "allowed_gains": [.5, .75, 1.]}


def agent_decision():
    return {"action": "propose_batch", "reason": "测量局部斜率变化，区分零点偏差与非线性。",
            "plan": {"cond": "A", "cells": [{"offset_mm": 1.75, "reps": 4}, {"offset_mm": 3.25, "reps": 2}],
                     "model_id": None, "gain": 1.,
                     "hypothesis": "零点至正向区间的误差可能来自局部斜率改变。",
                     "evidence_refs": ["level:zero", "level:four"],
                     "expected_outcome": "若新位置仍呈同向等量误差则支持零点偏差；若随位置变化则支持非线性。"}}


class AgentPlanningProviderTests(unittest.TestCase):
    make = ProviderTests.make

    def test_agent_formulates_novel_offsets_without_candidate_whitelist(self):
        value = agent_decision()
        instance, client = self.make(stream(json.dumps(value, ensure_ascii=False)))
        result = instance.propose(agent_context())
        self.assertEqual(result['plan']['cells'], value['plan']['cells'])
        self.assertNotIn('candidate_id', result)
        self.assertEqual(result['_meta']['planning_mode'], 'agent')
        payload = client.post.call_args.kwargs['json']
        self.assertEqual(payload['instructions'], provider.AGENT_SYSTEM)
        self.assertIn('NOT a whitelist', payload['instructions'])
        self.assertIn('a midpoint', payload['instructions'])
        self.assertIn('hypothesis', payload['instructions'])
        self.assertEqual(json.loads(payload['input'][0]['content']), agent_context())
        self.assertNotIn('tools', payload)
        self.assertNotIn('text', payload)  # Local validation, not claimed constrained decoding.
        self.assertFalse(payload['store'])
        self.assertTrue(payload['stream'])

    def test_agent_mode_does_not_require_or_validate_old_candidate_ids(self):
        ctx = agent_context()
        ctx['ranked_candidates'] = [{'missing_id': True}]
        instance, _ = self.make(stream(json.dumps(agent_decision())))
        self.assertEqual(instance.propose(ctx)['action'], 'propose_batch')

    def test_old_campaign_retains_original_prompt_and_protocol(self):
        value = {'action': 'select_batch', 'candidate_id': 'a_0_2', 'reason': '复测零点'}
        instance, client = self.make(stream(json.dumps(value)))
        result = instance.propose(context())
        self.assertEqual(result['candidate_id'], 'a_0_2')
        self.assertEqual(client.post.call_args.kwargs['json']['instructions'], provider.SYSTEM)
        self.assertNotIn('planning_mode', result['_meta'])

    def test_agent_shape_rejections_are_sanitized_retryable_proposal_errors(self):
        original = agent_decision()
        extra_plan = copy.deepcopy(original); extra_plan['plan']['robot_command'] = 'super-secret-raw-response'
        extra_cell = copy.deepcopy(original); extra_cell['plan']['cells'][0]['move_tcp'] = 'super-secret-raw-response'
        bad_reps = copy.deepcopy(original); bad_reps['plan']['cells'][0]['reps'] = True
        bad_cond = copy.deepcopy(original); bad_cond['plan']['cond'] = 'super-secret-raw-response'
        wrong_protocol = {'action': 'select_batch', 'candidate_id': 'a_0_2', 'reason': 'super-secret-raw-response'}
        variants = [extra_plan, extra_cell, bad_reps, bad_cond, wrong_protocol,
                    dict(original, _meta={'access_token': 'super-secret-raw-response'}),
                    dict(original, action='pause')]
        for value in variants:
            instance, _ = self.make(stream(json.dumps(value)))
            with self.subTest(value=value), self.assertRaises(provider.ProposalError) as caught:
                instance.propose(agent_context())
            self.assertTrue(caught.exception.retryable)
            self.assertEqual(caught.exception.sanitized_reason, str(caught.exception))
            self.assertNotIn('secret', str(caught.exception))
            self.assertNotIn('secret', caught.exception.code)

    def test_strict_model_json_failures_are_repairable_in_both_protocols(self):
        values = ['```json\n{}\n```', '{"action":"pause","action":"freeze","reason":"x","plan":null}',
                  '{"action":"pause","reason":NaN,"plan":null}', 'super-secret-raw-response']
        for ctx in (context(), agent_context()):
            for value in values:
                instance, _ = self.make(stream(value))
                with self.subTest(ctx=ctx, value=value), self.assertRaises(provider.ProposalError) as caught:
                    instance.propose(ctx)
                self.assertEqual(caught.exception.code, 'invalid_json')
                self.assertNotIn('secret', str(caught.exception))

    def test_transport_auth_and_incomplete_are_not_proposal_repair_errors(self):
        cases = [(None, 401), (None, 429),
                 (["data: {invalid-stream}"], 200),
                 (stream('{}', event='response.incomplete', status='incomplete'), 200),
                 (stream('{}', output=[{'type': 'function_call', 'name': 'move'}]), 200)]
        for events, status in cases:
            instance, client = self.make(events, status)
            with self.subTest(events=events, status=status), self.assertRaises(provider.ProviderError) as caught:
                instance.propose(agent_context())
            self.assertNotIsInstance(caught.exception, provider.ProposalError)
            self.assertEqual(client.post.call_count, 1)

    def test_freeze_requires_numerical_stop_and_null_plan(self):
        value = {'action': 'freeze', 'reason': '当前数值条件和前瞻反馈已满足。', 'plan': None}
        instance, _ = self.make(stream(json.dumps(value)))
        with self.assertRaises(provider.ProposalError) as caught:
            instance.propose(agent_context())
        self.assertEqual(caught.exception.code, 'freeze_not_eligible')
        ctx = agent_context(); ctx['stop']['eligible'] = True
        instance, _ = self.make(stream(json.dumps(value)))
        self.assertEqual(instance.propose(ctx)['action'], 'freeze')
        value['plan'] = agent_decision()['plan']
        instance, _ = self.make(stream(json.dumps(value)))
        with self.assertRaises(provider.ProposalError):
            instance.propose(ctx)

    def test_approved_B_plan_shape_and_pause(self):
        value = agent_decision()
        value['plan'].update(cond='B', model_id='fit_1', gain=.75)
        instance, _ = self.make(stream(json.dumps(value)))
        self.assertEqual(instance.propose(agent_context())['plan']['gain'], .75)
        paused = {'action': 'pause', 'reason': '当前证据不足，需要现场检查。', 'plan': None}
        instance, _ = self.make(stream(json.dumps(paused)))
        self.assertEqual(instance.propose(agent_context())['plan'], None)

    def test_explicit_mock_uses_new_shape_and_never_networks(self):
        with patch.object(provider, 'new_session') as network:
            instance = provider.AgentProvider({'backend': 'mock'})
            result = instance.propose(agent_context())
            self.assertEqual(result['action'], 'pause')
            self.assertIsNone(result['plan'])
            self.assertEqual(result['_meta']['backend'], 'mock')
            network.assert_not_called()


class AuthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import jwt
            from cryptography.hazmat.primitives.asymmetric import rsa
        except ImportError:
            raise unittest.SkipTest("Install PyJWT[crypto] to run offline OIDC tests")
        cls.jwt = jwt
        cls.key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        cls.jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(cls.key.public_key()))
        cls.jwk.update(kid="test-key", alg="RS256", use="sig")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.settings = {"auth_dir": str(Path(self.temp.name) / "auth"), "account": "test"}
        self.addCleanup(self.temp.cleanup)

    def signed(self, **overrides):
        claims = {"iss": auth.AUTH_BASE, "aud": "client_test", "sub": "subject_test",
                  "iat": int(time.time()) - 1, "exp": int(time.time()) + 600, "nonce": "expected-nonce"}
        claims.update(overrides)
        return self.jwt.encode(claims, self.key, algorithm="RS256", headers={"kid": "test-key"})

    def client(self):
        client = Mock()
        client.get.side_effect = lambda url, **kwargs: response(
            {"issuer": auth.AUTH_BASE, "jwks_uri": auth.AUTH_BASE + "/jwks",
             "revocation_endpoint": auth.AUTH_BASE + "/revoke"} if "openid-configuration" in url else
            {"keys": [self.jwk]})
        return client

    def record(self):
        return {"client_id": "client_test", "subject": "subject_test", "issuer": auth.AUTH_BASE,
                "email": "test@example.com", "ext_agent_host_id": "urn:uuid:test",
                "access_token": "old-access-secret", "refresh_token": "old-refresh-secret",
                "id_token": "old-id-secret", "scopes": auth.SCOPES.split(), "expires_at": time.time() - 1}

    def tokens(self):
        return {"access_token": "new-access-secret", "refresh_token": "new-refresh-secret",
                "token_type": "Bearer", "expires_in": 3600, "scope": auth.SCOPES,
                "id_token": self.signed()}

    def test_oidc_signature_nonce_audience_expiry_and_subject(self):
        identity = auth._validate_id_token(self.signed(), "client_test", self.client(), nonce="expected-nonce")
        self.assertEqual(identity["sub"], "subject_test")
        for claims in ({"nonce": "wrong"}, {"aud": "wrong"}, {"iss": "https://evil.example"},
                       {"exp": int(time.time()) - 1}, {"sub": "another"},
                       {"aud": ["client_test", "other"]}):
            with self.subTest(claims=claims), self.assertRaises(auth.AuthError):
                auth._validate_id_token(self.signed(**claims), "client_test", self.client(),
                                        nonce="expected-nonce", subject="subject_test")

    def test_algorithm_confusion_and_untrusted_discovery_are_rejected(self):
        token = self.jwt.encode({"sub": "test"}, "secret", algorithm="HS256", headers={"kid": "test-key"})
        with self.assertRaises(auth.AuthError):
            auth._validate_id_token(token, "client_test", self.client())
        client = Mock()
        client.get.return_value = response({"issuer": auth.AUTH_BASE, "jwks_uri": "https://example.com/jwks"})
        with self.assertRaises(auth.AuthError):
            auth._validate_id_token(self.signed(), "client_test", client)
        self.assertEqual(client.get.call_count, 1)

    def test_callback_state_duplicate_and_client_binding(self):
        good = {"code": ["code-secret"], "state": ["state"], "client_id": ["client_test"]}
        self.assertEqual(auth._parse_callback(good, "state", "client_test"), ("code-secret", "client_test"))
        for bad in (dict(good, state=["wrong"]), dict(good, state=["state", "state"]),
                    dict(good, client_id=["other"]), dict(good, error=["access_denied"])):
            with self.subTest(bad=bad), self.assertRaises(auth.AuthError):
                auth._parse_callback(bad, "state", "client_test")

    def test_refresh_replaces_rotating_tokens_atomically_and_redacts_status(self):
        auth._atomic_private(auth._record_path(self.settings), self.record())
        client = self.client()
        client.post.return_value = response(self.tokens())
        self.assertEqual(auth.access_token(self.settings, client), "new-access-secret")
        saved = auth._read_private(auth._record_path(self.settings))
        self.assertEqual(saved["refresh_token"], "new-refresh-secret")
        self.assertEqual(stat.S_IMODE(auth._record_path(self.settings).stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(auth.auth_directory(self.settings).stat().st_mode), 0o700)
        status = auth.auth_status(self.settings)
        self.assertTrue(status["signed_in"])
        self.assertNotIn("secret", json.dumps(status))
        self.assertNotIn("scope", client.post.call_args.kwargs["data"])
        self.assertFalse(client.post.call_args.kwargs["allow_redirects"])

    def test_fresh_token_does_not_refresh_and_earliest_refresh_is_honored(self):
        record = self.record()
        record["expires_at"] = time.time() + 1000
        auth._atomic_private(auth._record_path(self.settings), record)
        client = self.client()
        self.assertEqual(auth.access_token(self.settings, client), "old-access-secret")
        client.post.assert_not_called()
        record.update(expires_at=time.time() + 50, earliest_refresh_at=time.time() + 100)
        auth._atomic_private(auth._record_path(self.settings), record)
        self.assertEqual(auth.access_token(self.settings, client), "old-access-secret")
        with self.assertRaises(auth.AuthError):
            auth.access_token(self.settings, client, force_refresh=True)

    def test_failed_refresh_retains_previous_credentials_and_does_not_log_response(self):
        auth._atomic_private(auth._record_path(self.settings), self.record())
        client = self.client()
        client.post.return_value = response({"error": "secret-body"}, status=400)
        with self.assertRaises(auth.AuthError) as error:
            auth.access_token(self.settings, client)
        self.assertNotIn("secret", str(error.exception))
        self.assertEqual(auth._read_private(auth._record_path(self.settings))["refresh_token"], "old-refresh-secret")

    def test_refresh_wrong_account_or_missing_grant_does_not_replace(self):
        for overrides in ({"id_token": self.signed(sub="wrong")}, {"scope": "openid"},
                          {"refresh_token": ""}):
            auth._atomic_private(auth._record_path(self.settings), self.record())
            client = self.client()
            client.post.return_value = response(dict(self.tokens(), **overrides))
            with self.subTest(overrides=overrides), self.assertRaises(auth.AuthError):
                auth.access_token(self.settings, client)
            self.assertEqual(auth._read_private(auth._record_path(self.settings))["access_token"], "old-access-secret")

    def test_cache_outside_repo_and_private_permissions_enforced(self):
        with self.assertRaises(auth.AuthError):
            auth.auth_directory({"auth_dir": str(auth.REPO_ROOT / "outputs" / "tokens")})
        auth._atomic_private(auth._record_path(self.settings), self.record())
        auth._record_path(self.settings).chmod(0o644)
        with self.assertRaises(auth.AuthError):
            auth.auth_status(self.settings)

    def test_logout_revokes_and_keeps_registration_but_removes_every_token(self):
        auth._atomic_private(auth._record_path(self.settings), self.record())
        client = self.client()
        client.post.return_value = response(status=200)
        with patch.object(auth, "new_session", return_value=client):
            result = auth.logout(self.settings)
        self.assertEqual(result["remote_revocation"], "confirmed")
        record = auth._read_private(auth._record_path(self.settings))
        self.assertEqual(record["client_id"], "client_test")
        self.assertFalse(any("token" in key for key in record))
        self.assertEqual(client.post.call_args.kwargs["data"]["token_type_hint"], "refresh_token")

    def test_logout_failure_still_clears_tokens_and_reports_unconfirmed(self):
        auth._atomic_private(auth._record_path(self.settings), self.record())
        client = self.client()
        client.get.side_effect = RuntimeError("secret-network-detail")
        with patch.object(auth, "new_session", return_value=client), patch.object(auth.time, "sleep"):
            result = auth.logout(self.settings)
        self.assertEqual(result["remote_revocation"], "unconfirmed")
        self.assertNotIn("secret", json.dumps(result))
        self.assertFalse(auth.auth_status(self.settings)["signed_in"])

    def test_logout_retries_transient_network_before_clearing_credentials(self):
        auth._atomic_private(auth._record_path(self.settings), self.record())
        client = self.client()
        client.post.side_effect = [RuntimeError("network failure"), response(status=200)]
        with patch.object(auth, "new_session", return_value=client), patch.object(auth.time, "sleep"):
            result = auth.logout(self.settings)
        self.assertEqual(result["remote_revocation"], "confirmed")
        self.assertEqual(client.post.call_count, 2)

    def test_legacy_registration_reused_and_account_slots_remain_separate(self):
        legacy = Path(self.temp.name) / "legacy"
        legacy.mkdir()
        (legacy / "client_id").write_text("client_probe")
        (legacy / "host_id").write_text("urn:uuid:probe")
        settings = dict(self.settings, account="default", identity_dir=str(legacy))
        self.assertEqual(auth._registration(settings)["client_id"], "client_probe")
        self.assertNotIn("client_id", auth._registration(dict(settings, account="different")))

    def test_complete_login_reuses_registration_and_validates_new_identity(self):
        settings = dict(self.settings, callback_port=8766, open_browser=False)
        initial = {key: value for key, value in self.record().items() if "token" not in key}
        auth._atomic_private(auth._record_path(settings), initial)
        printed = []
        client = self.client()

        class Server:
            server_port = 8766
            def handle_request(inner):
                query = urllib.parse.parse_qs(urllib.parse.urlsplit(printed[-1]).query)
                self.assertEqual(query["client_id"], ["client_test"])
                self.assertNotIn("id_token_hint", query)
                self.assertEqual(query["code_challenge_method"], ["S256"])
                self.assertEqual(query["redirect_uri"], ["http://127.0.0.1:8766/auth/callback"])
                inner.result = ("oauth-code", "client_test")
                client.post.return_value = response(dict(self.tokens(), id_token=self.signed(nonce=query["nonce"][0])))
            def server_close(inner):
                pass

        with patch.object(auth.http.server, "HTTPServer", return_value=Server()), \
             patch.object(auth, "new_session", return_value=client), \
             patch.object(auth.webbrowser, "open") as browser:
            result = auth.login(settings, say=printed.append)
        browser.assert_not_called()
        self.assertTrue(result["signed_in"])
        self.assertNotIn("secret", "\n".join(printed))
        token_data = client.post.call_args.kwargs["data"]
        self.assertEqual(token_data["redirect_uri"], "http://127.0.0.1:8766/auth/callback")
        self.assertGreater(len(token_data["code_verifier"]), 43)


if __name__ == "__main__":
    unittest.main()
