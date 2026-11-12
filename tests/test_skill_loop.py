"""Campaign state/immutability tests. Every physical collector and model is fake."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import yaml

from bottle_scan.agentic_sim import default_T0
from bottle_scan.session import atomic_json
from bottle_scan import skill_loop as loop
from bottle_scan.skill_loop_feedback import assess_acceptance
from tests.test_agentic_executor import cfg
from tests.test_skill_loop_feedback import exploration_config, learned_model, physical_report


class CampaignFixture:
    def __init__(self, parent, *, confirmed=False, max_rounds=3, max_attempts=200, planner="agent_evidence"):
        self.parent = Path(parent)
        self.root = self.parent / "campaign"
        base = cfg()
        (self.parent / "config.ros.yaml").write_text("schema: offline_test_only\n")
        (self.parent / "base.yaml").write_text(yaml.safe_dump(base))
        self.target = {"id": "white-target-v1", "description": "Observed contact position",
            "normal": "Stable reference target", "needs_recovery": "Observed off-target contact",
            "needs": "Observed off-target contact", "completion": "Stable reference target reached"}
        design = {"offsets_mm": [-5., 0., 5.], "reps": 4, "seed": 12}
        spec = {"schema": "bottle_skill_config_v1", "base_config": "base.yaml",
            "runtime": {"action_axis": "a"}, "context": {}, "target_definition": self.target,
            "collection": design}
        (self.parent / "skill.yaml").write_text(yaml.safe_dump(spec))
        (self.parent / "exploration.yaml").write_text(yaml.safe_dump(exploration_config()))
        self.reference = self.parent / "reference"
        plan = [{"id": "00001", "cond": "A", "delta_mm": 0.}]
        atomic_json(self.reference / "session_meta.json", {
            "schema": "e1e2_session_v1", "config": base, "T0": default_T0().tolist(),
            "simulated": False, "cond": "A", "agentic": {"role": "exploration"}, "plan": plan})
        atomic_json(self.reference / "trials/00001/meta.json", {"id": "00001", "complete": True})
        self.config = {"schema": loop.SCHEMA, "target_definition": self.target,
            "skill_config": "skill.yaml", "exploration_config": "exploration.yaml",
            "reference_session": "reference", "initial_A": design, "development_B": design,
            "independent_B": design, "action_axis": "a", "seed": 31,
            "physical_setup_confirmed": confirmed, "acceptance": {}, "learning": {},
            "planner": {"mode": planner},
            "budget": {"max_rounds": max_rounds, "max_new_attempts": max_attempts, "max_batch_attempts": 32}}
        self.config_path = self.parent / "loop.yaml"
        self.config_path.write_text(yaml.safe_dump(self.config))
        self.train_calls = []
        self.collect_calls = []
        self.evaluation_calls = []

    def init(self, *, seed=False):
        return loop.init_campaign(self.config_path, self.root,
                                  seed_sessions=[self.reference] if seed else [])

    def trainer(self, sources, settings):
        self.train_calls.append(list(sources))
        model = learned_model()
        model["model_id"] = "frozen-" + str(len(self.train_calls))
        model["reference_T0"] = default_T0().tolist()
        return model

    def write_batch(self, *, incomplete=False):
        state = loop.read(self.root / "state.json")
        config = loop.read(self.root / "config.json")
        batch = state["batch"]
        kind = batch["kind"]
        meta = deepcopy(loop.read(self.reference / "session_meta.json"))
        meta.update(cond="A" if kind == "A" else "B", plan=batch["plan"],
                    agentic={"role": {"A": "exploration", "B": "development", "independent": "final_validation"}[kind]},
                    config=deepcopy(config["base_config"]))
        meta["config"]["recovery_skill"] = {"target_definition": self.target}
        if kind != "A":
            meta["fit"] = loop.read(self.root / state["model"]["path"])
        session = Path(batch["session"])
        atomic_json(session / "session_meta.json", meta)
        for trial in batch["plan"]:
            atomic_json(session / "trials" / trial["id"] / "meta.json",
                        {**trial, "complete": not incomplete})
        return session

    def collector(self, arguments):
        self.collect_calls.append(arguments)
        self.write_batch()
        return 0

    def evaluator(self, session, *, role, expected_model_id, fail=False):
        self.evaluation_calls.append((str(session), role, expected_model_id))
        report = physical_report()
        plan = loop.read(Path(session) / "session_meta.json")["plan"]
        while len(report["details"]) < len(plan):
            report["details"].append(deepcopy(report["details"][0]))
        report["details"] = report["details"][:len(plan)]
        old_paths = [row["trial_path"] for row in report["details"][:12]]
        for row, trial in zip(report["details"], plan):
            row.update(trial_path=str(Path(session) / "trials" / trial["id"]), model_id=expected_model_id)
        path_map = dict(zip(old_paths, [d["trial_path"] for d in report["details"]]))
        for action in report["actions"]:
            action["trial_path"] = path_map[action["trial_path"]]
        for cell in report["cells"]:
            cell["evidence_refs"] = [path_map[r.rsplit("/", 1)[0]] + "/meta.json" for r in cell["evidence_refs"]]
        n = len(plan)
        report.update(role=role, model_ids=[expected_model_id],
                      evidence_refs=[d["trial_path"] + "/meta.json" for d in report["details"]])
        report["coverage"].update(trials=n, recording_verified=n)
        report["completion_validation"].update(TP=n, target_count=n)
        if fail:
            report["actions"] = []
        return report

    def finish_current(self, *, evaluator=None, trainer=None):
        self.write_batch()
        return loop.advance(self.root, trainer=trainer or self.trainer, evaluator=evaluator or self.evaluator)


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def fixture(self, **kw):
        return CampaignFixture(self.temp.name, **kw)

    def test_initialization_is_offline_and_freezes_reference_plan_and_config(self):
        fixture = self.fixture()
        with patch("bottle_scan.agentic_executor.execute_batch", side_effect=AssertionError("hardware path")):
            state = fixture.init()
            self.assertEqual(state["phase"], "collect_A")
            result = loop.advance(fixture.root, trainer=fixture.trainer)
        self.assertEqual(result["phase"], "collect_A")
        self.assertEqual(fixture.train_calls, [])
        frozen = loop.read(fixture.root / "config.json")
        self.assertEqual(frozen["reference_protocol"]["T0"], default_T0().tolist())
        self.assertEqual(frozen["base_config"]["gripper"]["min_normalized"], .336)
        self.assertTrue(any(r["delta_mm"] == 0 for r in state["batch"]["plan"]))
        self.assertEqual(loop.digest(frozen), state["config_hash"])

    def test_explicit_closure_floor_override_preserves_reference_and_collects_new_protocol(self):
        from bottle_scan.skill_collection import main as collect_main
        fixture = self.fixture()
        reference_before = loop.source_snapshot(fixture.reference)
        fixture.config['execution_overrides'] = {'gripper': {'min_normalized': .330}}
        fixture.config_path.write_text(yaml.safe_dump(fixture.config))
        state = fixture.init()
        frozen = loop.read(fixture.root/'config.json')
        self.assertEqual(frozen['reference_protocol']['gripper']['min_normalized'], .336)
        self.assertEqual(frozen['execution_protocol']['gripper']['min_normalized'], .330)
        self.assertEqual(frozen['base_config']['gripper']['min_normalized'], .330)
        self.assertEqual(frozen['reference_protocol']['T0'], frozen['execution_protocol']['T0'])
        self.assertEqual(loop.source_snapshot(fixture.reference), reference_before)
        base_path = Path(state['batch']['spec']).parent/'base.yaml'
        self.assertEqual(yaml.safe_load(base_path.read_text())['gripper']['min_normalized'], .330)
        # Exercise the real collection argument/config/reference binding, then
        # stop at its injected fake executor: no hardware factory is entered.
        with patch('bottle_scan.skill_collection.execute_batch', return_value={'offline': True}) as execute:
            result = collect_main(['collect','--real','--config',state['batch']['spec'],
                '--cond','A','--role','exploration','--mode','development',
                '--reference-session',str(fixture.reference),'--min-normalized','0.330',
                '--out',state['batch']['session']])
        self.assertEqual(result, 0)
        self.assertEqual(execute.call_args.args[2]['gripper']['min_normalized'], .330)
        self.assertEqual(execute.call_args.args[4]['T0'], frozen['reference_protocol']['T0'])
        # A record with the new protocol must be consumable, despite its old
        # geometry reference retaining the historical floor.
        session = fixture.write_batch()
        root, state, config = loop._load(fixture.root)
        self.assertTrue(loop._consume(root, state, config))
        self.assertEqual(state['training_sources'][0]['path'], str(session))
        self.assertEqual(loop.source_snapshot(fixture.reference), reference_before)

    def test_closure_override_rejects_historical_seed_and_wrong_collected_floor(self):
        fixture = self.fixture()
        fixture.config['execution_overrides'] = {'gripper': {'min_normalized': .330}}
        fixture.config_path.write_text(yaml.safe_dump(fixture.config))
        with self.assertRaisesRegex(ValueError, 'seed protocol'):
            fixture.init(seed=True)
        self.assertFalse(fixture.root.exists())
        fixture.init()
        session = fixture.write_batch()
        path = session/'session_meta.json'
        meta = loop.read(path); meta['config']['gripper']['min_normalized'] = .336
        atomic_json(path, meta)
        root, state, config = loop._load(fixture.root)
        with self.assertRaisesRegex(ValueError, 'collected protocol'):
            loop._consume(root, state, config)
        self.assertEqual(state['training_sources'], [])

    def test_closure_override_accepts_only_explicit_finite_in_bounds_floor(self):
        invalid = [None, [], {'force': {'target': 7}}, {'gripper': {}},
                   {'gripper': {'min_normalized': .330, 'open_normalized': 1.}},
                   {'gripper': {'min_normalized': .330}, 'motion': {}},
                   *[{'gripper': {'min_normalized': value}} for value in
                     (True, '0.330', None, float('nan'), float('inf'), -0.001, 1., 1.1)]]
        for value in invalid:
            with self.subTest(value=str(value)), tempfile.TemporaryDirectory() as td:
                fixture = CampaignFixture(td)
                fixture.config['execution_overrides'] = value
                fixture.config_path.write_text(yaml.safe_dump(fixture.config))
                with self.assertRaisesRegex(ValueError, 'execution_overrides'):
                    fixture.init()
                self.assertFalse(fixture.root.exists())
        with tempfile.TemporaryDirectory() as td:
            fixture = CampaignFixture(td)
            path = fixture.reference/'session_meta.json'; meta = loop.read(path)
            meta['config']['gripper']['open_normalized'] = .5; atomic_json(path, meta)
            fixture.config['execution_overrides'] = {'gripper': {'min_normalized': .6}}
            fixture.config_path.write_text(yaml.safe_dump(fixture.config))
            with self.assertRaisesRegex(ValueError, 'below the reference open_normalized'):
                fixture.init()
            self.assertFalse(fixture.root.exists())

    def test_full_A_B_feedback_A_B_independent_chain_with_joint_versions(self):
        fixture = self.fixture(); fixture.init()
        state = fixture.finish_current()
        self.assertEqual(state["phase"], "collect_B")
        def first_fails(session, **kw):
            return fixture.evaluator(session, fail=True, **kw)
        state = fixture.finish_current(evaluator=first_fails)
        self.assertEqual(state["phase"], "collect_A")
        self.assertEqual(state["round"], 2)
        self.assertIsNotNone(state["batch"]["selection"])
        state = fixture.finish_current()
        self.assertEqual(state["phase"], "collect_B")
        self.assertEqual(state["model"]["model_id"], "frozen-2")
        state = fixture.finish_current()
        self.assertEqual(state["phase"], "collect_independent")
        state = fixture.finish_current()
        self.assertEqual(state["phase"], "complete")
        self.assertTrue(state["rounds"][-1]["independent"]["passed"])
        self.assertEqual(len(fixture.train_calls), 2)
        self.assertEqual(len(state["sealed_evaluation_sources"]), 1)
        self.assertFalse(any("independent" in p for call in fixture.train_calls for p in call))
        self.assertEqual([x[1] for x in fixture.evaluation_calls], ["development", "development", "independent"])
        self.assertEqual(loop.advance(fixture.root, trainer=fixture.trainer)["phase"], "complete")
        self.assertEqual(len(fixture.train_calls), 2)

    def test_blocked_model_skips_B_and_revises_A_without_claiming_success(self):
        fixture = self.fixture(); fixture.init()
        def blocked(sources, settings):
            model = fixture.trainer(sources, settings)
            model.update(usable_for_B=False, blocking_reasons=["No negative completion examples"],
                         data_gaps=[{"component": "exit", "class": "negative", "missing": 6}])
            return model
        state = fixture.finish_current(trainer=blocked)
        self.assertEqual(state["phase"], "collect_A")
        self.assertEqual(state["round"], 2)
        self.assertFalse(state["rounds"][0]["usable_for_B"])
        self.assertEqual(fixture.evaluation_calls, [])

    def test_budget_stops_without_success_even_with_ready_model(self):
        fixture = self.fixture(max_attempts=13); state = fixture.init()
        self.assertEqual(len(state["batch"]["plan"]), 13)
        state = fixture.finish_current()
        self.assertEqual(state["phase"], "budget_exhausted")
        self.assertEqual(state["new_attempts"], 13)
        self.assertNotIn("independent", state["rounds"][0])

    def test_round_budget_and_missing_labels_do_not_pass(self):
        fixture = self.fixture(max_rounds=1); fixture.init(); fixture.finish_current()
        def unlabelled(session, **kw):
            report = fixture.evaluator(session, **kw)
            report["details"][0]["initial_class"] = None
            return report
        state = fixture.finish_current(evaluator=unlabelled)
        self.assertEqual(state["phase"], "budget_exhausted")
        self.assertFalse(state["rounds"][0]["development"]["passed"])

    def test_failed_independent_evidence_is_sealed_and_never_guides_update(self):
        fixture = self.fixture(); fixture.init(); fixture.finish_current(); fixture.finish_current()
        state = fixture.finish_current(evaluator=lambda session, **kw: fixture.evaluator(session, fail=True, **kw))
        self.assertEqual(state["phase"], "independent_failed")
        self.assertEqual(len(fixture.train_calls), 1)
        self.assertEqual(loop.advance(fixture.root, trainer=fixture.trainer)["phase"], "independent_failed")
        self.assertEqual(len(fixture.train_calls), 1)
        self.assertTrue(state["sealed_evaluation_sources"])
        self.assertTrue(all("independent" not in x["path"] for x in state["training_sources"]))

    def test_run_requires_explicit_real_and_confirmed_setup_before_collector(self):
        fixture = self.fixture(); fixture.init(); collector = Mock(side_effect=AssertionError("must not collect"))
        with self.assertRaisesRegex(ValueError, "--real"):
            loop.run(fixture.root, collector=collector)
        with self.assertRaisesRegex(ValueError, "confirm"):
            loop.run(fixture.root, real=True, collector=collector)
        collector.assert_not_called()

    def test_run_uses_original_collector_contract_with_fake_collector_only(self):
        fixture = self.fixture(confirmed=True, max_attempts=13); fixture.init()
        state = loop.run(fixture.root, real=True, collector=fixture.collector,
                         trainer=fixture.trainer, evaluator=fixture.evaluator)
        self.assertEqual(state["phase"], "budget_exhausted")
        self.assertEqual(len(fixture.collect_calls), 1)
        arguments = fixture.collect_calls[0]
        self.assertIn("--step", arguments)
        self.assertEqual(arguments[arguments.index("--min-normalized")+1], "0.336")
        self.assertEqual(arguments[arguments.index("--reference-session")+1], str(fixture.reference))

    def test_llm_planning_without_allow_agent_stays_offline_and_pending(self):
        fixture = self.fixture(planner="llm"); fixture.init()
        def blocked(sources, settings):
            model = fixture.trainer(sources, settings)
            model.update(usable_for_B=False, blocking_reasons=["needs labels"])
            return model
        with patch("bottle_scan.agentic_provider.AgentProvider", side_effect=AssertionError("provider must remain offline")):
            state = fixture.finish_current(trainer=blocked)
        self.assertEqual(state["phase"], "awaiting_agent")
        self.assertEqual(state["round"], 1)

    def test_injected_agent_can_propose_only_bounded_reference_containing_A(self):
        fixture = self.fixture(planner="llm"); fixture.init(); fixture.write_batch()
        def blocked(sources, settings):
            model = fixture.trainer(sources, settings)
            model.update(usable_for_B=False, blocking_reasons=["needs contact labels"])
            return model
        def answer(context):
            self.assertFalse(context["stop"]["eligible"])
            return {"action": "propose_batch", "reason": "Resolve measured class gaps",
                    "plan": {"cond": "A", "gain": 1, "model_id": None,
                             "evidence_refs": context["valid_evidence_refs"],
                             "cells": [{"offset_mm": 0., "reps": 2}, {"offset_mm": 5., "reps": 2}],
                             "hypothesis": "Contact states differ at the selected offset",
                             "expected_outcome": "Independently labelled target and non-target states"}}
        provider = Mock(); provider.propose.side_effect = answer
        state = loop.advance(fixture.root, trainer=blocked, allow_agent=True, provider=provider)
        self.assertEqual(state["phase"], "collect_A")
        self.assertEqual(state["round"], 2)
        selected = loop.read(state["batch"]["selection"])
        self.assertEqual(selected["selector"], "llm")
        self.assertEqual(selected["planned_trial_attempts"], 4)
        self.assertTrue(all(x["condition"]["contact_height_l_mm"] == 0 and
                            x["condition"]["probe_force"] == 1.5 for x in selected["selected"]))
        provider.close.assert_not_called()

    def test_blocked_training_agent_receives_A_counts_not_invented_B_findings(self):
        fixture = self.fixture(planner="llm"); fixture.init(); fixture.write_batch()
        def blocked(sources, settings):
            model = fixture.trainer(sources, settings)
            model.update(usable_for_B=False,
                         blocking_reasons=["contradictory_contact_labels: trial-7"],
                         data_gaps=[{"component": "exit", "class": "negative", "missing": 4}])
            model["invocation_head"]["class_counts"] = {"positive": 5, "negative": 17}
            model["exit_head"]["class_counts"] = {"positive": 21, "negative": 0}
            return model
        def answer(context):
            diagnosis = context["diagnosis"]
            self.assertEqual(diagnosis["B_evaluation"], "not_evaluated_for_current_model")
            self.assertEqual(diagnosis["learned_class_counts"]["invocation"]["positive"], 5)
            self.assertEqual(diagnosis["learned_class_counts"]["invocation"]["negative"], 17)
            codes = {x["code"] for x in diagnosis["findings"]}
            self.assertIn("contradictory_contact_labels", codes)
            self.assertNotIn("no_needs_recovery_examples", codes)
            self.assertNotIn("no_normal_examples", codes)
            return {"action": "pause", "reason": "Review inconsistent source labels", "plan": None}
        provider = Mock(); provider.propose.side_effect = answer
        state = loop.advance(fixture.root, trainer=blocked, allow_agent=True, provider=provider)
        self.assertEqual(state["phase"], "awaiting_agent")
        self.assertEqual(provider.propose.call_count, 1)

    def test_injected_agent_cannot_freeze_or_expand_motion_or_omit_reference(self):
        for invalid in ("freeze", "motion", "reference", "evidence", "model", "budget",
                        "gain", "B", "force", "plan_force", "height", "reference_gap", "repeated_nonzero"):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as td:
                fixture = CampaignFixture(td, planner="llm"); fixture.init(); fixture.write_batch()
                def blocked(sources, settings):
                    model = fixture.trainer(sources, settings)
                    model.update(usable_for_B=False, blocking_reasons=["needs labels"])
                    return model
                def answer(context):
                    value = {"action": "propose_batch", "reason": "Synthetic negative test",
                        "plan": {"cond": "A", "gain": 1, "model_id": None,
                                 "evidence_refs": context["valid_evidence_refs"],
                                 "cells": [{"offset_mm": 0., "reps": 2}, {"offset_mm": 5., "reps": 2}],
                                 "hypothesis": "Test", "expected_outcome": "Test"}}
                    if invalid == "freeze": value["action"] = "freeze"
                    elif invalid == "motion": value["plan"]["cells"][1]["offset_mm"] = 50.
                    elif invalid == "reference": value["plan"]["cells"].pop(0)
                    elif invalid == "evidence": value["plan"]["evidence_refs"] = ["invented/source"]
                    elif invalid == "model": value["plan"]["model_id"] = "other-model"
                    elif invalid == "budget": value["plan"]["cells"][0]["reps"] = 100
                    elif invalid == "gain": value["plan"]["gain"] = 2
                    elif invalid == "B": value["plan"]["cond"] = "B"
                    elif invalid == "force": value["plan"]["cells"][0]["probe_force"] = 3.
                    elif invalid == "plan_force": value["plan"]["probe_force"] = 3.
                    elif invalid == "height": value["plan"]["cells"][0]["contact_height_l_mm"] = 4.
                    elif invalid == "reference_gap":
                        value["plan"]["cells"] = [{"offset_mm": 0., "reps": 2},
                            {"offset_mm": -5., "reps": 6}, {"offset_mm": 5., "reps": 6}]
                    elif invalid == "repeated_nonzero":
                        value["plan"]["cells"].append({"offset_mm": 5., "reps": 2})
                    return value
                provider = Mock(); provider.propose.side_effect = answer
                with self.assertRaises(ValueError) as caught:
                    loop.advance(fixture.root, trainer=blocked, allow_agent=True, provider=provider)
                if invalid == "reference_gap":
                    self.assertIn("reference gap", str(caught.exception))
                state = loop.read(fixture.root / "state.json")
                self.assertEqual(state["round"], 1)
                self.assertFalse((fixture.root / "round_002").exists())

    def test_agent_may_insert_repeated_budgeted_zero_references(self):
        fixture = self.fixture(planner="llm"); fixture.init(); fixture.write_batch()
        def blocked(sources, settings):
            model = fixture.trainer(sources, settings)
            model.update(usable_for_B=False, blocking_reasons=["needs labels"])
            return model
        def answer(context):
            return {"action": "propose_batch", "reason": "Preserve physical reference interval",
                "plan": {"cond": "A", "gain": 1, "model_id": None,
                         "evidence_refs": context["valid_evidence_refs"],
                         "cells": [{"offset_mm": 0., "reps": 2}, {"offset_mm": -5., "reps": 6},
                                   {"offset_mm": 0., "reps": 2}, {"offset_mm": 5., "reps": 6}],
                         "hypothesis": "Reference drift can be separated", "expected_outcome": "Controlled labels"}}
        provider = Mock(); provider.propose.side_effect = answer
        state = loop.advance(fixture.root, trainer=blocked, allow_agent=True, provider=provider)
        self.assertEqual(state["phase"], "collect_A")
        self.assertEqual(len(state["batch"]["plan"]), 16)
        self.assertEqual(sum(r["delta_mm"] == 0 for r in state["batch"]["plan"]), 4)

    def test_frozen_config_model_reference_and_training_source_changes_block_resume(self):
        for kind in ("config", "model", "reference", "source", "hardware", "raw_tcp"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as td:
                fixture = CampaignFixture(td); fixture.init(); state = fixture.finish_current()
                if kind == "config":
                    path = fixture.root / "config.json"
                elif kind == "model":
                    path = fixture.root / state["model"]["path"]
                elif kind == "reference":
                    path = fixture.reference / "trials/00001/meta.json"
                elif kind == "source":
                    path = Path(state["training_sources"][0]["path"]) / "trials/00001/meta.json"
                elif kind == "raw_tcp":
                    import h5py
                    path = Path(state["training_sources"][0]["path"]) / "trials/00001/streams.h5"
                    with h5py.File(path, "w") as file:
                        group = file.create_group("streams/tcp_measured")
                        group.create_dataset("monotonic_ns", data=[1])
                        group.create_dataset("data", data=[default_T0()])
                else:
                    path = fixture.parent / "config.ros.yaml"
                    path.write_text("schema: modified_test_config\n")
                if kind not in ("hardware", "raw_tcp"):
                    value = loop.read(path); value["tampered"] = True; atomic_json(path, value)
                with self.assertRaises(ValueError): loop.advance(fixture.root, trainer=fixture.trainer)

    def test_sealed_reports_and_independent_sources_cannot_be_rewritten(self):
        for kind in ("report", "source"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as td:
                fixture = CampaignFixture(td); fixture.init()
                fixture.finish_current(); fixture.finish_current(); state = fixture.finish_current()
                path = (Path(state["sealed_reports"][-1]["path"]) if kind == "report" else
                        Path(state["sealed_evaluation_sources"][0]["path"]) / "trials/00001/meta.json")
                value = loop.read(path); value["tampered"] = True; atomic_json(path, value)
                with self.assertRaises(ValueError): loop.advance(fixture.root, trainer=fixture.trainer)

    def test_independent_seed_and_duplicate_seed_are_rejected(self):
        fixture = self.fixture()
        meta = loop.read(fixture.reference / "session_meta.json")
        meta["agentic"]["role"] = "final_validation"
        atomic_json(fixture.reference / "session_meta.json", meta)
        with self.assertRaisesRegex(ValueError, "seed"):
            fixture.init(seed=True)
        meta["agentic"]["role"] = "exploration"; atomic_json(fixture.reference / "session_meta.json", meta)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            loop.init_campaign(fixture.config_path, fixture.root, seed_sessions=[fixture.reference, fixture.reference])

    def labelled_B_fixture(self):
        fixture = self.fixture(); fixture.init(); fixture.finish_current()
        session = fixture.write_batch()
        report = fixture.evaluator(session, role="development", expected_model_id="frozen-1")
        for detail in report["details"]:
            path = Path(detail["trial_path"]) / "meta.json"
            row = loop.read(path)
            row["skill_ground_truth"] = {"target_definition_id": fixture.target["id"]}
            truths = [False, True] if detail["initial_class"] == "needs_recovery" else [True]
            row["skill_observations"] = []
            row["skill_window_labels"] = []
            for index, truth in enumerate(truths):
                timestamp = 100 + index
                row["skill_observations"].append({"observation": {"timestamp_ns": timestamp},
                    "skill_outputs": {"model_id": "frozen-1", "learned_conditions": {
                        "exit": {"positive": truth, "valid": True}}}})
                row["skill_window_labels"].append({"timestamp_ns": timestamp, "target_reached": truth,
                    "source": "operator_review_of_observed_contact", "target_definition_id": fixture.target["id"],
                    "independent_of_experimental_offset": True})
            atomic_json(path, row)
        return fixture, session, report

    def test_evaluator_keeps_missing_predictions_unknown_and_labels_in_denominators(self):
        fixture, session, base_report = self.labelled_B_fixture()
        path = Path(base_report["details"][0]["trial_path"]) / "meta.json"
        row = loop.read(path); row["skill_observations"][0].pop("skill_outputs")
        atomic_json(path, row)
        with patch("bottle_scan.skill_offline.evaluate_skill_sessions", return_value=deepcopy(base_report)):
            report = loop.evaluate_batch(session, role="development", expected_model_id="frozen-1")
        self.assertEqual(report["completion_validation"]["unknown"], 1)
        self.assertEqual(report["completion_validation"]["target_count"], len(base_report["details"]))
        self.assertTrue(report["completion_validation"]["label_source_verified"])
        result = assess_acceptance(report, expected_model_id="frozen-1", model=learned_model())
        self.assertFalse(result["passed"])

    def test_evaluator_requires_all_decision_windows_labelled_and_target_id_consistent(self):
        fixture, session, base_report = self.labelled_B_fixture()
        with patch("bottle_scan.skill_offline.evaluate_skill_sessions", return_value=deepcopy(base_report)):
            good = loop.evaluate_batch(session, role="development", expected_model_id="frozen-1")
        self.assertTrue(assess_acceptance(good, expected_model_id="frozen-1", model=learned_model())["passed"])
        path = Path(base_report["details"][0]["trial_path"]) / "meta.json"
        row = loop.read(path)
        row["skill_window_labels"] = []
        row["skill_ground_truth"]["target_definition_id"] = "different-target"
        atomic_json(path, row)
        with patch("bottle_scan.skill_offline.evaluate_skill_sessions", return_value=deepcopy(base_report)):
            bad = loop.evaluate_batch(session, role="development", expected_model_id="frozen-1")
        self.assertFalse(bad["completion_validation"]["label_source_verified"])
        self.assertIsNone(bad["details"][0]["initial_class"])
        self.assertFalse(assess_acceptance(bad, expected_model_id="frozen-1", model=learned_model())["passed"])

    def test_evaluator_rejects_conflicting_duplicate_completion_labels(self):
        fixture, session, base_report = self.labelled_B_fixture()
        path = Path(base_report["details"][0]["trial_path"]) / "meta.json"
        row = loop.read(path)
        row["skill_window_labels"].append({**row["skill_window_labels"][0], "target_reached": False})
        atomic_json(path, row)
        with patch("bottle_scan.skill_offline.evaluate_skill_sessions", return_value=deepcopy(base_report)):
            with self.assertRaisesRegex(ValueError, "conflicting duplicate"):
                loop.evaluate_batch(session, role="development", expected_model_id="frozen-1")


if __name__ == "__main__":
    unittest.main()
