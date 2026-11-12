import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from bottle_scan.skill_agent_bridge import skill_planning_reference

class BridgeTests(unittest.TestCase):
    def test_independent_report_never_reads_sources(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'report.json'; p.write_text(json.dumps({'role':'independent'}))
            with patch('bottle_scan.skill_agent_bridge.evaluate_skill_sessions') as evaluate:
                with self.assertRaises(ValueError): skill_planning_reference(p)
                evaluate.assert_not_called()

    def test_changed_report_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'report.json'
            p.write_text(json.dumps({'role':'development','source_sessions':[{'path':'source'}]}))
            with patch('bottle_scan.skill_agent_bridge.evaluate_skill_sessions',return_value={'role':'development'}):
                with self.assertRaises(ValueError): skill_planning_reference(p)

    def test_current_three_output_evidence_is_visible(self):
        report={'role':'development','source_sessions':[{'path':'source'}],
                'invocation':{},'recovery':{},'exit':{},'applicability':{},
                'coverage':{},'evidence_refs':['source/trials/1/meta.json']}
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'report.json'; p.write_text(json.dumps(report))
            with patch('bottle_scan.skill_agent_bridge.evaluate_skill_sessions',return_value=report):
                result=skill_planning_reference(p)
                self.assertTrue(result['findings'])
                self.assertEqual(result['evidence_refs'],report['evidence_refs'])
