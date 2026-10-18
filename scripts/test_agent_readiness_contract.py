from pathlib import Path
import unittest
from unittest.mock import patch

import check_agent_readiness_contract as contract

from check_agent_readiness_contract import check_contract


class AgentReadinessContractTests(unittest.TestCase):
    def test_retired_refs_do_not_satisfy_current_readiness_policy(self) -> None:
        root = Path(__file__).resolve().parents[1]
        original_load = contract._load_json

        def load(path: Path) -> dict:
            data = original_load(path)
            if path.name == ".coding-tooling.json":
                data["conventionRefs"] = [
                    ref for ref in data["conventionRefs"] if ref != "PRINCIPLE-003"
                ] + ["AGENT-007", "AGENT-008", "TEST-002"]
            return data

        with patch.object(contract, "_load_json", side_effect=load):
            self.assertIn(
                "missing required convention refs: PRINCIPLE-003", check_contract(root)
            )

    def test_repository_agent_readiness_contract(self) -> None:
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(check_contract(root), [])


if __name__ == "__main__":
    unittest.main()
