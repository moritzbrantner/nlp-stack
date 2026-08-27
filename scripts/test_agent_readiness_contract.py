from pathlib import Path
import unittest

from check_agent_readiness_contract import check_contract


class AgentReadinessContractTests(unittest.TestCase):
    def test_repository_agent_readiness_contract(self) -> None:
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(check_contract(root), [])


if __name__ == "__main__":
    unittest.main()
