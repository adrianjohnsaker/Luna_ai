import json
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
import re

@dataclass
class Violation:
    type: str
    detail: str

class DriftLedger:
    """
    Minimal persistent ledger for constraint-carrying behavior.
    Store as JSON on disk; validate each new candidate response.
    """

    def __init__(self, path: str):
        self.path = path
        self.state: Dict[str, Any] = self._load_or_init()

    def _load_or_init(self) -> Dict[str, Any]:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {
                "session_id": "local",
                "turn_id": 0,
                "constraints": {
                    "structure_commitment": None,
                    "no_meta_commentary": False,
                    "no_prose": False,
                    "forbidden_terms": [],
                    "forbidden_synsets": []
                },
                "last": {}
            }

    def save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)

    # ---------- constraint utilities ----------

    def set_commitment(self, structure: str) -> None:
        self.state["constraints"]["structure_commitment"] = structure
        self.save()

    def add_forbidden_terms(self, terms: List[str]) -> None:
        ft = self.state["constraints"].setdefault("forbidden_terms", [])
        for t in terms:
            if t not in ft:
                ft.append(t)
        self.save()

    def set_no_meta_commentary(self, flag: bool) -> None:
        self.state["constraints"]["no_meta_commentary"] = bool(flag)
        self.save()

    # ---------- validators ----------

    def validate(self, text: str) -> Tuple[bool, List[Violation]]:
        c = self.state.get("constraints", {})
        violations: List[Violation] = []

        # Forbidden terms (literal)
        for term in c.get("forbidden_terms", []):
            if term and re.search(rf"\b{re.escape(term)}\b", text, re.IGNORECASE):
                violations.append(Violation("forbidden_term", f"term='{term}'"))

        # No meta-commentary heuristic
        if c.get("no_meta_commentary", False):
            meta_markers = ["rule", "violation", "constraint", "control", "i will now", "as an ai"]
            low = text.lower()
            if any(m in low for m in meta_markers):
                violations.append(Violation("meta_commentary", "meta markers present"))

        # Structure commitment heuristic
        commit = c.get("structure_commitment")
        if commit == "hierarchical":
            # Expect ordered structure cues; flag heavy rhizomatic cues (weak heuristic)
            if "1." not in text and "\n-" not in text and "\n•" not in text:
                violations.append(Violation("structure_break", "expected hierarchical markers"))
        elif commit == "rhizomatic":
            # Flag strong hierarchy: numbered list with linear build-up phrasing
            if re.search(r"^\s*\d+\.", text, re.MULTILINE):
                violations.append(Violation("structure_break", "numbered hierarchy detected"))

        return (len(violations) == 0), violations

    # ---------- turn processing ----------

    def process_turn(self, candidate_text: str) -> Dict[str, Any]:
        passed, violations = self.validate(candidate_text)
        self.state["turn_id"] = int(self.state.get("turn_id", 0)) + 1
        result = {
            "turn_id": self.state["turn_id"],
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "passed": passed,
            "violations": [asdict(v) for v in violations],
            "enforcement_action": "allow" if passed else "force_violation_report"
        }
        self.state["last"] = result
        self.save()
        return result

# module-level singleton for ease of calling from Kotlin
_LEDGER: Optional[DriftLedger] = None

def init(path: str) -> str:
    global _LEDGER
    _LEDGER = DriftLedger(path)
    return "ok"

def set_commitment(structure: str) -> str:
    if _LEDGER is None:
        return "err:not_initialized"
    _LEDGER.set_commitment(structure)
    return "ok"

def add_forbidden_terms(csv_terms: str) -> str:
    if _LEDGER is None:
        return "err:not_initialized"
    terms = [t.strip() for t in csv_terms.split(",") if t.strip()]
    _LEDGER.add_forbidden_terms(terms)
    return "ok"

def set_no_meta_commentary(flag: bool) -> str:
    if _LEDGER is None:
        return "err:not_initialized"
    _LEDGER.set_no_meta_commentary(flag)
    return "ok"

def validate_and_log(candidate_text: str) -> str:
    if _LEDGER is None:
        return json.dumps({"passed": False, "violations": [{"type":"init","detail":"not_initialized"}]})
    return json.dumps(_LEDGER.process_turn(candidate_text), ensure_ascii=False)
