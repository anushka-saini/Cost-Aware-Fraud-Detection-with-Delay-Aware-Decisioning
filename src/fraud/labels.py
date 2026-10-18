"""Labels that arrive after scoring, and when they may be used for training.

A fraud label is not known when a transaction is scored. It arrives later,
from an analyst's decision or a confirmed chargeback, and is only trusted once
it has had time to settle ("matured"). This module stores those labels with
both timestamps and answers one question for the retraining job: which labels
are usable as of a given day?

Time is measured in PaySim steps (hours), like everything else in the project.
"""

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

FIELDS = ["transaction_id", "is_fraud", "source", "transaction_step", "decided_step", "note"]
# Sources, most authoritative last. A later, stronger label replaces an earlier one.
SOURCE_PRIORITY = {"customer": 0, "analyst": 1, "chargeback": 2}


@dataclass
class Label:
    transaction_id: str
    is_fraud: int
    source: str                 # "analyst", "customer" or "chargeback"
    transaction_step: int       # when the transaction happened
    decided_step: int           # when the label was recorded
    note: str = ""

    def maturity_step(self, maturity_days: float) -> int:
        """First step at which this label counts as settled."""
        return int(self.transaction_step + maturity_days * 24)

    def is_mature(self, as_of_step: int, maturity_days: float) -> bool:
        return self.decided_step <= as_of_step and self.maturity_step(maturity_days) <= as_of_step


def usable_labels(labels, as_of_step: int, maturity_days: float) -> dict[str, Label]:
    """One settled label per transaction as of ``as_of_step``: the most
    authoritative source wins, then the most recent decision."""
    best: dict[str, Label] = {}
    for label in labels:
        if not label.is_mature(as_of_step, maturity_days):
            continue
        current = best.get(label.transaction_id)
        rank = (SOURCE_PRIORITY[label.source], label.decided_step)
        if current is None or rank >= (SOURCE_PRIORITY[current.source], current.decided_step):
            best[label.transaction_id] = label
    return best


class LabelStore:
    """Append-only CSV of feedback labels (a file-based stand-in for the
    service's feedback_labels table, used by tests and offline scripts)."""

    def __init__(self, path):
        self.path = Path(path)

    def add(self, label: Label) -> Label:
        if label.source not in SOURCE_PRIORITY:
            raise ValueError(f"Unknown label source: {label.source!r}")
        if label.is_fraud not in (0, 1):
            raise ValueError("is_fraud must be 0 or 1")
        new_file = not self.path.exists()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            if new_file:
                writer.writeheader()
            writer.writerow(asdict(label))
        return label

    def all(self) -> list[Label]:
        if not self.path.exists():
            return []
        with open(self.path, newline="", encoding="utf-8") as f:
            return [Label(r["transaction_id"], int(r["is_fraud"]), r["source"], int(r["transaction_step"]),
                          int(r["decided_step"]), r.get("note", "")) for r in csv.DictReader(f)]

    def usable(self, as_of_step: int, maturity_days: float) -> dict[str, Label]:
        return usable_labels(self.all(), as_of_step, maturity_days)


def last_fully_labelled_day(as_of_day: int, maturity_days: float) -> int:
    """The most recent whole day whose labels have all matured before day ``as_of_day`` starts."""
    return int(as_of_day - 1 - maturity_days)
