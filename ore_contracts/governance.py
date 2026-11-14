"""Guarded learning transactions for IF-09, IF-11 and IF-15."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from .advanced import LearningRequest, LearningResult
from . import sha256_json


def model_fingerprint(model: Any) -> str:
    value = getattr(model, "fingerprint", None)
    if isinstance(value, str) and value:
        return value
    if callable(value):
        computed = value()
        if isinstance(computed, str) and computed:
            return computed
    to_dict = getattr(model, "to_dict", None)
    if callable(to_dict):
        try:
            payload = to_dict(include_fingerprint=False)
        except TypeError:
            payload = to_dict()
        return sha256_json(payload)
    raise TypeError("model must expose a fingerprint or deterministic to_dict()")


@dataclass(frozen=True, slots=True)
class LearningTransactionOutcome:
    result: LearningResult
    child_model: Any | None = None


class LearningTransactionManager:
    """Create and test a detached child; never mutate or promote the parent.

    The caller must explicitly opt in through IF-09 and provide regression
    gates. This manager returns an accepted child as a separate object; the
    caller remains responsible for durable promotion/version registration.
    """

    def execute(
        self,
        parent_model: Any,
        request: LearningRequest,
        proposer: Callable[[Any, LearningRequest], Any],
        regression_gates: Iterable[tuple[str, Callable[[Any, Any], bool]]],
        *,
        metrics_before: dict[str, float] | None = None,
        metrics_after: dict[str, float] | None = None,
    ) -> LearningTransactionOutcome:
        if not isinstance(request, LearningRequest):
            raise TypeError("request must be a LearningRequest")
        parent_before = model_fingerprint(parent_model)
        if parent_before != request.parent_model_fingerprint:
            return LearningTransactionOutcome(LearningResult(
                "REJECTED", parent_before, None, (), (), provenance={"reason": "parent fingerprint mismatch"}
            ))
        if not request.explicitly_authorized:
            return LearningTransactionOutcome(LearningResult(
                "REJECTED", parent_before, None, (), (), provenance={"reason": "explicit authorization missing"}
            ))
        if not request.verified_targets and not request.unsupervised_regime:
            return LearningTransactionOutcome(LearningResult(
                "REJECTED", parent_before, None, (), (), provenance={"reason": "learning evidence policy missing"}
            ))

        gates = tuple(regression_gates)
        if not gates:
            return LearningTransactionOutcome(LearningResult(
                "REJECTED", parent_before, None, (), (),
                provenance={"reason": "at least one regression gate is required"},
            ))

        detached_parent = deepcopy(parent_model)
        try:
            child = proposer(detached_parent, request)
        except Exception as exc:
            return LearningTransactionOutcome(LearningResult(
                "REJECTED", parent_before, None, (), ("proposal",),
                provenance={"reason": "proposal exception", "error": f"{type(exc).__name__}: {exc}"},
            ))

        parent_after = model_fingerprint(parent_model)
        tests: list[str] = []
        failures: list[str] = []
        if parent_after != parent_before:
            failures.append("active parent model mutated during proposal")
        if child is None or child is parent_model or child is detached_parent:
            failures.append("proposal did not return a distinct child model")

        for name, gate in gates:
            tests.append(name)
            try:
                passed = bool(gate(deepcopy(parent_model), deepcopy(child)))
            except Exception as exc:
                passed = False
                failures.append(f"{name}: exception {type(exc).__name__}: {exc}")
            if not passed:
                failures.append(f"{name}: failed")

        if model_fingerprint(parent_model) != parent_before:
            failures.append("active parent model mutated during regression evaluation")
        if failures:
            return LearningTransactionOutcome(LearningResult(
                "ROLLED_BACK", parent_before, None, (), tuple(tests),
                metrics_before or {}, metrics_after or {}, tuple(failures),
                {"parent_unchanged": parent_after == parent_before, "child_discarded": True},
            ))

        child_fingerprint = model_fingerprint(child)
        if child_fingerprint == parent_before:
            return LearningTransactionOutcome(LearningResult(
                "REJECTED", parent_before, None, (), tuple(tests),
                metrics_before or {}, metrics_after or {},
                provenance={"reason": "proposal produced no fingerprint change"},
            ))
        declared_parent = getattr(child, "parent_fingerprint", None)
        if declared_parent is not None and declared_parent != parent_before:
            return LearningTransactionOutcome(LearningResult(
                "ROLLED_BACK", parent_before, None, (), tuple(tests),
                metrics_before or {}, metrics_after or {},
                ("child parent_fingerprint does not match the requested parent",),
                {"child_discarded": True},
            ))

        result = LearningResult(
            "ACCEPTED", parent_before, child_fingerprint,
            tuple(request.update_channels), tuple(tests),
            metrics_before or {}, metrics_after or {}, (),
            {"parent_unchanged": True, "promotion": "NOT_PERFORMED"},
        )
        return LearningTransactionOutcome(result, child)


__all__ = ["model_fingerprint", "LearningTransactionOutcome", "LearningTransactionManager"]
