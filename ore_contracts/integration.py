"""Canonical B-N9 orchestration, backend routing and evidence governance."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol

from . import OperationMode, RunContext, utc_now_iso
from .advanced import (
    AcceptanceStatus, EvaluationReport, ExperimentSpec, TraceEvent,
)


CANONICAL_BIOLOGICAL_PHASES = tuple(f"B-N{i}" for i in range(1, 9))
INFERENCE_BIOLOGICAL_PHASES = tuple(f"B-N{i}" for i in range(1, 8))


@dataclass(frozen=True, slots=True)
class PhaseInvocation:
    phase_id: str
    original_payload: Any
    latest_input: Any
    upstream_outputs: Mapping[str, Any]
    run_context: RunContext
    validation_spec: ExperimentSpec | None = None
    learning_authorized: bool = False


class FunctionalEngine(Protocol):
    def run(self, payload: Any, context: RunContext) -> Any: ...


class BiologicalPhaseBackend(Protocol):
    def run(self, invocation: PhaseInvocation) -> Any: ...


@dataclass(frozen=True, slots=True)
class FrozenFusionPolicy:
    """A cross-track combiner must be fitted on validation data and frozen."""

    policy_id: str
    validation_manifest_ref: str
    parameter_fingerprint: str
    combine: Callable[[Any, Mapping[str, Any], RunContext], Mapping[str, Any]]
    validation_fitted: bool = True
    frozen: bool = True

    def __post_init__(self) -> None:
        for name in ("policy_id", "validation_manifest_ref", "parameter_fingerprint"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")
        if not callable(self.combine):
            raise TypeError("fusion combine must be callable")


class CanonicalBackendRegistry:
    """Dependency-injection registry with phase ownership validation."""

    def __init__(self) -> None:
        self._functional: FunctionalEngine | None = None
        self._biological: dict[str, BiologicalPhaseBackend] = {}
        self._fusion: FrozenFusionPolicy | None = None

    def register_functional(self, backend: FunctionalEngine) -> None:
        if not callable(getattr(backend, "run", None)):
            raise TypeError("functional backend must expose run(payload, context)")
        self._functional = backend

    def register_biological(self, phase_id: str, backend: BiologicalPhaseBackend) -> None:
        if phase_id not in CANONICAL_BIOLOGICAL_PHASES:
            raise ValueError("only canonical B-N1 through B-N8 backends can be registered; B-N9 is the controller")
        if not callable(getattr(backend, "run", None)):
            raise TypeError("biological backend must expose run(PhaseInvocation)")
        self._biological[phase_id] = backend

    def register_fusion_policy(self, policy: FrozenFusionPolicy) -> None:
        if not policy.validation_fitted or not policy.frozen:
            raise ValueError("fusion policy must be validation-fitted and frozen")
        self._fusion = policy

    @property
    def functional(self) -> FunctionalEngine | None:
        return self._functional

    @property
    def biological(self) -> Mapping[str, BiologicalPhaseBackend]:
        return dict(self._biological)

    @property
    def fusion_policy(self) -> FrozenFusionPolicy | None:
        return self._fusion


@dataclass(frozen=True, slots=True)
class CanonicalRunResult:
    run_id: str
    mode: OperationMode
    functional_output: Any | None
    biological_outputs: Mapping[str, Any]
    fused_output: Mapping[str, Any] | None
    channel_status: Mapping[str, str]
    trace: tuple[TraceEvent, ...]
    diagnostics: Mapping[str, Any]
    acceptance_status: Mapping[str, str]
    provenance: Mapping[str, Any]


class CanonicalOREController:
    """B-N9 coordinator for functional, biological and hybrid experiment modes.

    The controller never fabricates a missing backend. B-N8 is an offline
    validator and is run only in VALIDATE mode. B-N9 itself is this controller.
    """

    def __init__(self, registry: CanonicalBackendRegistry | None = None) -> None:
        self.registry = registry or CanonicalBackendRegistry()

    @staticmethod
    def _learning_authorized(payload: Any) -> bool:
        if isinstance(payload, Mapping):
            request = payload.get("learning_request")
            return bool(getattr(request, "explicitly_authorized", False))
        request = getattr(payload, "learning_request", None)
        return bool(getattr(request, "explicitly_authorized", False))

    def run(
        self,
        context: RunContext,
        payload: Any,
        *,
        validation_spec: ExperimentSpec | None = None,
    ) -> CanonicalRunResult:
        if not isinstance(context, RunContext):
            raise TypeError("context must be a RunContext")
        trace: list[TraceEvent] = []
        status: dict[str, str] = {}
        biological_outputs: dict[str, Any] = {}
        functional_output = None
        fused_output = None
        errors: list[str] = []

        def record(module: str, operation: str, event_status: str, inputs=(), outputs=(), diagnostics=None):
            trace.append(TraceEvent(
                event_index=len(trace),
                wall_clock_utc=utc_now_iso(),
                module_id=module,
                operation=operation,
                input_artifact_ids=tuple(inputs),
                output_artifact_ids=tuple(outputs),
                status=event_status,
                numerical_diagnostics=dict(diagnostics or {}),
            ))

        mode = context.mode
        run_functional = mode in {OperationMode.FUNCTIONAL, OperationMode.TRAIN, OperationMode.INFER, OperationMode.HYBRID}
        run_biological = mode in {OperationMode.BIOLOGICAL, OperationMode.HYBRID}
        run_validator = mode is OperationMode.VALIDATE

        if run_functional:
            backend = self.registry.functional
            if backend is None:
                status["FUNCTIONAL"] = "CHANNEL_UNAVAILABLE"
                record("B-N9", "route_functional_engine", "CHANNEL_UNAVAILABLE")
            else:
                try:
                    functional_output = backend.run(payload, context)
                    status["FUNCTIONAL"] = "EXECUTED"
                    record("B-N9", "run_functional_engine", "EXECUTED")
                except Exception as exc:
                    status["FUNCTIONAL"] = "FAILED"
                    errors.append(f"functional backend: {type(exc).__name__}: {exc}")
                    record("B-N9", "run_functional_engine", "FAILED", diagnostics={"error": str(exc)})

        if run_biological:
            latest_input = payload
            upstream: dict[str, Any] = {}
            authorized = self._learning_authorized(payload)
            for phase_id in INFERENCE_BIOLOGICAL_PHASES:
                backend = self.registry.biological.get(phase_id)
                if backend is None:
                    status[phase_id] = "CHANNEL_UNAVAILABLE"
                    biological_outputs[phase_id] = {"status": "CHANNEL_UNAVAILABLE"}
                    record("B-N9", f"route_{phase_id.lower()}", "CHANNEL_UNAVAILABLE")
                    continue
                invocation = PhaseInvocation(
                    phase_id=phase_id,
                    original_payload=payload,
                    latest_input=latest_input,
                    upstream_outputs=dict(upstream),
                    run_context=context,
                    validation_spec=None,
                    learning_authorized=authorized,
                )
                try:
                    output = backend.run(invocation)
                    upstream[phase_id] = output
                    biological_outputs[phase_id] = output
                    latest_input = output
                    status[phase_id] = "EXECUTED"
                    record(phase_id, "execute_canonical_phase", "EXECUTED")
                except Exception as exc:
                    status[phase_id] = "FAILED"
                    biological_outputs[phase_id] = {"status": "FAILED", "error": str(exc)}
                    errors.append(f"{phase_id}: {type(exc).__name__}: {exc}")
                    record(phase_id, "execute_canonical_phase", "FAILED", diagnostics={"error": str(exc)})
                    # Later phases depend on outputs whose upstream phase failed.
                    for downstream in INFERENCE_BIOLOGICAL_PHASES[INFERENCE_BIOLOGICAL_PHASES.index(phase_id) + 1:]:
                        status.setdefault(downstream, "BLOCKED_UPSTREAM_FAILURE")
                        biological_outputs.setdefault(downstream, {"status": "BLOCKED_UPSTREAM_FAILURE"})
                    break

        if run_validator:
            if validation_spec is None:
                status["B-N8"] = "NOT_RUN_MISSING_EXPERIMENT_SPEC"
                record("B-N9", "route_brain_data_validation", "NOT_RUN_MISSING_EXPERIMENT_SPEC")
            else:
                validator = self.registry.biological.get("B-N8")
                if validator is None:
                    status["B-N8"] = "CHANNEL_UNAVAILABLE"
                    record("B-N9", "route_brain_data_validation", "CHANNEL_UNAVAILABLE")
                else:
                    invocation = PhaseInvocation(
                        phase_id="B-N8",
                        original_payload=payload,
                        latest_input=payload,
                        upstream_outputs={},
                        run_context=context,
                        validation_spec=validation_spec,
                        learning_authorized=False,
                    )
                    try:
                        output = validator.run(invocation)
                        biological_outputs["B-N8"] = output
                        status["B-N8"] = "EXECUTED"
                        record("B-N8", "evaluate_saved_traces_and_predictions", "EXECUTED")
                    except Exception as exc:
                        status["B-N8"] = "FAILED"
                        errors.append(f"B-N8: {type(exc).__name__}: {exc}")
                        record("B-N8", "evaluate_saved_traces_and_predictions", "FAILED", diagnostics={"error": str(exc)})

        if mode is OperationMode.HYBRID:
            policy = self.registry.fusion_policy
            if policy is None:
                status["FUSION"] = "NOT_RUN_NO_FROZEN_VALIDATION_POLICY"
            elif functional_output is None or not any(v == "EXECUTED" for k, v in status.items() if k.startswith("B-N")):
                status["FUSION"] = "NOT_RUN_MISSING_BRANCH"
            else:
                try:
                    candidate = policy.combine(functional_output, dict(biological_outputs), context)
                    if not isinstance(candidate, Mapping):
                        raise TypeError("fusion policy must return a mapping")
                    fused_output = {
                        "policy_id": policy.policy_id,
                        "validation_manifest_ref": policy.validation_manifest_ref,
                        "parameter_fingerprint": policy.parameter_fingerprint,
                        "component_outputs_preserved": True,
                        "result": dict(candidate),
                    }
                    status["FUSION"] = "EXECUTED"
                    record("B-N9", "fuse_declared_evidence_channels", "EXECUTED")
                except Exception as exc:
                    status["FUSION"] = "FAILED"
                    errors.append(f"fusion: {type(exc).__name__}: {exc}")
                    record("B-N9", "fuse_declared_evidence_channels", "FAILED", diagnostics={"error": str(exc)})

        status.setdefault("B-N9", "EXECUTED")
        record("B-N9", "finalize_integrated_run_and_lineage", "EXECUTED")
        return CanonicalRunResult(
            run_id=context.run_id,
            mode=mode,
            functional_output=functional_output,
            biological_outputs=biological_outputs,
            fused_output=fused_output,
            channel_status=status,
            trace=tuple(trace),
            diagnostics={
                "valid_trace_order": all(event.event_index == i for i, event in enumerate(trace)),
                "error_count": len(errors),
                "errors": errors,
                "biological_validation_is_offline": True,
                "learning_authorized": self._learning_authorized(payload),
            },
            acceptance_status={
                "VAL-SOFTWARE": AcceptanceStatus.PENDING.value,
                "VAL-TASK": AcceptanceStatus.PENDING.value,
                "VAL-BIOLOGICAL": AcceptanceStatus.PENDING.value,
            },
            provenance={
                "controller": "canonical.B-N9",
                "functional_backend": type(self.registry.functional).__name__ if self.registry.functional else None,
                "biological_backends": {key: type(value).__name__ for key, value in self.registry.biological.items()},
                "canonical_phase_ids": list(INFERENCE_BIOLOGICAL_PHASES),
                "legacy_aliases_used": False,
            },
        )


class LegacyN9FunctionalBackend:
    """Explicit bridge to the legacy ore_n9 runner for compatibility experiments."""

    def __init__(self, processor: Any | None = None) -> None:
        if processor is None:
            from ore_n9 import N9Processor
            processor = N9Processor()
        if not callable(getattr(processor, "process", None)):
            raise TypeError("legacy N9 processor must expose process(N9Request)")
        self.processor = processor

    def run(self, payload: Any, context: RunContext) -> Any:
        from ore_n9 import N9Request
        if not isinstance(payload, N9Request):
            raise TypeError("LegacyN9FunctionalBackend expects an ore_n9.N9Request payload")
        return self.processor.process(payload)


__all__ = [
    "CANONICAL_BIOLOGICAL_PHASES", "INFERENCE_BIOLOGICAL_PHASES",
    "PhaseInvocation", "FunctionalEngine", "BiologicalPhaseBackend",
    "FrozenFusionPolicy", "CanonicalBackendRegistry", "CanonicalRunResult",
    "CanonicalOREController", "LegacyN9FunctionalBackend",
]
