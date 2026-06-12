"""Weight profiles for the ICRS hybrid scoring engine (Task 9.1).

The composite score fuses four sub-scores under role-appropriate weights
(design "Scoring Architecture")::

    FinalScore = w1 * SemanticFitScore
               + w2 * CareerTrajectoryScore
               + w3 * BehavioralSignalScore
               + w4 * HardFilterPassScore
               - w5 * DisqualifyingFlagPenalty

This module defines :class:`WeightProfile` (the configurable ``w1..w5`` set),
the default and per-job-type registry of profiles (weight profiles are *data*,
not code), and :func:`select_weight_profile`, which resolves the profile for a
supplied job type with the validation and fallback behaviour required by the
acceptance criteria.

Validation / selection rules enforced here:
    - Each weight component is in the inclusive range ``[0,1]`` (Requirement 4.3,
      enforced at construction by field bounds).
    - The four fusion weights must satisfy ``w1 + w2 + w3 + w4 = 1.0`` within a
      tolerance of ``±0.001`` (Requirement 4.3). ``w5`` is an *independent*
      penalty coefficient and is deliberately excluded from this sum.
    - ``select_weight_profile`` returns the configured profile for the supplied
      job type, falling back to the default profile when no profile is defined
      for that type (Requirement 4.2).
    - If a selected profile's fusion weights do not sum to 1.0 within tolerance,
      it is rejected, the default is substituted, and the returned
      :class:`WeightProfileSelection` records that the profile was rejected
      (Requirement 4.8).
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field

from icrs.models.job import JobType

# Tolerance for the w1+w2+w3+w4 == 1.0 sum-to-one constraint (Requirement 4.3).
WEIGHT_SUM_TOLERANCE = 1e-3


class WeightProfile(BaseModel):
    """A configurable set of fusion weights applied during composite scoring.

    The four fusion weights (``w1``..``w4``) are constrained to ``[0,1]`` and
    are expected to sum to ``1.0`` within :data:`WEIGHT_SUM_TOLERANCE`. The
    sum-to-one constraint is intentionally *not* enforced at construction so an
    out-of-spec (e.g. recruiter-overridden) profile can be represented and then
    rejected by :func:`select_weight_profile` (Requirement 4.8). Per-component
    bounds, however, are enforced here.

    ``w5`` is an independent disqualifying-penalty coefficient; it is subtracted
    in the composite formula and does **not** participate in the sum-to-one
    constraint.
    """

    model_config = ConfigDict(frozen=True)

    w1: float = Field(
        ..., ge=0.0, le=1.0, description="Semantic fit weight, in [0,1]."
    )
    w2: float = Field(
        ..., ge=0.0, le=1.0, description="Career trajectory weight, in [0,1]."
    )
    w3: float = Field(
        ..., ge=0.0, le=1.0, description="Behavioral signal weight, in [0,1]."
    )
    w4: float = Field(
        ..., ge=0.0, le=1.0, description="Hard-filter pass weight, in [0,1]."
    )
    w5: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Disqualifying-penalty coefficient, independent of the "
        "sum-to-one constraint.",
    )

    # Descriptive aliases for the fusion weights (the design names them by role).
    @property
    def semantic(self) -> float:
        """Alias for ``w1`` (semantic fit weight)."""

        return self.w1

    @property
    def trajectory(self) -> float:
        """Alias for ``w2`` (career trajectory weight)."""

        return self.w2

    @property
    def behavioral(self) -> float:
        """Alias for ``w3`` (behavioral signal weight)."""

        return self.w3

    @property
    def hard_filter(self) -> float:
        """Alias for ``w4`` (hard-filter pass weight)."""

        return self.w4

    @property
    def penalty(self) -> float:
        """Alias for ``w5`` (disqualifying-penalty coefficient)."""

        return self.w5

    @property
    def fusion_sum(self) -> float:
        """The sum of the four fusion weights ``w1 + w2 + w3 + w4``."""

        return self.w1 + self.w2 + self.w3 + self.w4

    def sums_to_one(self, tolerance: float = WEIGHT_SUM_TOLERANCE) -> bool:
        """Whether ``w1+w2+w3+w4`` equals 1.0 within ``tolerance`` (Req 4.3)."""

        return abs(self.fusion_sum - 1.0) <= tolerance


# ----- Default and per-job-type registry (weight profiles are data) ---------

# Design "Default weights and per-job-type configuration".
DEFAULT_WEIGHT_PROFILE = WeightProfile(w1=0.40, w2=0.20, w3=0.15, w4=0.25, w5=0.30)

WEIGHT_PROFILES: dict[JobType, WeightProfile] = {
    JobType.TECHNICAL: WeightProfile(w1=0.35, w2=0.15, w3=0.30, w4=0.20, w5=0.30),
    JobType.LEADERSHIP: WeightProfile(w1=0.30, w2=0.35, w3=0.10, w4=0.25, w5=0.35),
    JobType.GENERALIST: WeightProfile(w1=0.45, w2=0.20, w3=0.10, w4=0.25, w5=0.25),
    JobType.SALES: WeightProfile(w1=0.35, w2=0.30, w3=0.10, w4=0.25, w5=0.30),
}


class WeightProfileSelection(BaseModel):
    """The resolved weight profile plus an indication of how it was resolved.

    Fields:
        profile: The :class:`WeightProfile` to use for scoring.
        requested_job_type: The job type that was requested (``None`` if none).
        fell_back: ``True`` when the default profile was substituted because no
            profile was defined for the requested type, or because the
            configured profile was rejected.
        rejected: ``True`` when a configured profile existed but was rejected
            for failing the sum-to-one constraint (Requirement 4.8).
        reason: A human-readable indication of why a fallback/rejection
            occurred (``None`` when the configured profile was used as-is).
    """

    model_config = ConfigDict(frozen=True)

    profile: WeightProfile
    requested_job_type: JobType | None = None
    fell_back: bool = False
    rejected: bool = False
    reason: str | None = None


def select_weight_profile(
    job_type: JobType | None,
    *,
    registry: Mapping[JobType, WeightProfile] | None = None,
    tolerance: float = WEIGHT_SUM_TOLERANCE,
) -> WeightProfileSelection:
    """Select the :class:`WeightProfile` for ``job_type`` with validated fallback.

    Behaviour (Requirements 4.2, 4.3, 4.8):
        - If a profile is configured for ``job_type`` and its fusion weights sum
          to 1.0 within ``tolerance``, that profile is returned unchanged.
        - If no profile is configured for ``job_type`` (including ``None`` or an
          unknown type), the default profile is returned with ``fell_back`` set.
        - If a configured profile exists but its fusion weights do not sum to
          1.0 within ``tolerance``, it is rejected, the default profile is
          returned, and both ``rejected`` and ``fell_back`` are set with a
          recorded ``reason``.

    Args:
        job_type: The job type whose profile to select; may be ``None``.
        registry: Optional override of the job-type → profile mapping. Defaults
            to the module-level :data:`WEIGHT_PROFILES`.
        tolerance: Sum-to-one tolerance; defaults to :data:`WEIGHT_SUM_TOLERANCE`.

    Returns:
        A :class:`WeightProfileSelection` carrying the resolved profile and the
        fallback/rejection indication.
    """

    reg = WEIGHT_PROFILES if registry is None else registry
    profile = reg.get(job_type) if job_type is not None else None

    if profile is None:
        reason = (
            f"No WeightProfile defined for job type {_fmt_job_type(job_type)}; "
            "falling back to the default WeightProfile."
        )
        return WeightProfileSelection(
            profile=DEFAULT_WEIGHT_PROFILE,
            requested_job_type=job_type,
            fell_back=True,
            rejected=False,
            reason=reason,
        )

    if not profile.sums_to_one(tolerance):
        reason = (
            f"WeightProfile for job type {_fmt_job_type(job_type)} rejected: "
            f"w1+w2+w3+w4={profile.fusion_sum:.6f} is not 1.0 within "
            f"±{tolerance}; falling back to the default WeightProfile."
        )
        return WeightProfileSelection(
            profile=DEFAULT_WEIGHT_PROFILE,
            requested_job_type=job_type,
            fell_back=True,
            rejected=True,
            reason=reason,
        )

    return WeightProfileSelection(
        profile=profile,
        requested_job_type=job_type,
        fell_back=False,
        rejected=False,
        reason=None,
    )


def get_weight_profile(
    job_type: JobType | None,
    *,
    registry: Mapping[JobType, WeightProfile] | None = None,
    tolerance: float = WEIGHT_SUM_TOLERANCE,
) -> WeightProfile:
    """Convenience wrapper returning only the resolved :class:`WeightProfile`.

    Discards the fallback/rejection indication carried by
    :func:`select_weight_profile`; use the full selector when that indication
    must be recorded.
    """

    return select_weight_profile(
        job_type, registry=registry, tolerance=tolerance
    ).profile


def _fmt_job_type(job_type: JobType | None) -> str:
    """Render a job type for inclusion in a human-readable indication."""

    if job_type is None:
        return "None"
    value = getattr(job_type, "value", job_type)
    return str(value)


__all__ = [
    "WEIGHT_SUM_TOLERANCE",
    "WeightProfile",
    "WeightProfileSelection",
    "DEFAULT_WEIGHT_PROFILE",
    "WEIGHT_PROFILES",
    "select_weight_profile",
    "get_weight_profile",
]
