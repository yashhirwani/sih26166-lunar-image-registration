"""
exceptions.py
-------------
Custom exception hierarchy for the Lunar Image Registration pipeline.
PS166 — SIH 2026

All pipeline errors inherit from RegistrationError so callers can catch the
entire family with a single `except RegistrationError` in the UI, while still
being able to distinguish specific failure modes when needed.

Every exception carries:
  - message (str): human-readable description, safe to show in the UI
  - stage (str): which pipeline step raised this (e.g. "Step 0 — Validation")
  - details (dict): machine-readable metadata (confidence flag, counts, paths…)

Usage example:
    raise InvalidInputError(
        "Source image is all-black — no features can be detected.",
        stage="Step 0 — Validation",
        details={"confidence": "invalid", "mean_intensity": 0.0}
    )
"""


class RegistrationError(Exception):
    """
    Base class for all pipeline-specific errors.

    Attributes:
        message (str): Human-readable error description, safe to display in UI.
        stage (str): Pipeline step that raised this error.
        details (dict): Machine-readable metadata for logging and debugging.
        confidence (str): Validity flag — one of:
            "invalid"    — input rejected before any processing
            "failed"     — processing started but could not complete
            "low"        — result produced but reliability is suspect
            "ok"         — result is reliable (this exception should NOT carry "ok")
    """

    def __init__(self, message: str, stage: str = "unknown", details: dict = None):
        super().__init__(message)
        self.message = message
        self.stage = stage
        self.details = details or {}
        # Subclasses can override this default
        self.confidence = self.details.get("confidence", "failed")

    def __str__(self):
        return f"[{self.stage}] {self.message}"

    def as_dict(self) -> dict:
        """Returns a serialisable summary for logging."""
        return {
            "error_type": type(self).__name__,
            "stage": self.stage,
            "message": self.message,
            "confidence": self.confidence,
            "details": self.details,
        }


# ── Specific exception types ────────────────────────────────────────────────

class InvalidInputError(RegistrationError):
    """
    Raised at Step 0 (Input Validation) or Step 1 (Loading) when the supplied
    images cannot be accepted as valid pipeline inputs.

    Covers:
    - Missing or unreadable file
    - Unsupported / corrupted format
    - All-black or all-white image
    - NaN / Inf pixels detected
    - Identical images supplied as source and reference
    - Plausible-overlap pre-check failed (near-zero overlap)
    - Oversized file (would exceed RAM budget)

    confidence: always "invalid"
    """

    def __init__(self, message: str, stage: str = "Step 0 — Validation", details: dict = None):
        details = details or {}
        details.setdefault("confidence", "invalid")
        super().__init__(message, stage=stage, details=details)
        self.confidence = "invalid"


class InsufficientDataError(RegistrationError):
    """
    Raised when there is technically valid input but not enough data to
    complete a pipeline step reliably.

    Covers:
    - Too few keypoints detected (< 10 → low-confidence; < 4 → hard abort)
    - Too few matches surviving the ratio test
    - Too few inliers after outlier rejection (< 4 → cannot call findHomography)
    - All matches clustered in one spatial region

    confidence: "low" when a degraded result is returned, "failed" when aborted.
    """

    def __init__(self, message: str, stage: str = "unknown", details: dict = None):
        details = details or {}
        details.setdefault("confidence", "failed")
        super().__init__(message, stage=stage, details=details)
        self.confidence = details["confidence"]


class DegenerateGeometryError(RegistrationError):
    """
    Raised when a geometric computation fails due to degenerate conditions.

    Covers:
    - cv2.findHomography returns None
    - Homography produces unrealistic transform (det ≈ 0, extreme shear/scale)
    - Homography fallback to affine also fails
    - Warped output is mostly black (likely misregistration)
    - Canvas size computed as zero or negative

    confidence: "failed" (result cannot be trusted)
    """

    def __init__(self, message: str, stage: str = "Step 8 — Transform", details: dict = None):
        details = details or {}
        details.setdefault("confidence", "failed")
        super().__init__(message, stage=stage, details=details)
        self.confidence = "failed"


class RefinementFailedError(RegistrationError):
    """
    Raised when sub-pixel refinement via phase correlation cannot produce a
    better result than the pre-refinement alignment.

    This is NOT a fatal error — the caller must catch it and fall back to the
    pre-refinement result rather than aborting the pipeline entirely.

    Covers:
    - Phase correlation diverges (shift larger than safety threshold)
    - Refinement on featureless / flat region (response too low)
    - RMSE after refinement is worse than RMSE before refinement

    confidence: "low" — the pre-refinement result is still returned; the
    sub-pixel accuracy claim cannot be made.
    """

    def __init__(self, message: str, stage: str = "Step 9 — Refinement", details: dict = None):
        details = details or {}
        details.setdefault("confidence", "low")
        super().__init__(message, stage=stage, details=details)
        self.confidence = "low"
