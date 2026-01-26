"""Zone policy module for KiCad DRC and export operations.

This module is the single source of truth for zone refill/check policies.
It defines pinned toolchain options for deterministic behavior.

Satisfies REQ-M1-006: If CPWG uses zones, DRC MUST be run with zone refill
enabled and exports MUST be run with zone checks enabled (KiCad CLI
flags/policy pinned in code and recorded in manifest).

Satisfies REQ-M1-013: The manifest MUST include an explicit zone policy
record (refill/check behavior and toolchain versioning).

Usage:
    from formula_foundry.coupongen.kicad.policy import (
        ZonePolicy,
        DEFAULT_ZONE_POLICY,
        get_zone_policy_record,
    )

    # Get policy for manifest recording
    policy_dict = get_zone_policy_record(kicad_cli_version="9.0.7")

    # Access flags directly
    drc_flag = DEFAULT_ZONE_POLICY.drc_refill_flag      # "--refill-zones"
    export_flag = DEFAULT_ZONE_POLICY.export_check_flag  # "--check-zones"
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True)
class ZonePolicy:
    """Zone refill/check policy for DRC and export operations.

    This record is included in manifests to make zone behavior explicit
    and deterministic. All policy fields are immutable.

    Attributes:
        policy_id: Unique identifier for this policy version.
        drc_refill_zones: Whether to refill zones before DRC.
        drc_refill_flag: CLI flag to enable zone refill in DRC.
        export_check_zones: Whether to check zones during export.
        export_check_flag: CLI flag to enable zone check in export.
        kicad_cli_version: Optional version of kicad-cli for provenance.

    Example:
        >>> policy = ZonePolicy(
        ...     policy_id="custom-v1",
        ...     drc_refill_zones=True,
        ...     drc_refill_flag="--refill-zones",
        ...     export_check_zones=True,
        ...     export_check_flag="--check-zones",
        ... )
        >>> policy.to_dict()
        {'policy_id': 'custom-v1', 'drc': {...}, 'export': {...}}
    """

    policy_id: str
    drc_refill_zones: bool
    drc_refill_flag: str
    export_check_zones: bool
    export_check_flag: str
    kicad_cli_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert policy to a dictionary for manifest serialization.

        Returns:
            Dictionary suitable for JSON serialization with nested drc and
            export sections. The kicad_cli_version is only included if set.

        Example:
            >>> DEFAULT_ZONE_POLICY.to_dict()
            {
                'policy_id': 'kicad-cli-zones-v1',
                'drc': {'refill_zones': True, 'flag': '--refill-zones'},
                'export': {'check_zones': True, 'flag': '--check-zones'}
            }
        """
        record: dict[str, Any] = {
            "policy_id": self.policy_id,
            "drc": {
                "refill_zones": self.drc_refill_zones,
                "flag": self.drc_refill_flag,
            },
            "export": {
                "check_zones": self.export_check_zones,
                "flag": self.export_check_flag,
            },
        }
        if self.kicad_cli_version:
            record["kicad_cli_version"] = self.kicad_cli_version
        return record

    def with_kicad_cli_version(self, version: str | None) -> ZonePolicy:
        """Return a new policy with the specified kicad-cli version.

        This is used to stamp the policy with toolchain version info
        at manifest generation time.

        Args:
            version: The kicad-cli version string (e.g., "9.0.7").

        Returns:
            A new ZonePolicy instance with the updated version, or self
            if the version is unchanged.
        """
        if version == self.kicad_cli_version:
            return self
        return replace(self, kicad_cli_version=version)


# Default zone policy (REQ-M1-006)
#
# This policy enforces:
# - Zone refill before DRC to ensure copper pour accuracy
# - Zone check during export to validate zone integrity
#
# These flags are pinned for deterministic behavior across builds.
DEFAULT_ZONE_POLICY = ZonePolicy(
    policy_id="kicad-cli-zones-v1",
    drc_refill_zones=True,
    drc_refill_flag="--refill-zones",
    export_check_zones=True,
    export_check_flag="--check-zones",
)


def get_zone_policy_record(
    *,
    kicad_cli_version: str | None = None,
    policy: ZonePolicy | None = None,
) -> dict[str, Any]:
    """Get a zone policy record for manifest embedding.

    This is the primary API for obtaining zone policy data for manifests.
    It returns a dictionary suitable for JSON serialization.

    Args:
        kicad_cli_version: Optional version string to include for provenance.
        policy: Optional custom policy. Defaults to DEFAULT_ZONE_POLICY.

    Returns:
        Dictionary with policy details suitable for manifest.json.

    Example:
        >>> record = get_zone_policy_record(kicad_cli_version="9.0.7")
        >>> record["policy_id"]
        'kicad-cli-zones-v1'
        >>> record["kicad_cli_version"]
        '9.0.7'
    """
    effective_policy = policy or DEFAULT_ZONE_POLICY
    if kicad_cli_version:
        effective_policy = effective_policy.with_kicad_cli_version(kicad_cli_version)
    return effective_policy.to_dict()


__all__ = [
    "ZonePolicy",
    "DEFAULT_ZONE_POLICY",
    "get_zone_policy_record",
]
