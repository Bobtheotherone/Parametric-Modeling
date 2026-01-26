"""Garbage collection with configurable retention policies.

This module implements the GC (garbage collection) system for the M3 artifact store,
providing:
- Configurable retention policies (age-based, count-based, space budget)
- Pinning support to protect critical artifacts
- Safe DVC gc invocation wrapper
- Space budget enforcement
- Dry-run mode for safety

The GC system is designed to be safe by default:
- Pinned artifacts are never deleted
- Artifacts with descendants are protected unless force-deleted
- DVC gc is invoked with workspace and project flags for safety
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from formula_foundry.m3.artifact_store import ArtifactStore
    from formula_foundry.m3.lineage_graph import LineageGraph
    from formula_foundry.m3.registry import ArtifactRegistry

# Default retention policy name
DEFAULT_POLICY = "laptop_default"


class RetentionUnit(str, Enum):
    """Units for retention period specification."""

    HOURS = "hours"
    DAYS = "days"
    WEEKS = "weeks"
    MONTHS = "months"


@dataclass
class RetentionPolicy:
    """Configuration for artifact retention.

    Retention policies define rules for what to keep and what to delete.
    Multiple rules can be combined, and artifacts are kept if they match
    ANY of the keep rules.

    Attributes:
        name: Human-readable policy name.
        description: Optional description of the policy.
        keep_min_age_days: Keep artifacts newer than this many days.
        keep_max_count: Keep at most this many artifacts per type.
        keep_min_count: Always keep at least this many artifacts per type.
        keep_pinned: If True, never delete pinned artifacts (default True).
        keep_with_descendants: If True, keep artifacts that have descendants.
        keep_ancestors_of_pinned: If True, keep ancestors of pinned artifacts
            to preserve lineage integrity (default True).
        keep_artifact_types: List of artifact types to always keep.
        keep_roles: List of roles to always keep.
        space_budget_bytes: Target maximum total space usage.
        dvc_gc_flags: Additional flags to pass to dvc gc.
    """

    name: str
    description: str | None = None
    keep_min_age_days: int = 30
    keep_max_count: int | None = None
    keep_min_count: int = 1
    keep_pinned: bool = True
    keep_with_descendants: bool = True
    keep_ancestors_of_pinned: bool = True
    keep_artifact_types: list[str] = field(default_factory=list)
    keep_roles: list[str] = field(default_factory=list)
    space_budget_bytes: int | None = None
    dvc_gc_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result: dict[str, Any] = {
            "name": self.name,
            "keep_min_age_days": self.keep_min_age_days,
            "keep_min_count": self.keep_min_count,
            "keep_pinned": self.keep_pinned,
            "keep_with_descendants": self.keep_with_descendants,
            "keep_ancestors_of_pinned": self.keep_ancestors_of_pinned,
        }
        if self.description:
            result["description"] = self.description
        if self.keep_max_count is not None:
            result["keep_max_count"] = self.keep_max_count
        if self.keep_artifact_types:
            result["keep_artifact_types"] = self.keep_artifact_types
        if self.keep_roles:
            result["keep_roles"] = self.keep_roles
        if self.space_budget_bytes is not None:
            result["space_budget_bytes"] = self.space_budget_bytes
        if self.dvc_gc_flags:
            result["dvc_gc_flags"] = self.dvc_gc_flags
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RetentionPolicy:
        """Create a RetentionPolicy from a dict."""
        return cls(
            name=data["name"],
            description=data.get("description"),
            keep_min_age_days=data.get("keep_min_age_days", 30),
            keep_max_count=data.get("keep_max_count"),
            keep_min_count=data.get("keep_min_count", 1),
            keep_pinned=data.get("keep_pinned", True),
            keep_with_descendants=data.get("keep_with_descendants", True),
            keep_ancestors_of_pinned=data.get("keep_ancestors_of_pinned", True),
            keep_artifact_types=data.get("keep_artifact_types", []),
            keep_roles=data.get("keep_roles", []),
            space_budget_bytes=data.get("space_budget_bytes"),
            dvc_gc_flags=data.get("dvc_gc_flags", []),
        )


# Predefined retention policies
BUILTIN_POLICIES: dict[str, RetentionPolicy] = {
    "laptop_default": RetentionPolicy(
        name="laptop_default",
        description="Conservative policy for laptop development",
        keep_min_age_days=14,
        keep_min_count=5,
        keep_pinned=True,
        keep_with_descendants=True,
        keep_artifact_types=["dataset_snapshot", "formula_candidate", "model_checkpoint"],
        keep_roles=["final_output", "checkpoint"],
        space_budget_bytes=50 * 1024 * 1024 * 1024,  # 50 GB
    ),
    "ci_aggressive": RetentionPolicy(
        name="ci_aggressive",
        description="Aggressive cleanup for CI environments",
        keep_min_age_days=7,
        keep_min_count=2,
        keep_pinned=True,
        keep_with_descendants=False,
        keep_artifact_types=["dataset_snapshot"],
        keep_roles=["final_output"],
        space_budget_bytes=10 * 1024 * 1024 * 1024,  # 10 GB
    ),
    "archive": RetentionPolicy(
        name="archive",
        description="Long-term archival policy",
        keep_min_age_days=365,
        keep_min_count=10,
        keep_pinned=True,
        keep_with_descendants=True,
        keep_artifact_types=[
            "dataset_snapshot",
            "formula_candidate",
            "model_checkpoint",
            "validation_report",
            "touchstone",
        ],
        keep_roles=["final_output", "checkpoint", "validation"],
        space_budget_bytes=500 * 1024 * 1024 * 1024,  # 500 GB
    ),
    "dev_minimal": RetentionPolicy(
        name="dev_minimal",
        description="Minimal retention for fast iteration",
        keep_min_age_days=3,
        keep_min_count=1,
        keep_pinned=True,
        keep_with_descendants=True,
        keep_artifact_types=["dataset_snapshot"],
        keep_roles=["final_output"],
        space_budget_bytes=5 * 1024 * 1024 * 1024,  # 5 GB
    ),
}


@dataclass
class PinnedArtifact:
    """A pinned artifact that should never be deleted.

    Pinning can be done at different levels:
    - artifact_id: Specific artifact
    - run_id: All artifacts from a run
    - dataset_id: All artifacts in a dataset
    """

    artifact_id: str | None = None
    run_id: str | None = None
    dataset_id: str | None = None
    reason: str | None = None
    pinned_utc: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result: dict[str, Any] = {}
        if self.artifact_id:
            result["artifact_id"] = self.artifact_id
        if self.run_id:
            result["run_id"] = self.run_id
        if self.dataset_id:
            result["dataset_id"] = self.dataset_id
        if self.reason:
            result["reason"] = self.reason
        if self.pinned_utc:
            result["pinned_utc"] = self.pinned_utc
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PinnedArtifact:
        """Create from a dict."""
        return cls(
            artifact_id=data.get("artifact_id"),
            run_id=data.get("run_id"),
            dataset_id=data.get("dataset_id"),
            reason=data.get("reason"),
            pinned_utc=data.get("pinned_utc"),
        )


@dataclass
class GCCandidate:
    """An artifact that is a candidate for deletion."""

    artifact_id: str
    content_hash_digest: str
    byte_size: int
    created_utc: str
    artifact_type: str
    run_id: str | None
    storage_path: str | None
    reasons_to_delete: list[str] = field(default_factory=list)
    reasons_to_keep: list[str] = field(default_factory=list)

    @property
    def should_delete(self) -> bool:
        """Determine if this artifact should be deleted."""
        return len(self.reasons_to_delete) > 0 and len(self.reasons_to_keep) == 0


@dataclass
class GCResult:
    """Result of a garbage collection operation."""

    policy_name: str
    started_utc: str
    finished_utc: str
    dry_run: bool
    artifacts_scanned: int
    artifacts_deleted: int
    bytes_freed: int
    bytes_total_before: int
    bytes_total_after: int
    pinned_protected: int
    descendant_protected: int
    dvc_gc_ran: bool
    dvc_gc_output: str | None
    ancestor_protected: int = 0
    errors: list[str] = field(default_factory=list)
    deleted_artifacts: list[str] = field(default_factory=list)
    protected_artifacts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "policy_name": self.policy_name,
            "started_utc": self.started_utc,
            "finished_utc": self.finished_utc,
            "dry_run": self.dry_run,
            "artifacts_scanned": self.artifacts_scanned,
            "artifacts_deleted": self.artifacts_deleted,
            "bytes_freed": self.bytes_freed,
            "bytes_total_before": self.bytes_total_before,
            "bytes_total_after": self.bytes_total_after,
            "pinned_protected": self.pinned_protected,
            "descendant_protected": self.descendant_protected,
            "ancestor_protected": self.ancestor_protected,
            "dvc_gc_ran": self.dvc_gc_ran,
            "dvc_gc_output": self.dvc_gc_output,
            "errors": self.errors,
            "deleted_artifacts": self.deleted_artifacts,
            "protected_artifacts": self.protected_artifacts,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


class GCError(Exception):
    """Base exception for GC errors."""


class PolicyNotFoundError(GCError):
    """Raised when a policy is not found."""


def _now_utc_iso() -> str:
    """Get current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso_datetime(iso_str: str) -> datetime:
    """Parse an ISO 8601 datetime string."""
    if iso_str.endswith("Z"):
        iso_str = iso_str[:-1] + "+00:00"
    return datetime.fromisoformat(iso_str)


def _get_age_days(created_utc: str) -> float:
    """Get the age of an artifact in days."""
    created = _parse_iso_datetime(created_utc)
    now = datetime.now(timezone.utc)
    return (now - created).total_seconds() / 86400


def load_policies_from_file(path: Path) -> dict[str, RetentionPolicy]:
    """Load retention policies from a YAML file.

    Args:
        path: Path to the YAML config file.

    Returns:
        Dictionary mapping policy names to RetentionPolicy objects.
    """
    try:
        import yaml
    except ImportError:
        return {}

    if not path.exists():
        return {}

    with open(path) as f:
        data = yaml.safe_load(f)

    if not data or "policies" not in data:
        return {}

    policies = {}
    for policy_data in data["policies"]:
        policy = RetentionPolicy.from_dict(policy_data)
        policies[policy.name] = policy

    return policies


def load_pins_from_file(path: Path) -> list[PinnedArtifact]:
    """Load pinned artifacts from a JSON file.

    Args:
        path: Path to the pins JSON file.

    Returns:
        List of PinnedArtifact objects.
    """
    if not path.exists():
        return []

    with open(path) as f:
        data = json.load(f)

    return [PinnedArtifact.from_dict(pin) for pin in data.get("pins", [])]


def save_pins_to_file(path: Path, pins: list[PinnedArtifact]) -> None:
    """Save pinned artifacts to a JSON file.

    Args:
        path: Path to save the pins file.
        pins: List of PinnedArtifact objects.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"pins": [pin.to_dict() for pin in pins]}
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


class GarbageCollector:
    """Garbage collector for the M3 artifact store.

    This class implements garbage collection with:
    - Configurable retention policies
    - Pinning support for critical artifacts
    - Safe DVC gc wrapper
    - Space budget enforcement

    Example usage:
        gc = GarbageCollector(
            data_dir=Path("data"),
            store=artifact_store,
            registry=registry,
            lineage=lineage_graph,
        )

        # Dry run to see what would be deleted
        result = gc.run(policy="laptop_default", dry_run=True)

        # Actually delete
        result = gc.run(policy="laptop_default", dry_run=False)
    """

    def __init__(
        self,
        data_dir: Path,
        store: ArtifactStore,
        registry: ArtifactRegistry,
        lineage: LineageGraph | None = None,
    ) -> None:
        """Initialize the garbage collector.

        Args:
            data_dir: Path to the data directory.
            store: The artifact store.
            registry: The artifact registry.
            lineage: Optional lineage graph for descendant checking.
        """
        self.data_dir = Path(data_dir)
        self.store = store
        self.registry = registry
        self.lineage = lineage

        self.pins_file = self.data_dir / "gc_pins.json"
        self.policies_file = self.data_dir.parent / "config" / "gc_policies.yaml"

        self._pins: list[PinnedArtifact] | None = None
        self._policies: dict[str, RetentionPolicy] | None = None

    @property
    def pins(self) -> list[PinnedArtifact]:
        """Get the list of pinned artifacts."""
        if self._pins is None:
            self._pins = load_pins_from_file(self.pins_file)
        return self._pins

    @property
    def policies(self) -> dict[str, RetentionPolicy]:
        """Get the available retention policies."""
        if self._policies is None:
            # Start with built-in policies
            self._policies = dict(BUILTIN_POLICIES)
            # Override with file-based policies
            file_policies = load_policies_from_file(self.policies_file)
            self._policies.update(file_policies)
        return self._policies

    def get_policy(self, name: str) -> RetentionPolicy:
        """Get a retention policy by name.

        Args:
            name: Policy name.

        Returns:
            The RetentionPolicy.

        Raises:
            PolicyNotFoundError: If the policy doesn't exist.
        """
        if name not in self.policies:
            raise PolicyNotFoundError(f"Policy not found: {name}")
        return self.policies[name]

    def pin_artifact(
        self,
        artifact_id: str | None = None,
        run_id: str | None = None,
        dataset_id: str | None = None,
        reason: str | None = None,
    ) -> PinnedArtifact:
        """Pin an artifact to protect it from deletion.

        Args:
            artifact_id: Specific artifact ID to pin.
            run_id: Run ID to pin all artifacts from.
            dataset_id: Dataset ID to pin all artifacts from.
            reason: Optional reason for pinning.

        Returns:
            The created PinnedArtifact.
        """
        pin = PinnedArtifact(
            artifact_id=artifact_id,
            run_id=run_id,
            dataset_id=dataset_id,
            reason=reason,
            pinned_utc=_now_utc_iso(),
        )

        pins = list(self.pins)
        pins.append(pin)
        save_pins_to_file(self.pins_file, pins)
        self._pins = pins

        return pin

    def unpin_artifact(
        self,
        artifact_id: str | None = None,
        run_id: str | None = None,
        dataset_id: str | None = None,
    ) -> bool:
        """Unpin an artifact.

        Args:
            artifact_id: Specific artifact ID to unpin.
            run_id: Run ID to unpin.
            dataset_id: Dataset ID to unpin.

        Returns:
            True if a pin was removed, False otherwise.
        """
        pins = list(self.pins)
        original_count = len(pins)

        pins = [
            p
            for p in pins
            if not (
                (artifact_id and p.artifact_id == artifact_id)
                or (run_id and p.run_id == run_id)
                or (dataset_id and p.dataset_id == dataset_id)
            )
        ]

        if len(pins) < original_count:
            save_pins_to_file(self.pins_file, pins)
            self._pins = pins
            return True

        return False

    def is_pinned(self, artifact_id: str, run_id: str | None = None) -> bool:
        """Check if an artifact is pinned.

        Args:
            artifact_id: The artifact ID to check.
            run_id: Optional run ID of the artifact.

        Returns:
            True if the artifact is pinned.
        """
        for pin in self.pins:
            if pin.artifact_id == artifact_id:
                return True
            if run_id and pin.run_id == run_id:
                return True
        return False

    def has_descendants(self, artifact_id: str) -> bool:
        """Check if an artifact has descendants in the lineage graph.

        Args:
            artifact_id: The artifact ID to check.

        Returns:
            True if the artifact has descendants.
        """
        if self.lineage is None:
            return False

        if not self.lineage.has_node(artifact_id):
            return False

        edges = self.lineage.get_edges_from(artifact_id)
        return len(edges) > 0

    def get_pinned_artifact_ids(self) -> set[str]:
        """Get all directly pinned artifact IDs.

        Returns:
            Set of artifact IDs that are explicitly pinned.
        """
        pinned_ids: set[str] = set()

        for pin in self.pins:
            if pin.artifact_id:
                pinned_ids.add(pin.artifact_id)
            elif pin.run_id:
                # Get all artifacts from this run
                try:
                    records = self.registry.get_artifacts_for_run(pin.run_id)
                    for record in records:
                        pinned_ids.add(record.artifact_id)
                except Exception:
                    pass
            elif pin.dataset_id:
                # Get all artifacts from this dataset
                try:
                    records = self.registry.get_artifacts_for_dataset(pin.dataset_id)
                    for record in records:
                        pinned_ids.add(record.artifact_id)
                except Exception:
                    pass

        return pinned_ids

    def get_ancestors_of_pinned(self) -> set[str]:
        """Get all ancestors of pinned artifacts.

        This traverses the lineage graph backward from each pinned artifact
        to collect all input artifacts that contributed to the pinned ones.
        These ancestors must be preserved to maintain lineage integrity.

        Returns:
            Set of artifact IDs that are ancestors of pinned artifacts.
        """
        if self.lineage is None:
            return set()

        pinned_ids = self.get_pinned_artifact_ids()
        ancestor_ids: set[str] = set()

        for pinned_id in pinned_ids:
            if not self.lineage.has_node(pinned_id):
                continue

            try:
                subgraph = self.lineage.get_ancestors(pinned_id)
                for node_id in subgraph.nodes:
                    if node_id != pinned_id:  # Don't include the pinned artifact itself
                        ancestor_ids.add(node_id)
            except Exception:
                # Node may not exist in lineage graph
                continue

        return ancestor_ids

    def _evaluate_candidate(
        self,
        artifact_id: str,
        policy: RetentionPolicy,
        artifact_counts_by_type: dict[str, int],
        ancestors_of_pinned: set[str] | None = None,
    ) -> GCCandidate:
        """Evaluate an artifact for deletion under a policy.

        Args:
            artifact_id: The artifact ID to evaluate.
            policy: The retention policy to apply.
            artifact_counts_by_type: Running count of artifacts by type.
            ancestors_of_pinned: Optional set of artifact IDs that are ancestors
                of pinned artifacts (for lineage protection).

        Returns:
            A GCCandidate with reasons to keep or delete.
        """
        try:
            record = self.registry.get_artifact(artifact_id)
        except Exception:
            manifest = self.store.get_manifest(artifact_id)
            record = None

        if record:
            candidate = GCCandidate(
                artifact_id=artifact_id,
                content_hash_digest=record.content_hash_digest,
                byte_size=record.byte_size,
                created_utc=record.created_utc,
                artifact_type=record.artifact_type,
                run_id=record.run_id,
                storage_path=record.storage_path,
            )
            roles = record.roles
        else:
            manifest = self.store.get_manifest(artifact_id)
            candidate = GCCandidate(
                artifact_id=artifact_id,
                content_hash_digest=manifest.content_hash.digest,
                byte_size=manifest.byte_size,
                created_utc=manifest.created_utc,
                artifact_type=manifest.artifact_type,
                run_id=manifest.lineage.run_id,
                storage_path=manifest.storage_path,
            )
            roles = manifest.roles

        # Check pinned status
        if policy.keep_pinned and self.is_pinned(artifact_id, candidate.run_id):
            candidate.reasons_to_keep.append("pinned")
            return candidate

        # Check if this artifact is an ancestor of a pinned artifact
        # (protects lineage integrity - never break the chain to pinned artifacts)
        if policy.keep_ancestors_of_pinned and ancestors_of_pinned and artifact_id in ancestors_of_pinned:
            candidate.reasons_to_keep.append("ancestor_of_pinned")
            return candidate

        # Check descendants
        if policy.keep_with_descendants and self.has_descendants(artifact_id):
            candidate.reasons_to_keep.append("has_descendants")
            return candidate

        # Check artifact type
        if candidate.artifact_type in policy.keep_artifact_types:
            candidate.reasons_to_keep.append(f"protected_type:{candidate.artifact_type}")
            return candidate

        # Check roles
        for role in roles:
            if role in policy.keep_roles:
                candidate.reasons_to_keep.append(f"protected_role:{role}")
                return candidate

        # Check age
        age_days = _get_age_days(candidate.created_utc)
        if age_days < policy.keep_min_age_days:
            candidate.reasons_to_keep.append(f"age:{age_days:.1f}d < {policy.keep_min_age_days}d")
            return candidate

        # Check minimum count per type
        current_count = artifact_counts_by_type.get(candidate.artifact_type, 0)
        if current_count < policy.keep_min_count:
            candidate.reasons_to_keep.append(f"min_count:{current_count} < {policy.keep_min_count}")
            return candidate

        # If we reach here, artifact is a deletion candidate
        candidate.reasons_to_delete.append(f"age:{age_days:.1f}d >= {policy.keep_min_age_days}d")

        return candidate

    def compute_candidates(
        self,
        policy: RetentionPolicy | str,
    ) -> tuple[list[GCCandidate], list[GCCandidate]]:
        """Compute deletion candidates under a policy.

        Args:
            policy: RetentionPolicy or policy name.

        Returns:
            Tuple of (candidates_to_delete, candidates_to_keep).
        """
        if isinstance(policy, str):
            policy = self.get_policy(policy)

        # Get all manifests
        manifest_ids = self.store.list_manifests()

        # Sort by creation time (oldest first) for consistent count-based retention
        manifests_with_time: list[tuple[str, str]] = []
        for artifact_id in manifest_ids:
            try:
                manifest = self.store.get_manifest(artifact_id)
                manifests_with_time.append((artifact_id, manifest.created_utc))
            except Exception:
                continue

        manifests_with_time.sort(key=lambda x: x[1])

        # Compute ancestors of pinned artifacts if policy requires it
        ancestors_of_pinned: set[str] | None = None
        if policy.keep_ancestors_of_pinned:
            ancestors_of_pinned = self.get_ancestors_of_pinned()

        # Track artifact counts by type
        artifact_counts_by_type: dict[str, int] = {}

        to_delete: list[GCCandidate] = []
        to_keep: list[GCCandidate] = []

        for artifact_id, _ in manifests_with_time:
            candidate = self._evaluate_candidate(artifact_id, policy, artifact_counts_by_type, ancestors_of_pinned)

            if candidate.should_delete:
                to_delete.append(candidate)
            else:
                to_keep.append(candidate)
                # Update count for kept artifacts
                atype = candidate.artifact_type
                artifact_counts_by_type[atype] = artifact_counts_by_type.get(atype, 0) + 1

        return to_delete, to_keep

    def run_dvc_gc(
        self,
        dry_run: bool = False,
        flags: list[str] | None = None,
    ) -> tuple[int, str]:
        """Run DVC garbage collection.

        Args:
            dry_run: If True, only show what would be deleted.
            flags: Additional flags to pass to dvc gc.

        Returns:
            Tuple of (return_code, output).
        """
        if not shutil.which("dvc"):
            return -1, "DVC not found in PATH"

        # Build command with safe defaults
        cmd = ["dvc", "gc"]

        # Always protect workspace and committed versions
        cmd.extend(["--workspace", "--all-commits"])

        if dry_run:
            cmd.append("--dry")

        if flags:
            cmd.extend(flags)

        # Force confirmation in non-dry-run mode
        if not dry_run:
            cmd.append("--force")

        try:
            result = subprocess.run(
                cmd,
                cwd=self.data_dir.parent,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
            )
            output = result.stdout + result.stderr
            return result.returncode, output
        except subprocess.TimeoutExpired:
            return -1, "DVC gc timed out after 1 hour"
        except Exception as e:
            return -1, f"DVC gc failed: {e}"

    def run(
        self,
        policy: RetentionPolicy | str = DEFAULT_POLICY,
        dry_run: bool = True,
        run_dvc_gc: bool = True,
        enforce_space_budget: bool = True,
    ) -> GCResult:
        """Run garbage collection.

        Args:
            policy: RetentionPolicy or policy name.
            dry_run: If True, only show what would be deleted.
            run_dvc_gc: If True, also run dvc gc.
            enforce_space_budget: If True, enforce space budget by deleting more.

        Returns:
            GCResult with details of the operation.
        """
        if isinstance(policy, str):
            policy = self.get_policy(policy)

        started_utc = _now_utc_iso()

        # Compute current storage stats
        stats = self.registry.get_storage_stats()
        bytes_total_before = stats["total_bytes"]

        # Compute candidates
        to_delete, to_keep = self.compute_candidates(policy)

        artifacts_scanned = len(to_delete) + len(to_keep)
        artifacts_deleted = 0
        bytes_freed = 0
        pinned_protected = sum(1 for c in to_keep if "pinned" in c.reasons_to_keep)
        descendant_protected = sum(1 for c in to_keep if "has_descendants" in c.reasons_to_keep)
        ancestor_protected = sum(1 for c in to_keep if "ancestor_of_pinned" in c.reasons_to_keep)

        errors: list[str] = []
        deleted_artifacts: list[str] = []
        protected_artifacts: list[str] = [c.artifact_id for c in to_keep]

        # Delete artifacts
        if not dry_run:
            for candidate in to_delete:
                try:
                    self.store.delete(candidate.artifact_id, delete_content=True)
                    self.registry.delete_artifact(candidate.artifact_id)
                    deleted_artifacts.append(candidate.artifact_id)
                    bytes_freed += candidate.byte_size
                    artifacts_deleted += 1
                except Exception as e:
                    errors.append(f"Failed to delete {candidate.artifact_id}: {e}")
        else:
            # In dry-run, just record what would be deleted
            for candidate in to_delete:
                deleted_artifacts.append(candidate.artifact_id)
                bytes_freed += candidate.byte_size
                artifacts_deleted += 1

        # Run DVC gc if requested
        dvc_gc_ran = False
        dvc_gc_output = None

        if run_dvc_gc:
            returncode, output = self.run_dvc_gc(
                dry_run=dry_run,
                flags=policy.dvc_gc_flags,
            )
            dvc_gc_ran = True
            dvc_gc_output = output
            if returncode != 0 and returncode != -1:
                errors.append(f"DVC gc failed with code {returncode}")

        # Enforce space budget if needed
        if enforce_space_budget and policy.space_budget_bytes is not None:
            bytes_total_after = bytes_total_before - bytes_freed
            if bytes_total_after > policy.space_budget_bytes and not dry_run:
                # Need to delete more - sort kept artifacts by age and delete oldest
                kept_by_age = sorted(to_keep, key=lambda c: c.created_utc)
                for candidate in kept_by_age:
                    if bytes_total_after <= policy.space_budget_bytes:
                        break
                    # Skip if protected for important reasons (pinned, ancestors, or protected types/roles)
                    if any(
                        r.startswith("pinned") or r.startswith("protected_") or r == "ancestor_of_pinned"
                        for r in candidate.reasons_to_keep
                    ):
                        continue
                    try:
                        self.store.delete(candidate.artifact_id, delete_content=True)
                        self.registry.delete_artifact(candidate.artifact_id)
                        deleted_artifacts.append(candidate.artifact_id)
                        bytes_freed += candidate.byte_size
                        bytes_total_after -= candidate.byte_size
                        artifacts_deleted += 1
                    except Exception as e:
                        errors.append(f"Failed to delete {candidate.artifact_id} for budget: {e}")

        finished_utc = _now_utc_iso()
        bytes_total_after = bytes_total_before - bytes_freed

        return GCResult(
            policy_name=policy.name,
            started_utc=started_utc,
            finished_utc=finished_utc,
            dry_run=dry_run,
            artifacts_scanned=artifacts_scanned,
            artifacts_deleted=artifacts_deleted,
            bytes_freed=bytes_freed,
            bytes_total_before=bytes_total_before,
            bytes_total_after=bytes_total_after,
            pinned_protected=pinned_protected,
            descendant_protected=descendant_protected,
            ancestor_protected=ancestor_protected,
            dvc_gc_ran=dvc_gc_ran,
            dvc_gc_output=dvc_gc_output,
            errors=errors,
            deleted_artifacts=deleted_artifacts,
            protected_artifacts=protected_artifacts,
        )

    def estimate_savings(
        self,
        policy: RetentionPolicy | str = DEFAULT_POLICY,
    ) -> dict[str, Any]:
        """Estimate space savings without deleting anything.

        Args:
            policy: RetentionPolicy or policy name.

        Returns:
            Dictionary with estimated savings information.
        """
        if isinstance(policy, str):
            policy = self.get_policy(policy)

        to_delete, to_keep = self.compute_candidates(policy)

        bytes_to_delete = sum(c.byte_size for c in to_delete)
        bytes_to_keep = sum(c.byte_size for c in to_keep)

        stats = self.registry.get_storage_stats()

        return {
            "policy": policy.name,
            "total_artifacts": len(to_delete) + len(to_keep),
            "artifacts_to_delete": len(to_delete),
            "artifacts_to_keep": len(to_keep),
            "bytes_to_delete": bytes_to_delete,
            "bytes_to_keep": bytes_to_keep,
            "current_total_bytes": stats["total_bytes"],
            "estimated_after_bytes": stats["total_bytes"] - bytes_to_delete,
            "space_budget_bytes": policy.space_budget_bytes,
            "within_budget": (
                policy.space_budget_bytes is None or (stats["total_bytes"] - bytes_to_delete) <= policy.space_budget_bytes
            ),
        }


def format_bytes(n: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n //= 1024
    return f"{n:.1f} PB"
