"""Pre-KiCad connectivity oracle using union-find (Section 13.2.5).

This module implements a pre-KiCad connectivity check that verifies the SIG net
forms a connected path from the left port to the right port before invoking
KiCad DRC. This catches fundamental topology errors early without relying on
external tooling.

Connectivity is determined using union-find with geometry adjacency rules:
- Segment endpoints match via/pad centers
- Endpoints within via/pad radius are considered connected

For M1 compliance, this module:
1. Treats each conductive object (segments, vias, pads) as nodes
2. Uses geometric adjacency to establish connections
3. Verifies the SIG net forms one connected component from left to right port

REQ-M1-008: Tier 2 analytic spatial constraints
CP-2.3: Topology verification (pre-KiCad) per ECO-M1-ALIGN-0001 Section 13.2.5
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

from .tiers import ConstraintResult, ConstraintTier, TierChecker, _bool_constraint

if TYPE_CHECKING:
    from typing import Any

    from ..geom.layout import LayoutPlan


class NodeType(Enum):
    """Types of conductive nodes in the connectivity graph."""

    PORT_PAD = auto()  # Connector signal pad
    SEGMENT_START = auto()  # Trace segment start point
    SEGMENT_END = auto()  # Trace segment end point
    VIA_CENTER = auto()  # Via center (discontinuity)


@dataclass(frozen=True, slots=True)
class ConnectivityNode:
    """A node in the connectivity graph.

    Represents a point in the layout where electrical connections can occur.

    Attributes:
        x_nm: X coordinate in nanometers
        y_nm: Y coordinate in nanometers
        radius_nm: Connection radius (pad/via radius) for adjacency checks
        node_type: Type of node (pad, segment endpoint, via)
        net_name: Net name for this node (e.g., "SIG")
        label: Human-readable label for debugging
    """

    x_nm: int
    y_nm: int
    radius_nm: int
    node_type: NodeType
    net_name: str
    label: str


@dataclass
class UnionFind:
    """Union-find data structure for connectivity tracking.

    Implements path compression and union by rank for efficient
    connected component tracking.
    """

    parent: dict[int, int] = field(default_factory=dict)
    rank: dict[int, int] = field(default_factory=dict)

    def find(self, x: int) -> int:
        """Find the root of a node with path compression."""
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        """Union two nodes by rank."""
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return

        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

    def connected(self, x: int, y: int) -> bool:
        """Check if two nodes are in the same connected component."""
        return self.find(x) == self.find(y)


def _distance_squared(x1: int, y1: int, x2: int, y2: int) -> int:
    """Compute squared Euclidean distance between two points."""
    dx = x2 - x1
    dy = y2 - y1
    return dx * dx + dy * dy


def _are_adjacent(node_a: ConnectivityNode, node_b: ConnectivityNode) -> bool:
    """Determine if two nodes are geometrically adjacent.

    Two nodes are adjacent if:
    1. They are on the same net, AND
    2. Their centers match exactly, OR
    3. One node's center is within the other's radius, OR
    4. The distance between centers is less than the sum of radii

    Args:
        node_a: First connectivity node
        node_b: Second connectivity node

    Returns:
        True if nodes are adjacent, False otherwise
    """
    # Must be on the same net
    if node_a.net_name != node_b.net_name:
        return False

    dist_sq = _distance_squared(node_a.x_nm, node_a.y_nm, node_b.x_nm, node_b.y_nm)

    # Exact match
    if dist_sq == 0:
        return True

    # Check if within combined radius (with tolerance for integer rounding)
    combined_radius = node_a.radius_nm + node_b.radius_nm
    combined_radius_sq = combined_radius * combined_radius

    return dist_sq <= combined_radius_sq


@dataclass
class ConnectivityOracle:
    """Connectivity oracle for pre-KiCad topology verification.

    Uses union-find to track connected components in the layout
    and verify that the SIG net forms a single path from left to right port.

    Attributes:
        nodes: List of connectivity nodes extracted from the layout
        uf: Union-find structure for connectivity tracking
        node_indices: Mapping from node labels to indices
    """

    nodes: list[ConnectivityNode] = field(default_factory=list)
    uf: UnionFind = field(default_factory=UnionFind)
    node_indices: dict[str, int] = field(default_factory=dict)

    def add_node(self, node: ConnectivityNode) -> int:
        """Add a node to the connectivity graph.

        Args:
            node: The connectivity node to add

        Returns:
            Index of the added node
        """
        index = len(self.nodes)
        self.nodes.append(node)
        self.node_indices[node.label] = index
        return index

    def build_adjacencies(self) -> None:
        """Build adjacency relationships using union-find.

        Iterates over all pairs of nodes and unions those that are
        geometrically adjacent. Uses O(n²) algorithm which is acceptable
        for the small number of nodes in coupon layouts.
        """
        n = len(self.nodes)
        for i in range(n):
            for j in range(i + 1, n):
                if _are_adjacent(self.nodes[i], self.nodes[j]):
                    self.uf.union(i, j)

    def is_connected(self, label_a: str, label_b: str) -> bool:
        """Check if two nodes are in the same connected component.

        Args:
            label_a: Label of first node
            label_b: Label of second node

        Returns:
            True if connected, False otherwise
        """
        idx_a = self.node_indices.get(label_a)
        idx_b = self.node_indices.get(label_b)

        if idx_a is None or idx_b is None:
            return False

        return self.uf.connected(idx_a, idx_b)

    def get_connected_component(self, label: str) -> list[str]:
        """Get all node labels in the same connected component.

        Args:
            label: Label of node to start from

        Returns:
            List of labels in the same component
        """
        idx = self.node_indices.get(label)
        if idx is None:
            return []

        root = self.uf.find(idx)
        return [node.label for i, node in enumerate(self.nodes) if self.uf.find(i) == root]

    def validate_sig_connectivity(self) -> tuple[bool, str]:
        """Validate that the SIG net is fully connected.

        Checks that there is exactly one connected component containing
        all SIG nodes, and that both left and right port pads are connected.

        Returns:
            Tuple of (is_valid, error_message)
        """
        sig_nodes = [i for i, node in enumerate(self.nodes) if node.net_name == "SIG"]

        if len(sig_nodes) < 2:
            return False, "SIG net has fewer than 2 nodes"

        # Check all SIG nodes are in the same component
        root = self.uf.find(sig_nodes[0])
        for idx in sig_nodes[1:]:
            if self.uf.find(idx) != root:
                node = self.nodes[idx]
                first_node = self.nodes[sig_nodes[0]]
                return (
                    False,
                    f"SIG net is not fully connected: '{node.label}' is disconnected from '{first_node.label}'",
                )

        return True, ""


def build_oracle_from_layout(layout: LayoutPlan, via_radius_nm: int = 0) -> ConnectivityOracle:
    """Build a connectivity oracle from a LayoutPlan.

    Extracts connectivity nodes from the layout plan:
    - Port signal pads (left and right)
    - Segment endpoints
    - Via center (if discontinuity present)

    The key insight is that a trace segment's start and end are implicitly
    connected by the segment itself (a conducting path). We model this by
    adding explicit union operations between segment endpoints after building
    geometric adjacencies.

    Args:
        layout: The LayoutPlan to analyze
        via_radius_nm: Radius of the signal via pad for adjacency (0 = point match only)

    Returns:
        ConnectivityOracle populated with nodes and adjacencies built
    """
    oracle = ConnectivityOracle()

    # Add left port signal pad
    oracle.add_node(
        ConnectivityNode(
            x_nm=layout.left_port.signal_pad_x_nm,
            y_nm=layout.left_port.signal_pad_y_nm,
            radius_nm=0,  # Exact match required at pad center
            node_type=NodeType.PORT_PAD,
            net_name="SIG",
            label="left_port_pad",
        )
    )

    # Add right port signal pad
    oracle.add_node(
        ConnectivityNode(
            x_nm=layout.right_port.signal_pad_x_nm,
            y_nm=layout.right_port.signal_pad_y_nm,
            radius_nm=0,  # Exact match required at pad center
            node_type=NodeType.PORT_PAD,
            net_name="SIG",
            label="right_port_pad",
        )
    )

    # Track segment endpoint pairs for implicit connectivity
    segment_pairs: list[tuple[str, str]] = []

    # Add segment endpoints
    for seg in layout.segments:
        if seg.net_name != "SIG":
            continue

        start_label = f"seg_{seg.label}_start"
        end_label = f"seg_{seg.label}_end"

        oracle.add_node(
            ConnectivityNode(
                x_nm=seg.x_start_nm,
                y_nm=seg.y_nm,
                radius_nm=seg.width_nm // 2,  # Half trace width for adjacency
                node_type=NodeType.SEGMENT_START,
                net_name="SIG",
                label=start_label,
            )
        )

        oracle.add_node(
            ConnectivityNode(
                x_nm=seg.x_end_nm,
                y_nm=seg.y_nm,
                radius_nm=seg.width_nm // 2,  # Half trace width for adjacency
                node_type=NodeType.SEGMENT_END,
                net_name="SIG",
                label=end_label,
            )
        )

        # Track this pair for implicit connectivity
        segment_pairs.append((start_label, end_label))

    # Add via center if discontinuity present
    if layout.x_disc_nm is not None:
        oracle.add_node(
            ConnectivityNode(
                x_nm=layout.x_disc_nm,
                y_nm=layout.y_centerline_nm,
                radius_nm=via_radius_nm,
                node_type=NodeType.VIA_CENTER,
                net_name="SIG",
                label="via_center",
            )
        )

    # Build geometric adjacencies
    oracle.build_adjacencies()

    # Connect segment endpoints implicitly (a segment is a conducting path)
    # This is the key step: the segment itself provides connectivity
    for start_label, end_label in segment_pairs:
        start_idx = oracle.node_indices.get(start_label)
        end_idx = oracle.node_indices.get(end_label)
        if start_idx is not None and end_idx is not None:
            oracle.uf.union(start_idx, end_idx)

    return oracle


def check_layout_connectivity(
    layout: LayoutPlan,
    via_radius_nm: int = 0,
) -> tuple[bool, list[str]]:
    """Check connectivity of a LayoutPlan.

    Verifies that:
    1. The SIG net forms one connected component
    2. Both left and right port pads are connected

    Args:
        layout: The LayoutPlan to check
        via_radius_nm: Radius of signal via for adjacency checks

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors: list[str] = []

    oracle = build_oracle_from_layout(layout, via_radius_nm)

    # Check basic layout connectivity (LayoutPlan's own method)
    layout_errors = layout.validate_connectivity()
    errors.extend(layout_errors)

    # Check SIG net connectivity via union-find
    is_connected, sig_error = oracle.validate_sig_connectivity()
    if not is_connected:
        errors.append(sig_error)

    # Check specific port-to-port connectivity
    if not oracle.is_connected("left_port_pad", "right_port_pad"):
        errors.append("Left port pad is not connected to right port pad")

    return len(errors) == 0, errors


class ConnectivityChecker(TierChecker):
    """Tier 2 constraint checker for pre-KiCad connectivity verification.

    Implements Section 13.2.5 of the ECO: verify that the SIG net forms
    a connected path from left to right port using union-find with
    geometry adjacency rules.
    """

    @property
    def tier(self) -> ConstraintTier:
        return "T2"

    def check(
        self,
        spec: Any,
        fab_limits: dict[str, int],
        resolved: Any | None = None,
    ) -> list[ConstraintResult]:
        """Check connectivity constraints.

        Args:
            spec: CouponSpec being validated
            fab_limits: Dictionary of fab capability limits
            resolved: Optional ResolvedDesign with layout_plan

        Returns:
            List of constraint results
        """
        results: list[ConstraintResult] = []

        # Get layout plan from resolved design
        layout = None
        if resolved is not None:
            layout = getattr(resolved, "layout_plan", None)
            if layout is None:
                # Try the private attribute
                layout = getattr(resolved, "_layout_plan", None)

        if layout is None:
            # Cannot check connectivity without layout plan
            results.append(
                _bool_constraint(
                    "T2_CONNECTIVITY_LAYOUT_AVAILABLE",
                    "Layout plan must be available for connectivity check",
                    tier="T2",
                    condition=False,
                    reason="ResolvedDesign does not have a layout_plan",
                )
            )
            return results

        # Get via radius from spec if discontinuity is present
        via_radius_nm = 0
        if spec.discontinuity is not None:
            via_radius_nm = int(spec.discontinuity.signal_via.pad_diameter_nm) // 2

        # Run connectivity check
        is_valid, errors = check_layout_connectivity(layout, via_radius_nm)

        # Add result for overall connectivity
        results.append(
            _bool_constraint(
                "T2_SIG_NET_CONNECTED",
                "SIG net must form a connected path from left to right port",
                tier="T2",
                condition=is_valid,
                reason="; ".join(errors) if errors else "",
            )
        )

        # Add specific segment connectivity results
        segment_errors = layout.validate_connectivity()
        results.append(
            _bool_constraint(
                "T2_SEGMENTS_CONNECTED",
                "All trace segments must be connected without gaps",
                tier="T2",
                condition=len(segment_errors) == 0,
                reason="; ".join(segment_errors) if segment_errors else "",
            )
        )

        # Add port connectivity result
        oracle = build_oracle_from_layout(layout, via_radius_nm)
        port_connected = oracle.is_connected("left_port_pad", "right_port_pad")
        results.append(
            _bool_constraint(
                "T2_PORTS_CONNECTED",
                "Left and right ports must be connected via SIG net",
                tier="T2",
                condition=port_connected,
                reason="Ports are not in the same connected component" if not port_connected else "",
            )
        )

        return results


# Public API
__all__ = [
    "ConnectivityChecker",
    "ConnectivityNode",
    "ConnectivityOracle",
    "NodeType",
    "UnionFind",
    "build_oracle_from_layout",
    "check_layout_connectivity",
]
