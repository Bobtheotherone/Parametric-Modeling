"""Tests for pre-KiCad connectivity oracle (CP-2.5, Section 13.2.5).

This module tests the union-find based connectivity oracle that verifies
the SIG net forms a connected path from left to right port.

REQ-M1-008: Tier 2 analytic spatial constraints
CP-2.3: Topology verification (pre-KiCad)
"""

from __future__ import annotations

import pytest

from formula_foundry.coupongen.constraints.connectivity import (
    ConnectivityChecker,
    ConnectivityNode,
    ConnectivityOracle,
    NodeType,
    UnionFind,
    build_oracle_from_layout,
    check_layout_connectivity,
)
from formula_foundry.coupongen.geom.layout import (
    LayoutPlan,
    PortPlan,
    SegmentPlan,
    create_f0_layout_plan,
    create_f1_layout_plan,
)
from formula_foundry.coupongen.geom.primitives import OriginMode


class TestUnionFind:
    """Tests for the union-find data structure."""

    def test_find_self(self) -> None:
        """Newly added element should be its own root."""
        uf = UnionFind()
        assert uf.find(0) == 0
        assert uf.find(5) == 5

    def test_union_connects_elements(self) -> None:
        """Union should connect two elements."""
        uf = UnionFind()
        uf.union(0, 1)
        assert uf.connected(0, 1)

    def test_transitive_connectivity(self) -> None:
        """Connected via intermediate elements."""
        uf = UnionFind()
        uf.union(0, 1)
        uf.union(1, 2)
        assert uf.connected(0, 2)

    def test_separate_components(self) -> None:
        """Unconnected elements should not be connected."""
        uf = UnionFind()
        uf.union(0, 1)
        uf.union(2, 3)
        assert not uf.connected(0, 2)
        assert not uf.connected(1, 3)

    def test_path_compression(self) -> None:
        """Path compression should flatten tree."""
        uf = UnionFind()
        # Create chain: 0 -> 1 -> 2 -> 3
        for i in range(3):
            uf.union(i, i + 1)

        # Find should compress path
        root = uf.find(0)
        assert uf.parent[0] == root


class TestConnectivityNode:
    """Tests for ConnectivityNode."""

    def test_node_creation(self) -> None:
        """Node should store all attributes."""
        node = ConnectivityNode(
            x_nm=1_000_000,
            y_nm=500_000,
            radius_nm=150_000,
            node_type=NodeType.PORT_PAD,
            net_name="SIG",
            label="test_node",
        )
        assert node.x_nm == 1_000_000
        assert node.y_nm == 500_000
        assert node.radius_nm == 150_000
        assert node.node_type == NodeType.PORT_PAD
        assert node.net_name == "SIG"
        assert node.label == "test_node"

    def test_node_is_immutable(self) -> None:
        """Node should be immutable."""
        node = ConnectivityNode(
            x_nm=1_000_000,
            y_nm=0,
            radius_nm=0,
            node_type=NodeType.SEGMENT_START,
            net_name="SIG",
            label="test",
        )
        with pytest.raises(AttributeError):
            node.x_nm = 2_000_000  # type: ignore[misc]


class TestConnectivityOracle:
    """Tests for ConnectivityOracle."""

    def test_add_node(self) -> None:
        """Adding nodes should increment indices."""
        oracle = ConnectivityOracle()
        idx1 = oracle.add_node(
            ConnectivityNode(
                x_nm=0, y_nm=0, radius_nm=0, node_type=NodeType.PORT_PAD, net_name="SIG", label="a"
            )
        )
        idx2 = oracle.add_node(
            ConnectivityNode(
                x_nm=1_000_000, y_nm=0, radius_nm=0, node_type=NodeType.PORT_PAD, net_name="SIG", label="b"
            )
        )
        assert idx1 == 0
        assert idx2 == 1
        assert len(oracle.nodes) == 2

    def test_exact_position_adjacency(self) -> None:
        """Nodes at exact same position should be adjacent."""
        oracle = ConnectivityOracle()
        oracle.add_node(
            ConnectivityNode(
                x_nm=5_000_000,
                y_nm=0,
                radius_nm=0,
                node_type=NodeType.PORT_PAD,
                net_name="SIG",
                label="a",
            )
        )
        oracle.add_node(
            ConnectivityNode(
                x_nm=5_000_000,
                y_nm=0,
                radius_nm=0,
                node_type=NodeType.SEGMENT_START,
                net_name="SIG",
                label="b",
            )
        )
        oracle.build_adjacencies()
        assert oracle.is_connected("a", "b")

    def test_within_radius_adjacency(self) -> None:
        """Nodes within combined radius should be adjacent."""
        oracle = ConnectivityOracle()
        oracle.add_node(
            ConnectivityNode(
                x_nm=0,
                y_nm=0,
                radius_nm=150_000,
                node_type=NodeType.VIA_CENTER,
                net_name="SIG",
                label="via",
            )
        )
        oracle.add_node(
            ConnectivityNode(
                x_nm=100_000,  # 100um away, within 150um + 100um radius
                y_nm=0,
                radius_nm=100_000,
                node_type=NodeType.SEGMENT_END,
                net_name="SIG",
                label="seg_end",
            )
        )
        oracle.build_adjacencies()
        assert oracle.is_connected("via", "seg_end")

    def test_different_nets_not_adjacent(self) -> None:
        """Nodes on different nets should not be adjacent."""
        oracle = ConnectivityOracle()
        oracle.add_node(
            ConnectivityNode(
                x_nm=0, y_nm=0, radius_nm=0, node_type=NodeType.PORT_PAD, net_name="SIG", label="sig"
            )
        )
        oracle.add_node(
            ConnectivityNode(
                x_nm=0, y_nm=0, radius_nm=0, node_type=NodeType.PORT_PAD, net_name="GND", label="gnd"
            )
        )
        oracle.build_adjacencies()
        assert not oracle.is_connected("sig", "gnd")

    def test_sig_connectivity_validation_pass(self) -> None:
        """Fully connected SIG net should pass validation."""
        oracle = ConnectivityOracle()
        # Create a simple chain: pad -> segment start -> segment end -> pad
        oracle.add_node(
            ConnectivityNode(
                x_nm=0, y_nm=0, radius_nm=0, node_type=NodeType.PORT_PAD, net_name="SIG", label="left_pad"
            )
        )
        oracle.add_node(
            ConnectivityNode(
                x_nm=0, y_nm=0, radius_nm=150_000, node_type=NodeType.SEGMENT_START, net_name="SIG", label="seg_start"
            )
        )
        oracle.add_node(
            ConnectivityNode(
                x_nm=10_000_000, y_nm=0, radius_nm=150_000, node_type=NodeType.SEGMENT_END, net_name="SIG", label="seg_end"
            )
        )
        oracle.add_node(
            ConnectivityNode(
                x_nm=10_000_000, y_nm=0, radius_nm=0, node_type=NodeType.PORT_PAD, net_name="SIG", label="right_pad"
            )
        )
        oracle.build_adjacencies()

        # Manually connect segment start and end (simulating the implicit connectivity
        # that build_oracle_from_layout provides via the segment itself being a conductor)
        seg_start_idx = oracle.node_indices["seg_start"]
        seg_end_idx = oracle.node_indices["seg_end"]
        oracle.uf.union(seg_start_idx, seg_end_idx)

        is_valid, error = oracle.validate_sig_connectivity()
        assert is_valid, f"Expected valid but got error: {error}"

    def test_sig_connectivity_validation_fail(self) -> None:
        """Disconnected SIG net should fail validation."""
        oracle = ConnectivityOracle()
        oracle.add_node(
            ConnectivityNode(
                x_nm=0, y_nm=0, radius_nm=0, node_type=NodeType.PORT_PAD, net_name="SIG", label="left_pad"
            )
        )
        oracle.add_node(
            ConnectivityNode(
                x_nm=50_000_000,  # Far away, not connected
                y_nm=0,
                radius_nm=0,
                node_type=NodeType.PORT_PAD,
                net_name="SIG",
                label="right_pad",
            )
        )
        oracle.build_adjacencies()

        is_valid, error = oracle.validate_sig_connectivity()
        assert not is_valid
        assert "not fully connected" in error.lower() or "disconnected" in error.lower()

    def test_get_connected_component(self) -> None:
        """Should return all nodes in the same component."""
        oracle = ConnectivityOracle()
        oracle.add_node(
            ConnectivityNode(
                x_nm=0, y_nm=0, radius_nm=100_000, node_type=NodeType.PORT_PAD, net_name="SIG", label="a"
            )
        )
        oracle.add_node(
            ConnectivityNode(
                x_nm=50_000, y_nm=0, radius_nm=100_000, node_type=NodeType.SEGMENT_START, net_name="SIG", label="b"
            )
        )
        oracle.add_node(
            ConnectivityNode(
                x_nm=10_000_000, y_nm=0, radius_nm=0, node_type=NodeType.PORT_PAD, net_name="SIG", label="c"
            )
        )
        oracle.build_adjacencies()

        component_a = oracle.get_connected_component("a")
        assert "a" in component_a
        assert "b" in component_a
        assert "c" not in component_a


class TestLayoutConnectivity:
    """Tests for layout-based connectivity checking."""

    def test_f0_layout_is_connected(self) -> None:
        """F0 (through-line) layout should be connected."""
        layout = create_f0_layout_plan(
            board_length_nm=80_000_000,
            board_width_nm=20_000_000,
            board_corner_radius_nm=2_000_000,
            left_port_x_nm=5_000_000,
            right_port_x_nm=75_000_000,
            trace_width_nm=300_000,
            trace_layer="F.Cu",
            footprint="Coupongen_Connectors:SMA_EndLaunch_Generic",
        )

        is_valid, errors = check_layout_connectivity(layout)
        assert is_valid, f"Expected connected but got errors: {errors}"

    def test_f1_layout_is_connected(self) -> None:
        """F1 (via transition) layout should be connected."""
        layout = create_f1_layout_plan(
            board_length_nm=80_000_000,
            board_width_nm=20_000_000,
            board_corner_radius_nm=2_000_000,
            left_port_x_nm=5_000_000,
            right_port_x_nm=75_000_000,
            left_length_nm=30_000_000,
            trace_width_nm=300_000,
            trace_layer="F.Cu",
            footprint="Coupongen_Connectors:SMA_EndLaunch_Generic",
        )

        is_valid, errors = check_layout_connectivity(layout, via_radius_nm=450_000)
        assert is_valid, f"Expected connected but got errors: {errors}"

    def test_build_oracle_extracts_correct_nodes(self) -> None:
        """Oracle should extract all nodes from layout."""
        layout = create_f1_layout_plan(
            board_length_nm=80_000_000,
            board_width_nm=20_000_000,
            board_corner_radius_nm=2_000_000,
            left_port_x_nm=5_000_000,
            right_port_x_nm=75_000_000,
            left_length_nm=30_000_000,
            trace_width_nm=300_000,
            trace_layer="F.Cu",
            footprint="Coupongen_Connectors:SMA_EndLaunch_Generic",
        )

        oracle = build_oracle_from_layout(layout, via_radius_nm=450_000)

        # Should have: 2 port pads + 4 segment endpoints (2 segments * 2 ends) + 1 via
        # = 7 nodes
        assert len(oracle.nodes) == 7

        # Check specific nodes exist
        assert "left_port_pad" in oracle.node_indices
        assert "right_port_pad" in oracle.node_indices
        assert "via_center" in oracle.node_indices
        assert "seg_left_start" in oracle.node_indices
        assert "seg_left_end" in oracle.node_indices
        assert "seg_right_start" in oracle.node_indices
        assert "seg_right_end" in oracle.node_indices


class TestConnectivityChecker:
    """Tests for ConnectivityChecker as Tier 2 constraint."""

    def _make_resolved_with_layout(self, layout: LayoutPlan):
        """Create a mock resolved design with layout plan."""
        class MockResolved:
            def __init__(self, lp: LayoutPlan):
                self._layout_plan = lp

            @property
            def layout_plan(self) -> LayoutPlan:
                return self._layout_plan

        return MockResolved(layout)

    def _make_spec_with_discontinuity(self):
        """Create a mock spec with discontinuity."""
        class MockVia:
            pad_diameter_nm = 900_000

        class MockDiscontinuity:
            signal_via = MockVia()

        class MockSpec:
            discontinuity = MockDiscontinuity()

        return MockSpec()

    def test_checker_tier(self) -> None:
        """Checker should report Tier 2."""
        checker = ConnectivityChecker()
        assert checker.tier == "T2"

    def test_checker_with_connected_layout(self) -> None:
        """Checker should pass for connected layout."""
        layout = create_f1_layout_plan(
            board_length_nm=80_000_000,
            board_width_nm=20_000_000,
            board_corner_radius_nm=2_000_000,
            left_port_x_nm=5_000_000,
            right_port_x_nm=75_000_000,
            left_length_nm=30_000_000,
            trace_width_nm=300_000,
            trace_layer="F.Cu",
            footprint="Coupongen_Connectors:SMA_EndLaunch_Generic",
        )
        resolved = self._make_resolved_with_layout(layout)
        spec = self._make_spec_with_discontinuity()
        fab_limits: dict[str, int] = {}

        checker = ConnectivityChecker()
        results = checker.check(spec, fab_limits, resolved)

        assert len(results) >= 3  # SIG_NET_CONNECTED, SEGMENTS_CONNECTED, PORTS_CONNECTED
        sig_result = next(r for r in results if r.constraint_id == "T2_SIG_NET_CONNECTED")
        assert sig_result.passed

    def test_checker_without_resolved_layout(self) -> None:
        """Checker should fail gracefully without layout plan."""
        checker = ConnectivityChecker()
        results = checker.check(None, {}, None)  # type: ignore[arg-type]

        assert len(results) >= 1
        layout_result = next(r for r in results if r.constraint_id == "T2_CONNECTIVITY_LAYOUT_AVAILABLE")
        assert not layout_result.passed

    def test_checker_results_are_tier2(self) -> None:
        """All results should be Tier 2."""
        layout = create_f0_layout_plan(
            board_length_nm=80_000_000,
            board_width_nm=20_000_000,
            board_corner_radius_nm=2_000_000,
            left_port_x_nm=5_000_000,
            right_port_x_nm=75_000_000,
            trace_width_nm=300_000,
            trace_layer="F.Cu",
            footprint="Coupongen_Connectors:SMA_EndLaunch_Generic",
        )
        resolved = self._make_resolved_with_layout(layout)

        class MockSpec:
            discontinuity = None

        checker = ConnectivityChecker()
        results = checker.check(MockSpec(), {}, resolved)

        for result in results:
            assert result.tier == "T2"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_segment_layout(self) -> None:
        """F0 layout with single segment should be connected."""
        left_port = PortPlan(
            x_ref_nm=5_000_000,
            y_ref_nm=0,
            signal_pad_x_nm=5_000_000,
            signal_pad_y_nm=0,
            footprint="test",
            rotation_mdeg=0,
            side="left",
        )
        right_port = PortPlan(
            x_ref_nm=75_000_000,
            y_ref_nm=0,
            signal_pad_x_nm=75_000_000,
            signal_pad_y_nm=0,
            footprint="test",
            rotation_mdeg=180000,
            side="right",
        )
        segment = SegmentPlan(
            x_start_nm=5_000_000,
            x_end_nm=75_000_000,
            y_nm=0,
            width_nm=300_000,
            layer="F.Cu",
            net_name="SIG",
            label="through",
        )
        layout = LayoutPlan(
            origin_mode=OriginMode.EDGE_L_CENTER,
            board_length_nm=80_000_000,
            board_width_nm=20_000_000,
            board_corner_radius_nm=2_000_000,
            left_port=left_port,
            right_port=right_port,
            segments=(segment,),
            x_disc_nm=None,
            y_centerline_nm=0,
            coupon_family="F0_CAL_THRU_LINE",
        )

        is_valid, errors = check_layout_connectivity(layout)
        assert is_valid, f"Expected valid but got: {errors}"

    def test_zero_length_segment(self) -> None:
        """Zero-length segment (point) should still connect adjacent nodes."""
        oracle = ConnectivityOracle()
        # Segment with zero length (start == end)
        oracle.add_node(
            ConnectivityNode(
                x_nm=1_000_000,
                y_nm=0,
                radius_nm=150_000,
                node_type=NodeType.SEGMENT_START,
                net_name="SIG",
                label="seg_start",
            )
        )
        oracle.add_node(
            ConnectivityNode(
                x_nm=1_000_000,  # Same position
                y_nm=0,
                radius_nm=150_000,
                node_type=NodeType.SEGMENT_END,
                net_name="SIG",
                label="seg_end",
            )
        )
        oracle.build_adjacencies()
        assert oracle.is_connected("seg_start", "seg_end")

    def test_very_large_coordinates(self) -> None:
        """Should handle very large nm coordinates without overflow."""
        oracle = ConnectivityOracle()
        # Use coordinates near 32-bit int limits
        large_x = 2_000_000_000  # 2 meters in nm
        oracle.add_node(
            ConnectivityNode(
                x_nm=large_x,
                y_nm=0,
                radius_nm=0,
                node_type=NodeType.PORT_PAD,
                net_name="SIG",
                label="a",
            )
        )
        oracle.add_node(
            ConnectivityNode(
                x_nm=large_x,
                y_nm=0,
                radius_nm=0,
                node_type=NodeType.SEGMENT_START,
                net_name="SIG",
                label="b",
            )
        )
        oracle.build_adjacencies()
        assert oracle.is_connected("a", "b")

    def test_multiple_connected_components(self) -> None:
        """Should detect multiple disconnected components."""
        oracle = ConnectivityOracle()
        # Component 1
        oracle.add_node(
            ConnectivityNode(
                x_nm=0, y_nm=0, radius_nm=100_000, node_type=NodeType.PORT_PAD, net_name="SIG", label="c1_a"
            )
        )
        oracle.add_node(
            ConnectivityNode(
                x_nm=50_000, y_nm=0, radius_nm=100_000, node_type=NodeType.SEGMENT_START, net_name="SIG", label="c1_b"
            )
        )
        # Component 2 (far away)
        oracle.add_node(
            ConnectivityNode(
                x_nm=100_000_000, y_nm=0, radius_nm=100_000, node_type=NodeType.SEGMENT_END, net_name="SIG", label="c2_a"
            )
        )
        oracle.add_node(
            ConnectivityNode(
                x_nm=100_050_000, y_nm=0, radius_nm=100_000, node_type=NodeType.PORT_PAD, net_name="SIG", label="c2_b"
            )
        )
        oracle.build_adjacencies()

        # Should not be connected across components
        assert not oracle.is_connected("c1_a", "c2_a")
        # Should fail SIG validation
        is_valid, error = oracle.validate_sig_connectivity()
        assert not is_valid
