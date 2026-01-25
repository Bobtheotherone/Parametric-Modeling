from __future__ import annotations

from collections.abc import Sequence


def _normalize_route(route: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(route)
    if not normalized:
        raise ValueError("smoke route must include at least one agent")
    return normalized


def next_agent_for_route(route: Sequence[str], index: int) -> str:
    """Return the agent at the given route index.

    Raises:
        ValueError: when the route is empty.
        IndexError: when index is outside the route bounds.
    """
    normalized = _normalize_route(route)
    if index < 0 or index >= len(normalized):
        raise IndexError(f"smoke route index {index} out of range for length {len(normalized)}")
    return normalized[index]


def smoke_route_override_reason(
    *,
    requested: str,
    routed: str,
    index: int,
    route_len: int,
) -> str | None:
    """Return a clear reason when the smoke route overrides the requested agent."""
    if requested == routed:
        return None
    return f"smoke route override (index {index + 1}/{route_len}): {requested} -> {routed}"


def resolve_smoke_route(
    *,
    requested: str,
    route: Sequence[str],
    index: int,
) -> tuple[str, str | None]:
    """Resolve the next agent from the smoke route with an override reason."""
    normalized = _normalize_route(route)
    routed = next_agent_for_route(normalized, index)
    reason = smoke_route_override_reason(
        requested=requested,
        routed=routed,
        index=index,
        route_len=len(normalized),
    )
    return routed, reason
