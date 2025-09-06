from __future__ import annotations

from typing import Dict, Iterable, Tuple, List, Optional
import pygraphviz as pgv
import textwrap


FILL_COLOR = {"entity": "#F4D03F", "relation": "#E74C3C", "attribute": "#3498DB"}
FONT_COLOR = {"entity": "black", "relation": "white", "attribute": "white"}


def _parse_types_from_tuples(tuple_out: Dict) -> Dict[int, str]:
    """Infer a node type from the tuple `content` prefix."""
    id2type: Dict[int, str] = {}
    for t in tuple_out.get("tuples", []):
        content = (t.get("content") or "").lower()
        if content.startswith("entity"):
            id2type[t["id"]] = "entity"
        elif content.startswith("relation"):
            id2type[t["id"]] = "relation"
        elif content.startswith("attribute"):
            id2type[t["id"]] = "attribute"
        else:
            id2type[t["id"]] = "entity"
    return id2type


def _parse_questions(question_out: Dict) -> Dict[int, str]:
    """Map id -> natural language question string."""
    return {
        q["id"]: q.get("question", f"ID {q['id']}")
        for q in question_out.get("questions", [])
    }


def _edges_from_dependencies(
    dep_out: Dict, *, direction: str = "child_to_parent"
) -> List[Tuple[int, int]]:
    """Create edge list from dependency records.

    Args:
        dep_out: {"dependencies": [{"id": child_id, "dependencies": [parent_ids...]}]}
        direction: "child_to_parent"  (default)  or  "parent_to_child"
    """
    edges: List[Tuple[int, int]] = []
    for d in dep_out.get("dependencies", []):
        child = d["id"]
        for parent in d.get("dependencies", []):
            if direction == "child_to_parent":
                edges.append((child, parent))
            else:
                edges.append((parent, child))
    return edges


def _wrap(text: str, width: int) -> str:
    return "\\n".join(textwrap.wrap(text, width=width)) if width and width > 0 else text


def build_graph(
    tuple_out: Dict,
    question_out: Dict,
    dep_out: Dict,
    *,
    direction: str = "child_to_parent",
    rankdir: str = "LR",
    wrap: int = 28,
    node_defaults: Optional[Dict[str, str]] = None,
    graph_defaults: Optional[Dict[str, str]] = None,
) -> pgv.AGraph:
    """Build a pygraphviz AGraph for the DSG DAG.

    Returns the AGraph (you can lay out/draw it yourself if you want)."""
    id2type = _parse_types_from_tuples(tuple_out)
    id2q = _parse_questions(question_out)
    edges = _edges_from_dependencies(dep_out, direction=direction)

    # Defaults
    graph_attr = dict(rankdir=rankdir, splines="spline", nodesep="0.5", ranksep="0.7")
    if graph_defaults:
        graph_attr.update(graph_defaults)

    # Create graph
    G = pgv.AGraph(strict=False, directed=True, **graph_attr)

    # Nodes
    for nid, label in id2q.items():
        ntype = id2type.get(nid, "entity")
        attrs = dict(
            label=_wrap(label, wrap),
            shape="box",
            style="rounded,filled",
            color="black",
            penwidth="2",
            fillcolor=FILL_COLOR[ntype],
            fontcolor=FONT_COLOR[ntype],
            fontsize="12",
            fontname="Helvetica",
            margin="0.12,0.08",
        )
        if node_defaults:
            attrs.update(node_defaults)
        G.add_node(nid, **attrs)

    # Edges (from dependencies)
    for u, v in edges:
        G.add_edge(u, v, arrowsize="0.8", penwidth="1.8")

    return G


def render_dag_pygraphviz(
    tuple_out: Dict,
    question_out: Dict,
    dep_out: Dict,
    out_path: str,
    *,
    direction: str = "child_to_parent",
    rankdir: str = "LR",
    wrap: int = 28,
    node_defaults: Optional[Dict[str, str]] = None,
    graph_defaults: Optional[Dict[str, str]] = None,
    layout_prog: str = "dot",
) -> str:
    """High-level convenience: build, layout, and draw the DAG to `out_path`.

    The output format is inferred from `out_path` extension (e.g., .png, .svg).
    Returns `out_path`.
    """
    G = build_graph(
        tuple_out,
        question_out,
        dep_out,
        direction=direction,
        rankdir=rankdir,
        wrap=wrap,
        node_defaults=node_defaults,
        graph_defaults=graph_defaults,
    )
    G.layout(prog=layout_prog)
    G.draw(out_path)
    return out_path


__all__ = ["build_graph", "render_dag_pygraphviz"]
