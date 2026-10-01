"""Deterministic Text Renderer for Business Explanations (Phase 7D).

Converts a structured BusinessExplanation container into a clean, human-readable
business briefing format without recalculating metrics or inventing prose.
"""

from __future__ import annotations

from commerce_ai.explanations.schemas import BusinessExplanation


def render_explanation_to_text(explanation: BusinessExplanation) -> str:
    """Render a BusinessExplanation into structured, executive text format."""
    lines: list[str] = []

    # 1. Headline
    lines.append(f"HEADLINE: {explanation.headline}")
    lines.append("")

    # 2. Summary
    lines.append("SUMMARY:")
    lines.append(f"  {explanation.summary}")
    lines.append("")

    # 3. Key Findings
    if explanation.key_findings:
        lines.append("KEY FINDINGS:")
        for idx, finding in enumerate(explanation.key_findings, 1):
            prov_tag = f"[{finding.provenance.value}]"
            conf_tag = f"({finding.confidence.value} Confidence)"
            lines.append(f"  {idx}. {finding.statement} {prov_tag} {conf_tag}")
        lines.append("")

    # 4. Comparisons if present
    if explanation.comparisons:
        lines.append("COMPARISONS:")
        for comp in explanation.comparisons:
            m = comp.get("metric", "Metric")
            cur = comp.get("current_value")
            prev = comp.get("comparison_value")
            pct = comp.get("percentage_change")
            pct_str = f" ({pct:+.1f}%)" if pct is not None else ""
            lines.append(f"  - {m}: Current = {cur}, Baseline = {prev}{pct_str}")
        lines.append("")

    # 5. Trends if present
    if explanation.trends:
        lines.append("TRENDS:")
        for tr in explanation.trends:
            lines.append(f"  - {tr.get('metric')}: {tr.get('direction')} from {tr.get('start_value')} ({tr.get('start_date')}) to {tr.get('end_value')} ({tr.get('end_date')})")
        lines.append("")

    # 6. Risks if present
    if explanation.risks:
        lines.append("RISK OBSERVATIONS:")
        for r in explanation.risks:
            lines.append(f"  - {r}")
        lines.append("")

    # 7. Limitations & Governance
    if explanation.limitations:
        lines.append("LIMITATIONS & GOVERNANCE:")
        for lim in explanation.limitations:
            lines.append(f"  - {lim}")
        lines.append("")

    # 8. Data Quality Notes if present
    if explanation.data_quality_notes:
        lines.append("DATA QUALITY NOTES:")
        for dq in explanation.data_quality_notes:
            lines.append(f"  - {dq}")
        lines.append("")

    # 9. Lineage & Metadata
    lines.append("PROVENANCE & AUDIT:")
    lines.append(f"  Explanation ID: {explanation.explanation_id}")
    lines.append(f"  Overall Provenance: {explanation.provenance.value}")
    lines.append(f"  Confidence Tier: {explanation.confidence.value}")
    lines.append(f"  Status: {explanation.status.value}")
    lines.append(f"  Supporting Evidence Count: {len(explanation.supporting_evidence)}")

    return "\n".join(lines)
