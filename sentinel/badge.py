"""Security score badge generator for Sentinel AI.

Generates SVG badge images showing a project's security score,
designed for embedding in README files.

Usage:
    sentinel badge                     # Generate badge for current project
    sentinel badge --dir /path/to/proj # Generate for specific project
    sentinel badge --output badge.svg  # Save to file
    sentinel badge --style flat        # Badge style: flat, flat-square
"""

from __future__ import annotations

from pathlib import Path


_BADGE_TEMPLATE = """\
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{width}" height="20" role="img" aria-label="security: {value_text}">
  <title>security: {value_text}</title>
  <linearGradient id="s" x2="0" y2="100%">
    <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
    <stop offset="1" stop-opacity=".1"/>
  </linearGradient>
  <clipPath id="r">
    <rect width="{width}" height="20" rx="3" fill="#fff"/>
  </clipPath>
  <g clip-path="url(#r)">
    <rect width="{label_width}" height="20" fill="#555"/>
    <rect x="{label_width}" width="{value_width}" height="20" fill="{color}"/>
    <rect width="{width}" height="20" fill="url(#s)"/>
  </g>
  <g fill="#fff" text-anchor="middle" font-family="Verdana,Geneva,DejaVu Sans,sans-serif" text-rendering="geometricPrecision" font-size="110">
    <text aria-hidden="true" x="{label_center}" y="150" fill="#010101" fill-opacity=".3" transform="scale(.1)" textLength="{label_text_width}">{label}</text>
    <text x="{label_center}" y="140" transform="scale(.1)" fill="#fff" textLength="{label_text_width}">{label}</text>
    <text aria-hidden="true" x="{value_center}" y="150" fill="#010101" fill-opacity=".3" transform="scale(.1)" textLength="{value_text_width}">{value_text}</text>
    <text x="{value_center}" y="140" transform="scale(.1)" fill="#fff" textLength="{value_text_width}">{value_text}</text>
  </g>
</svg>"""


def _score_color(score: int) -> str:
    """Get badge color based on security score."""
    if score >= 90:
        return "#4c1"  # bright green
    elif score >= 80:
        return "#97ca00"  # green
    elif score >= 70:
        return "#a4a61d"  # yellow-green
    elif score >= 60:
        return "#dfb317"  # yellow
    elif score >= 50:
        return "#fe7d37"  # orange
    else:
        return "#e05d44"  # red


def _risk_color(risk: str) -> str:
    """Get badge color based on risk level."""
    return {
        "none": "#4c1",
        "low": "#97ca00",
        "medium": "#dfb317",
        "high": "#fe7d37",
        "critical": "#e05d44",
    }.get(risk, "#9f9f9f")


def generate_badge(
    score: int,
    *,
    label: str = "sentinel security",
    style: str = "flat",
) -> str:
    """Generate an SVG badge for a security score.

    Args:
        score: Security score 0-100.
        label: Left-side label text.
        style: Badge style (currently only 'flat').

    Returns:
        SVG string.
    """
    value_text = f"{score}/100"
    color = _score_color(score)

    # Calculate widths (approximate character width of 6.5px for Verdana 11px)
    char_width = 6.5
    padding = 10
    label_text_width = int(len(label) * char_width * 10)
    value_text_width = int(len(value_text) * char_width * 10)
    label_width = int(len(label) * char_width + padding * 2)
    value_width = int(len(value_text) * char_width + padding * 2)
    width = label_width + value_width

    return _BADGE_TEMPLATE.format(
        width=width,
        label_width=label_width,
        value_width=value_width,
        label_center=int(label_width * 10 / 2),
        value_center=int((label_width + value_width / 2) * 10),
        label_text_width=label_text_width,
        value_text_width=value_text_width,
        label=label,
        value_text=value_text,
        color=color,
    )


def generate_risk_badge(
    risk: str,
    findings: int = 0,
    *,
    label: str = "sentinel risk",
) -> str:
    """Generate an SVG badge for risk level.

    Args:
        risk: Risk level string (none, low, medium, high, critical).
        findings: Number of findings.
        label: Left-side label text.

    Returns:
        SVG string.
    """
    value_text = f"{risk}" if findings == 0 else f"{risk} ({findings})"
    color = _risk_color(risk)

    char_width = 6.5
    padding = 10
    label_text_width = int(len(label) * char_width * 10)
    value_text_width = int(len(value_text) * char_width * 10)
    label_width = int(len(label) * char_width + padding * 2)
    value_width = int(len(value_text) * char_width + padding * 2)
    width = label_width + value_width

    return _BADGE_TEMPLATE.format(
        width=width,
        label_width=label_width,
        value_width=value_width,
        label_center=int(label_width * 10 / 2),
        value_center=int((label_width + value_width / 2) * 10),
        label_text_width=label_text_width,
        value_text_width=value_text_width,
        label=label,
        value_text=value_text,
        color=color,
    )


def generate_badge_for_project(
    project_dir: Path | None = None,
    *,
    output: Path | None = None,
) -> tuple[str, int]:
    """Run project scan and generate a badge SVG.

    Args:
        project_dir: Project directory to scan.
        output: Optional path to write SVG file.

    Returns:
        Tuple of (SVG string, security score).
    """
    from sentinel.project_scanner import scan_project

    report = scan_project(project_dir)
    svg = generate_badge(report.score)

    if output:
        output.write_text(svg, encoding="utf-8")

    return svg, report.score
