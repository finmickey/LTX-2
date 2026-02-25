"""Generate a self-contained HTML grid of training sample videos with rewards."""

from __future__ import annotations


def generate_training_samples_html(
    epoch: int,
    global_step: int,
    prompt_sections: list[dict],
) -> str:
    """Build a self-contained HTML page showing training samples.

    Args:
        epoch: Current epoch number.
        global_step: Current optimizer step.
        prompt_sections: List of dicts, one per prompt. Each dict has:
            - prompt_text: str
            - samples: list of dicts with keys:
                - filename: str (relative path to .mp4)
                - individual_rewards: dict[str, float]
                - total_reward: float
                - preference: list[float] | None  (only in pareto mode)
            Samples should already be sorted by total_reward descending.
            - reward_names: list[str]
            - has_preferences: bool

    Returns:
        Complete HTML string.
    """
    has_preferences = prompt_sections[0]["has_preferences"] if prompt_sections else False
    reward_names = prompt_sections[0]["reward_names"] if prompt_sections else []

    # Build header row
    header_cols = ["#", "Video", "Prompt"]
    if has_preferences:
        header_cols.append("Preference")
    header_cols.extend(reward_names)
    header_cols.append("Total")

    header_html = "".join(f"<th>{c}</th>" for c in header_cols)

    sections_html = []
    for sec in prompt_sections:
        prompt_text = sec["prompt_text"]
        samples = sec["samples"]
        n = len(samples)

        rows = []
        for rank_idx, s in enumerate(samples):
            # Green (best) to red (worst) gradient based on rank
            if n > 1:
                t = rank_idx / (n - 1)
            else:
                t = 0.0
            r = int(60 + t * 195)
            g = int(220 - t * 180)
            bg = f"rgba({r}, {g}, 60, 0.18)"

            cells = [
                f'<td>{rank_idx + 1}</td>',
                f'<td><video src="{s["filename"]}" width="192" autoplay loop muted playsinline></video></td>',
                f'<td class="prompt">{_esc(prompt_text[:120])}</td>',
            ]
            if has_preferences:
                pref = s.get("preference")
                if pref is not None:
                    pref_str = ", ".join(f"{v * 100:.0f}%" for v in pref)
                    cells.append(f"<td>{pref_str}</td>")
                else:
                    cells.append("<td>-</td>")
            for rname in reward_names:
                val = s["individual_rewards"].get(rname, 0.0)
                cells.append(f"<td>{val:.4f}</td>")
            cells.append(f'<td><b>{s["total_reward"]:.4f}</b></td>')

            row_html = "".join(cells)
            rows.append(f'<tr style="background:{bg}">{row_html}</tr>')

        table_rows = "\n".join(rows)
        sections_html.append(
            f'<h2>{_esc(prompt_text[:200])}</h2>\n'
            f"<table>\n<tr>{header_html}</tr>\n{table_rows}\n</table>"
        )

    body = "\n<hr>\n".join(sections_html)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Training Samples — Epoch {epoch}, Step {global_step}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 20px; background: #1a1a2e; color: #e0e0e0; }}
h1 {{ color: #fff; }}
h2 {{ color: #ccc; font-size: 14px; margin-top: 32px; word-break: break-word; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; }}
th {{ background: #16213e; padding: 8px 12px; text-align: left; font-size: 13px; }}
td {{ padding: 6px 12px; font-size: 13px; vertical-align: middle; border-bottom: 1px solid #333; }}
td.prompt {{ max-width: 220px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
video {{ border-radius: 4px; }}
hr {{ border: none; border-top: 1px solid #444; margin: 32px 0; }}
</style>
</head>
<body>
<h1>Training Samples — Epoch {epoch}, Step {global_step}</h1>
{body}
</body>
</html>"""


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
