from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def save_trial_summary(
    trial_dir: Path,
    trial_number: int,
    params: Dict[str, Any],
    best_epoch: int,
    best_val_rmse: float,
    val_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
) -> None:
    lines = [
        f"Trial: {trial_number}",
        f"Timestamp (UTC): {datetime.utcnow().isoformat()}",
        "",
        "Best Validation RMSE",
        f"{best_val_rmse:.6f}",
        f"Best Epoch: {best_epoch}",
        "",
        "Parameters",
    ]
    for key in sorted(params.keys()):
        lines.append(f"- {key}: {params[key]}")

    lines.extend(["", "Validation Metrics"])
    for key in sorted(val_metrics.keys()):
        lines.append(f"- {key}: {val_metrics[key]:.6f}")

    lines.extend(["", "Test Metrics"])
    for key in sorted(test_metrics.keys()):
        lines.append(f"- {key}: {test_metrics[key]:.6f}")

    (trial_dir / "summary.txt").write_text("\n".join(lines))
