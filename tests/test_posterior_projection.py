from __future__ import annotations

from pathlib import Path

from pfm.visualization.posterior_figures import write_posterior_figures


def _space(scale: float) -> dict[str, object]:
    return {
        "mean_pred_var": scale,
        "mean_squared_error": scale * 1.1,
        "error_to_var_ratio": 1.1,
        "bins": [
            {
                "mean_pred_var": scale * 0.8,
                "mean_squared_error": scale * 0.9,
                "error_to_var_ratio": 1.125,
            },
            {
                "mean_pred_var": scale * 1.2,
                "mean_squared_error": scale * 1.3,
                "error_to_var_ratio": 1.083,
            },
        ],
        "histogram_bins": [
            {"center_pred_var": scale * 0.8, "fraction": 0.4},
            {"center_pred_var": scale * 1.2, "fraction": 0.6},
        ],
    }


def test_one_posterior_numerical_owner_projects_exactly_six_chart_families(
    tmp_path: Path,
) -> None:
    controlled = {
        "by_t": {
            "0.25": {
                "t": 0.25,
                "velocity": _space(1.0),
                "endpoint": _space(0.5),
                "variance_nll": {
                    "velocity_complete_nll_per_dim": 1.2,
                    "endpoint_complete_nll_per_dim": 0.8,
                },
            },
            "0.75": {
                "t": 0.75,
                "velocity": _space(0.9),
                "endpoint": _space(0.1),
                "variance_nll": {
                    "velocity_complete_nll_per_dim": 1.1,
                    "endpoint_complete_nll_per_dim": 0.2,
                },
            },
        }
    }
    random_time = {
        "variance_calibration": {
            "velocity": _space(1.0),
            "endpoint": _space(0.25),
        },
        "time_calibration": {
            "velocity": {
                "bins": [
                    {"t_mean": 0.25, **_space(1.0)["bins"][0]},
                    {"t_mean": 0.75, **_space(1.0)["bins"][1]},
                ]
            },
            "endpoint": {
                "bins": [
                    {"t_mean": 0.25, **_space(0.25)["bins"][0]},
                    {"t_mean": 0.75, **_space(0.25)["bins"][1]},
                ]
            },
        },
    }

    paths = write_posterior_figures(tmp_path, controlled, random_time)

    assert set(paths) == {
        "controlled_variance_error",
        "controlled_complete_nll",
        "random_variance_calibration",
        "random_error_variance_ratio",
        "random_variance_histogram",
        "random_time_calibration",
    }
    for formats in paths.values():
        assert set(formats) == {"png", "pdf"}
        assert (tmp_path / formats["png"]).is_file()
        assert (tmp_path / formats["pdf"]).is_file()
    assert not list(tmp_path.rglob("*.json"))
    assert not list(tmp_path.rglob("*.csv"))
    assert not list(tmp_path.rglob("*.md"))
