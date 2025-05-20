from optuna.visualization import (
    plot_contour,
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
    plot_pareto_front,
    plot_slice,
)

from syftr.configuration import cfg


def plot_accuracy_and_wc_time_w_optuna(study):
    plot_optimization_history(
        study,
        target=lambda t: t.values[0],
        target_name=cfg.plotting.target_accuracy_name,
    ).show()

    plot_optimization_history(
        study,
        target=lambda t: t.values[1],
        target_name=cfg.plotting.target_latency_name,
    ).show()

    plot_parallel_coordinate(
        study,
        target=lambda t: t.values[0],
        target_name=cfg.plotting.target_accuracy_name,
    ).show()

    plot_parallel_coordinate(
        study,
        target=lambda t: t.values[1],
        target_name=cfg.plotting.target_latency_name,
    ).show()

    plot_slice(
        study,
        target=lambda t: t.values[0],
        target_name=cfg.plotting.target_accuracy_name,
    ).show()

    plot_slice(
        study,
        target=lambda t: t.values[1],
        target_name=cfg.plotting.target_latency_name,
    ).show()

    plot_contour(
        study,
        target=lambda t: t.values[0],
        target_name=cfg.plotting.target_accuracy_name,
    ).show()

    plot_contour(
        study,
        target=lambda t: t.values[1],
        target_name=cfg.plotting.target_latency_name,
    ).show()

    plot_param_importances(
        study,
        target=lambda t: t.values[0],
        target_name=cfg.plotting.target_accuracy_name,
    )

    plot_param_importances(
        study,
        target=lambda t: t.values[1],
        target_name=cfg.plotting.target_latency_name,
    )

    plot_pareto_front(
        study=study,
        target_names=[
            cfg.plotting.target_latency_name,
            cfg.plotting.target_accuracy_name,
        ],
        targets=lambda t: (t.values[1], t.values[0]),
    ).show()
