from typing import Literal

from White_Box_Lie_Detection.repeng.activations.probe_preparations import ActivationArrays
from White_Box_Lie_Detection.repeng.probes.base import BaseProbe
from White_Box_Lie_Detection.repeng.probes.contrast_consistent_search import CcsTrainingConfig, train_ccs_probe
from White_Box_Lie_Detection.repeng.probes.difference_in_means import train_dim_probe
from White_Box_Lie_Detection.repeng.probes.linear_artificial_tomography import train_lat_probe
from White_Box_Lie_Detection.repeng.probes.linear_discriminant_analysis import train_lda_probe
from White_Box_Lie_Detection.repeng.probes.logistic_regression import (
    LrConfig,
    train_grouped_lr_probe,
    train_lr_probe,
)
from White_Box_Lie_Detection.repeng.probes.principal_component_analysis import (
    train_grouped_pca_probe,
    train_pca_probe,
)
from White_Box_Lie_Detection.repeng.probes.random import train_random_probe

ProbeMethod = Literal["ccs", "lat", "dim", "lda", "lr", "lr-g", "pca", "pca-g", "rand"]

ALL_PROBES: list[ProbeMethod] = [
    "ccs",
    "lat",
    "dim",
    "lda",
    "lr",
    "lr-g",
    "pca",
    "pca-g",
]
SUPERVISED_PROBES: list[ProbeMethod] = ["dim", "lda", "lr", "lr-g"]
UNSUPERVISED_PROBES: list[ProbeMethod] = list(set(ALL_PROBES) - set(SUPERVISED_PROBES))
GROUPED_PROBES: list[ProbeMethod] = ["ccs", "lr-g", "pca-g"]
UNGROUPED_PROBES: list[ProbeMethod] = list(set(ALL_PROBES) - set(GROUPED_PROBES))


def train_probe(
    probe_method: ProbeMethod, arrays: ActivationArrays
) -> BaseProbe | None:
    if probe_method == "ccs":
        if arrays.groups is None:
            return None
        return train_ccs_probe(
            CcsTrainingConfig(),
            activations=arrays.activations,
            groups=arrays.groups,
            answer_types=arrays.answer_types,
            # N.B.: Technically unsupervised!
            labels=arrays.labels,
        )
    elif probe_method == "lat":
        return train_lat_probe(
            activations=arrays.activations,
            answer_types=arrays.answer_types,
        )
    elif probe_method == "dim":
        return train_dim_probe(
            activations=arrays.activations,
            labels=arrays.labels,
        )
    elif probe_method == "lda":
        return train_lda_probe(
            activations=arrays.activations,
            labels=arrays.labels,
        )
    elif probe_method == "lr":
        return train_lr_probe(
            LrConfig(),
            activations=arrays.activations,
            labels=arrays.labels,
        )
    elif probe_method == "lr-g":
        if arrays.groups is None:
            return None
        return train_grouped_lr_probe(
            LrConfig(),
            activations=arrays.activations,
            groups=arrays.groups,
            labels=arrays.labels,
        )
    elif probe_method == "pca":
        return train_pca_probe(
            activations=arrays.activations,
            answer_types=arrays.answer_types,
        )
    elif probe_method == "pca-g":
        if arrays.groups is None:
            return None
        return train_grouped_pca_probe(
            activations=arrays.activations,
            groups=arrays.groups,
            answer_types=arrays.answer_types,
        )
    elif probe_method == "rand":
        return train_random_probe(
            activations=arrays.activations,
        )
    else:
        raise ValueError(f"Unknown probe_method: {probe_method}")
