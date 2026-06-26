"""
Partial Dependence Plot utilities for HealthRisk AI.

This module generates PDP and ICE visualizations for trained regression
or classification models. These plots support model transparency by showing
how selected input features influence model predictions.
"""

from pathlib import Path
from typing import Iterable, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import PartialDependenceDisplay


PathLike = Union[str, Path]


def validate_features(X: pd.DataFrame, features: Iterable[str]) -> list[str]:
    """
    Validate that all requested features exist in the feature matrix.

    Parameters
    ----------
    X:
        Feature matrix used by the model.
    features:
        Feature names to plot.

    Returns
    -------
    list[str]
        Validated feature names.

    Raises
    ------
    ValueError
        If any requested feature is missing from X.
    """
    features = list(features)
    missing = [feature for feature in features if feature not in X.columns]

    if missing:
        raise ValueError(
            f"Missing feature(s) in feature matrix: {missing}. "
            f"Available columns include: {list(X.columns)[:10]}..."
        )

    return features


def generate_pdp_plot(
    model,
    X: pd.DataFrame,
    feature: str,
    output_dir: PathLike = "reports",
    file_prefix: str = "pdp",
    grid_resolution: int = 50,
    kind: str = "average",
    figsize: tuple[int, int] = (8, 5),
) -> Path:
    """
    Generate and save a Partial Dependence Plot for one feature.

    Parameters
    ----------
    model:
        Trained scikit-learn compatible model.
    X:
        Feature matrix.
    feature:
        Feature name to explain.
    output_dir:
        Directory where the plot will be saved.
    file_prefix:
        Prefix for the output filename.
    grid_resolution:
        Number of grid points for PDP.
    kind:
        "average" for PDP or "individual" for ICE.
    figsize:
        Figure size.

    Returns
    -------
    Path
        Saved image path.
    """
    validate_features(X, [feature])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    PartialDependenceDisplay.from_estimator(
        model,
        X,
        [feature],
        kind=kind,
        grid_resolution=grid_resolution,
        ax=ax,
    )

    title_type = "ICE Plot" if kind == "individual" else "Partial Dependence Plot"
    ax.set_title(f"{title_type}: {feature}")

    output_path = output_dir / f"{file_prefix}_{feature}.png"

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def generate_multiple_pdp_plots(
    model,
    X: pd.DataFrame,
    features: Iterable[str],
    output_dir: PathLike = "reports",
    grid_resolution: int = 50,
) -> list[Path]:
    """
    Generate PDP plots for multiple features.

    Parameters
    ----------
    model:
        Trained scikit-learn compatible model.
    X:
        Feature matrix.
    features:
        Feature names to explain.
    output_dir:
        Directory where plots will be saved.
    grid_resolution:
        Number of grid points for PDP.

    Returns
    -------
    list[Path]
        Saved plot paths.
    """
    features = validate_features(X, features)

    saved_paths = []

    for feature in features:
        path = generate_pdp_plot(
            model=model,
            X=X,
            feature=feature,
            output_dir=output_dir,
            file_prefix="pdp",
            grid_resolution=grid_resolution,
            kind="average",
        )
        saved_paths.append(path)

    return saved_paths


def generate_ice_plot(
    model,
    X: pd.DataFrame,
    feature: str,
    output_dir: PathLike = "reports",
    grid_resolution: int = 30,
    sample_size: Optional[int] = 200,
) -> Path:
    """
    Generate and save an Individual Conditional Expectation plot.

    Parameters
    ----------
    model:
        Trained scikit-learn compatible model.
    X:
        Feature matrix.
    feature:
        Feature name to explain.
    output_dir:
        Directory where the plot will be saved.
    grid_resolution:
        Number of grid points.
    sample_size:
        Optional number of records to sample for speed/readability.

    Returns
    -------
    Path
        Saved ICE plot path.
    """
    validate_features(X, [feature])

    if sample_size is not None and len(X) > sample_size:
        X_plot = X.sample(sample_size, random_state=42)
    else:
        X_plot = X.copy()

    path = generate_pdp_plot(
        model=model,
        X=X_plot,
        feature=feature,
        output_dir=output_dir,
        file_prefix="ice",
        grid_resolution=grid_resolution,
        kind="individual",
    )

    return path


def generate_pdp_report(
    model,
    X: pd.DataFrame,
    features: Iterable[str],
    output_dir: PathLike = "reports",
) -> dict[str, list[Path]]:
    """
    Generate PDP and ICE plots for a list of model features.

    Parameters
    ----------
    model:
        Trained model.
    X:
        Feature matrix.
    features:
        Feature names to explain.
    output_dir:
        Directory where plots will be saved.

    Returns
    -------
    dict[str, list[Path]]
        Dictionary containing PDP and ICE image paths.
    """
    features = validate_features(X, features)

    pdp_paths = generate_multiple_pdp_plots(
        model=model,
        X=X,
        features=features,
        output_dir=output_dir,
    )

    ice_paths = []

    for feature in features:
        ice_path = generate_ice_plot(
            model=model,
            X=X,
            feature=feature,
            output_dir=output_dir,
        )
        ice_paths.append(ice_path)

    return {
        "pdp": pdp_paths,
        "ice": ice_paths,
    }