from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Union, Optional
import numpy as np
from collections import namedtuple

from plotly import express as px
from plotly import graph_objects as go


@dataclass
class FEPredictions:
    actuals_by_group: Dict[str, np.ndarray]
    yhats_by_group: Dict[str, np.ndarray]
    residuals_by_group: Dict[str, np.ndarray] = field(default_factory=dict)
    mean_residual_by_group: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.mean_residual_by_group: Dict[str, float] = self.compute_mean_of_residuals()

    def compute_mean_of_residuals(self) -> Dict[str, float]:
        """
        compute average residual by group (for recovering the group intercepts??)
        """
        means_dict: Dict[str, float] = {g: np.mean(residuals) for g, residuals in self.residuals_by_group.items()}
        return means_dict

    def residuals_distribution(self, **layout_kwargs) -> None:
        """
        plots histogram distribution of residuals by group
        """
        fig: go.Figure = go.Figure()
        key: str = 'fe' if layout_kwargs['fe'] else 'ols'
        for g, residuals in self.residuals_by_group.items():
            fig.add_trace(go.Histogram(x=residuals, name=f"{g}_{key}"))
        fig.update_layout(height=layout_kwargs['height'], width=layout_kwargs['width'])
        fig.update_xaxes(title_text='residual')
        fig.show()


@dataclass
class FEParams:
    covar_names: List[str]
    bootstrap_params: List[np.ndarray]
    ci_size_of_test: float = field(default=0.05)
    param_estimates: Dict[str, float] = field(default_factory=dict)
    param_vcov: Optional[np.ndarray] = field(default=None)
    param_ci: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    param_stderr: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.param_estimates: Dict[str, float] = self.compute_param_estimates()
        self.param_vcov: np.ndarray = self.compute_variance_of_bootstraps()
        self.param_stderr: Dict[str, float] = self.compute_stderrs()
        self.param_ci: Dict[str, Tuple[float, float]] = self.compute_bootstrap_confidence_intervals()
        
    def compute_param_estimates(self) -> Dict[str, float]:
        """
        computes mean estimates from the boostrap trials
        """
        estimates: Dict[str, float] = {name: np.mean([params[i] for params in self.bootstrap_params]) \
            for i, name in enumerate(self.covar_names)}
        return estimates

    def compute_variance_of_bootstraps(self) -> np.ndarray:
        """
        computes variance/covariance matrix of parameters
        """
        stacked: np.ndarray = np.hstack(self.bootstrap_params).T
        vcov: np.ndarray = np.cov(stacked, rowvar=False)
        return vcov

    def compute_bootstrap_confidence_intervals(self) -> Dict[str, Tuple[float, float]]:
        """
        computes 95% interval (can be later adapted for any desired size)
        """
        ci: Dict[str, Tuple[float, float]] = {name: (est - self.param_stderr[name] * 1.96, est + self.param_stderr[name] * 1.96) \
            for name, est in self.param_estimates.items()}
        return ci
        
    def compute_stderrs(self) -> Dict[str, float]:
        """
        computes standard errors of the estimates
        """
        n: int = len(self.bootstrap_params)
        var: np.ndarray = np.diag(self.param_vcov)
        stderrs: Dict[str, float] = {name: np.sqrt(v / n) for name, v in zip(self.covar_names, var)}
        return stderrs
        
    def plot_param_distribution(self, covariate: str, **kwargs) -> None:
        """
        plots the distribution of the estimates 
        nb: uses the sames order of covariates in FixedEffectsOLS.x_names  
        """
        mapping: Dict[str, int] = {'ldx': 0, 'ls': 1}
        estimates: List[float] = [arr[mapping[covariate]][0] for arr in self.bootstrap_params]
        fig: go.Figure = px.histogram(x=estimates, nbins=kwargs['nbins'], height=kwargs['height'], \
            width=kwargs['width'])
        key: str = 'fe' if kwargs['fe'] else 'ols'
        fig.update_xaxes(title_text=f"{covariate}_{key}")
        fig.show()

