import sys
sys.path.append('/Users/JDMS/Downloads/coding exercise')

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Union, Optional
from collections import namedtuple

import numpy as np
from analysis import FEParams, FEPredictions


EstData: namedtuple = namedtuple(typename='EstData', field_names=['y', 'X'])

@dataclass
class FixedEffectsOLS:
    in_data: List[Dict[str, Union[str, float]]]
    group_varname: str
    y_name: str
    x_names: List[str]
    group_names: List[str] = field(default_factory=list)
    y_by_group: Dict[str, np.ndarray] = field(default_factory=dict)
    x_by_group: Dict[str, np.ndarray] = field(default_factory=dict)
    bootstrap_draws: int = field(default=250)
    implement_fixed_effects: bool = field(default=True)
    fit_intercept: bool = field(default=False)

    def __post_init__(self) -> None:
        self.group_names: List[str] = list(set([obs[self.group_varname] for obs in self.in_data]))
        self.y_by_group = {g: self.collect_measures_by_group(labels=[self.y_name], group=g) for g in self.group_names}
        self.x_by_group: Dict[str, np.ndarray] = {g: self.collect_measures_by_group(labels=self.x_names, group=g)
                                                  for g in self.group_names
                                                 }

    def collect_measures_by_group(self, labels: List[str], group: str) -> np.ndarray:
        """
        groups the response variable vector and regressors 
        """
        group_data: List[Dict[str, Union[str, float]]] = [d for d in self.in_data if d[self.group_varname] == group]
        measures: np.ndarray = np.array([[d[l] for l in labels] for d in group_data])
        return measures

    @staticmethod
    def get_mean_dicts(y_by_group: Dict[str, np.ndarray],
                       x_by_group: Dict[str, np.ndarray],
                       demean_groupwise: bool
                       ) -> Dict[str, Dict[str, np.ndarray]]:

        original_y: np.ndarray = np.concatenate(list(y_by_group.values()), axis=0)
        original_x: np.ndarray = np.concatenate(list(x_by_group.values()), axis=0)

        mean_y_dict: Dict[str, np.ndarray]
        mean_x_dict: Dict[str, np.ndarray]
        if demean_groupwise:
            mean_y_dict = {g: np.mean(y, axis=0) for g, y in y_by_group.items()}
            mean_x_dict = {g: np.mean(x, axis=0) for g, x in x_by_group.items()}
        else:
            mean_y_dict = {g: np.mean(original_y, axis=0) for g in y_by_group.keys()}
            mean_x_dict = {g: np.mean(original_x, axis=0) for g in x_by_group.keys()}

        return {'y': mean_y_dict,
                'x': mean_x_dict
                }

    @staticmethod
    def create_regression_data(y_by_group: Dict[str, np.ndarray],
                               x_by_group: Dict[str, np.ndarray],
                               demean_groupwise: bool
                               ) -> EstData:
        ...
        """
        creates data object holding the (de-meaned) design matrix and response variable
        concatenates the group data into a single array
        """
        means_dict = FixedEffectsOLS.get_mean_dicts(y_by_group=y_by_group,
                                                 x_by_group=x_by_group,
                                                 demean_groupwise=demean_groupwise
                                                 )
        y_group_means: Dict[str, np.ndarray] = means_dict['y']
        x_group_means: Dict[str, np.ndarray] = means_dict['x']

        y_by_group_demeaned: Dict[str, np.ndarray] = {g: y - y_group_means[g] for g, y in y_by_group.items()}
        x_by_group_demeaned: Dict[str, np.ndarray] = {g: x - x_group_means[g] for g, x in x_by_group.items()}

        y: np.ndarray = np.concatenate(list(y_by_group_demeaned.values()), axis=0)
        x: np.ndarray = np.concatenate(list(x_by_group_demeaned.values()), axis=0)

        out: EstData = EstData(y=y, X=x)
        return out
        
    @staticmethod
    def estimate(regression_data: EstData, fit_intercept: bool) -> np.ndarray:
        """
        Find the coefficient vector of the least-squares hyperplane
        """
        X: np.ndarray = regression_data.X
        y: np.ndarray = regression_data.y

        n: int
        p: int
        n, p = X.shape
        if fit_intercept:
            X = np.hstack((np.ones((n, 1)), X))
        return np.linalg.solve(X.T @ X, X.T @ y)

    @staticmethod
    def draw_bootstrap_sample(data_dict: Dict[str, np.ndarray],
                              index_dict: Dict[str, np.ndarray],
                              ) -> Dict[str, np.ndarray]:
        """
        use the index_dict to draw a bootstrap sample from the data
        """
        return {g: data_dict[g][index_dict[g]] for g in data_dict.keys()}
        
    def bootstrap_estimate(self, rng: np.random.Generator, **kwargs) -> FEParams:
        covar_names: List[str] = self.x_names

        def fit_iteration(x_dict: Dict[str, np.ndarray],
                          y_dict: Dict[str, np.ndarray]
                          ) -> np.ndarray:

            obscount_dict: Dict[str, int] = {g: a.shape[0] for g, a in y_dict.items()}
            index_dict: Dict[str, np.ndarray] = {g: rng.integers(low=0, high=n, size=n)
                                                 for g, n in obscount_dict.items()
                                                 }

            sampled_x: Dict[str, np.ndarray] = FixedEffectsOLS.draw_bootstrap_sample(data_dict=x_dict,
                                                                                     index_dict=index_dict
                                                                                     )
            sampled_y: Dict[str, np.ndarray] = FixedEffectsOLS.draw_bootstrap_sample(data_dict=y_dict,
                                                                                     index_dict=index_dict
                                                                                     )

            regdata: EstData = FixedEffectsOLS.create_regression_data(x_by_group=sampled_x,
                                                                      y_by_group=sampled_y,
                                                                      demean_groupwise=self.implement_fixed_effects
                                                                      )


            betahats: np.ndarray = FixedEffectsOLS.estimate(regression_data=regdata, fit_intercept=self.fit_intercept)
            return betahats

        bootstrap_params: List[np.ndarray] = [fit_iteration(x_dict=self.x_by_group,
                                                            y_dict=self.y_by_group
                                                            ) for _ in range(self.bootstrap_draws)
                                              ]
  
        out: FEParams = FEParams(covar_names=covar_names, bootstrap_params=bootstrap_params, **kwargs)
        return out

    def fe_predict(self, beta: Optional[FEParams] = None) -> FEPredictions:
        """
        predicts the response vector Y given the coefficient estimates 
        """
        yhats_by_group: Dict[str, np.ndarray] = {}
        residuals_by_group: Dict[str, np.ndarray] = {}
        for group, x_values in self.x_by_group.items():
            yhat: np.ndarray = x_values @ np.array(list(beta.param_estimates.values()))
            yhats_by_group[group] = yhat
            residuals_by_group[group] = self.y_by_group[group].flatten() - yhat

        out: FEPredictions = FEPredictions(actuals_by_group=self.y_by_group, yhats_by_group=yhats_by_group, \
            residuals_by_group=residuals_by_group)
        return out
        
        
def add_log_transforms(raw_data: List[Dict[str, Union[str, float]]],
                       vars_to_transform: List[str],
                       newname_dict: Optional[Dict[str, str]] = None
                       ) -> None:
    """
    transforms the data in place, by adding log transformations of the variables
    """
    for d in raw_data:
        for v in vars_to_transform:
            d[newname_dict[v]] = np.log(d[v])


if __name__ == "__main__":
    import json 

    with open('./simulated_housing_transactions.json', 'r') as simdata_file:
        simdata: List[Dict[str, Union[str, float]]] = json.load(simdata_file)
    print(simdata[3272])

    full_data = [d.copy() for d in simdata]
    add_log_transforms(raw_data=full_data,
                    vars_to_transform=['transaction_value',
                                        'distance',
                                        'size'
                                        ],
                    newname_dict={'transaction_value': 'ltv', 'distance': 'ldx', 'size': 'ls'}
                    )
    print(full_data[3272])

    s: int = 297728298824347083322041603037191577855
    ss: np.random.SeedSequence = np.random.SeedSequence(entropy=s)
    my_rng: np.random.Generator = np.random.default_rng(seed=ss.generate_state(n_words=1))

    fetst: FixedEffectsOLS = FixedEffectsOLS(in_data=full_data,
                                         group_varname='city',
                                         y_name='ltv',
                                         x_names=['ldx', 'ls'],
                                         bootstrap_draws=500,
                                            )

    fe_beta: FEParams = fetst.bootstrap_estimate(rng=my_rng)
    fe_yhats: FEPredictions = fetst.fe_predict(beta=fe_beta)

    olstst: FixedEffectsOLS = FixedEffectsOLS(in_data=full_data,
                                          group_varname='city',
                                          y_name='ltv',
                                          x_names=['ldx', 'ls'],
                                          bootstrap_draws=500,
                                          implement_fixed_effects=False
                                          )
    olsbeta: FEParams = olstst.bootstrap_estimate(rng=my_rng)
    ols_yhats: FEPredictions = olstst.fe_predict(beta=olsbeta)

    fe_beta.plot_param_distribution(covariate='ls', nbins=50, height=700, width=1000, fe=True)
    fe_beta.plot_param_distribution(covariate='ldx', nbins=50, height=700, width=1000, fe=True)
    olsbeta.plot_param_distribution(covariate='ls', nbins=50, height=700, width=1000, fe=False)
    olsbeta.plot_param_distribution(covariate='ldx', nbins=50, height=700, width=1000, fe=False)
    fe_yhats.residuals_distribution(height=700, width=1000, fe=True)
    ols_yhats.residuals_distribution(height=700, width=1000, fe=False)
