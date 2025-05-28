# MIT License

# Copyright (c) 2024 Lars Andersen Bratholm

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# https://github.com/optuna/optuna/issues/5299

"""
Variation of the multivariate group-decomposed TPESampler, that samples parameter subspaces in a hierarchical
fashion. E.g. if the possible parameter combinations that can be sampled are {x, y, a} and {x, y, b}, then the
parameter subspaces are.

[{x, y}, {a}, {b}]. Since x and y are always sampled, n_ei_candidates sampled of {x, y} is generated. Then a
classifier estimates which of the sampled parameter values will result in 'a' being sampled, and which will
result in 'b' being sampled. Then either 'a' or 'b' is sampled depending on the values of {x, y}.
"""

import functools
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import optuna
import sklearn.preprocessing
from loguru import logger
from numpy.typing import NDArray
from optuna.distributions import BaseDistribution, CategoricalDistribution
from optuna.samplers import TPESampler
from optuna.samplers._tpe.parzen_estimator import _ParzenEstimator
from optuna.samplers._tpe.sampler import _calculate_weights_below_for_multi_objective
from optuna.search_space import intersection_search_space
from optuna.study import Study
from optuna.trial import FrozenTrial, TrialState
from sklearn.tree import DecisionTreeClassifier
from typing_extensions import Unpack


class DefaultGroupClassifier:
    """
    Create a Decision Tree classifier on the parent group to determine which child group will be sampled.
    """

    def __init__(self) -> None:
        self.classifiers: Dict[int, DecisionTreeClassifier] = {}
        self.encoders: Dict[
            str,
            Callable[
                [NDArray[np.float64]], Union[NDArray[np.float64], NDArray[np.int_]]
            ],
        ] = {}
        self._groups: List[
            Dict[
                str,
                BaseDistribution,
            ]
        ] = []

    def __call__(
        self,
        samples: Dict[str, NDArray[np.float64]],
        trials: List[FrozenTrial],
        groups: List[
            Dict[
                str,
                BaseDistribution,
            ]
        ],
        child_group_indices: List[int],
        parent_group_index: int,
    ) -> NDArray[np.int_]:
        """
        Make predictions based on the sampled parent parameters of which child group will be sampled.
        """
        self._validate_cache(groups)
        parent_group = groups[parent_group_index]
        self._create_encoders(parent_group)
        trial_features = self._create_trial_features(trials, parent_group)
        trial_targets = self._create_trial_targets(trials, child_group_indices)
        self._fit_classifier(trial_features, trial_targets, parent_group_index)
        sample_features = self._create_sample_features(samples, parent_group)
        class_label_predictions = self.classifiers[parent_group_index].predict(
            sample_features
        )
        predictions: NDArray[np.int_] = self.classifiers[parent_group_index].classes_[
            class_label_predictions
        ]
        return predictions

    def _validate_cache(
        self,
        groups: List[
            Dict[
                str,
                BaseDistribution,
            ]
        ],
    ) -> None:
        """
        Check if.
        """
        if self._groups != groups:
            if len(self._groups) > 0:
                logger.debug(f"groups were {self._groups}, now {groups}")
            self._groups = groups
            self.classifiers = {}

    def _create_encoders(self, group: Dict[str, BaseDistribution]) -> None:
        for param_name, distribution in group.items():
            if param_name in self.encoders:
                continue
            if isinstance(distribution, CategoricalDistribution):
                choices = [
                    distribution.to_internal_repr(choice)
                    for choice in distribution.choices
                ]
                self.encoders[param_name] = (
                    lambda x: sklearn.preprocessing.label_binarize(x, classes=choices)
                )
            else:
                self.encoders[param_name] = lambda x: np.asarray(x).reshape(-1, 1)

    def _create_trial_features(
        self, trials: List[FrozenTrial], group: Dict[str, BaseDistribution]
    ) -> NDArray[np.float64]:
        features = []
        for param_name, distribution in group.items():
            internal_repr: NDArray[np.float64] = np.asarray(
                [
                    distribution.to_internal_repr(trial.params[param_name])
                    for trial in trials
                ]
            )
            param_features = self.encoders[param_name](internal_repr)
            features.append(param_features)
        features_: NDArray[np.float64] = np.concatenate(features, axis=1)
        assert len(trials) == features_.shape[0]
        return features_

    def _create_trial_targets(
        self, trials: List[FrozenTrial], group_indices: List[int]
    ) -> NDArray[np.int_]:
        n_groups = len(group_indices)
        targets = []
        for trial in trials:
            trial_params = set(trial.params)
            for target_idx, group_idx in enumerate(group_indices):
                if trial_params.issuperset(self._groups[group_idx]):
                    targets.append(target_idx)
                    break
            else:
                # No children in trial
                targets.append(n_groups)

        targets_ = np.asarray(targets)
        assert targets_.shape[0] == len(trials)
        return targets_

    def _fit_classifier(
        self, features: NDArray[np.float64], targets: NDArray[np.int_], index: int
    ) -> None:
        # Use cached classifier if it correctly classifies
        # all trials
        if index in self.classifiers:
            predictions = self.classifiers[index].predict(features)
            if (predictions == targets).all():
                return

        n_samples = features.shape[0]
        # Keep increasing depth until all samples are classified correctly
        for depth in range(1, n_samples):
            classifier = DecisionTreeClassifier(max_depth=depth, max_features=1)
            classifier.fit(features, targets)
            predictions = classifier.predict(features)
            if (predictions == targets).all():
                self.classifiers[index] = classifier
                return

        raise RuntimeError("Could not fit classifier")

    def _create_sample_features(
        self,
        samples: Dict[str, NDArray[np.float64]],
        parent_group: Dict[str, BaseDistribution],
    ) -> NDArray[np.float64]:
        sample_features: List[Union[NDArray[np.float64], NDArray[np.int_]]] = []
        for param_name in parent_group:
            param_features: Union[NDArray[np.float64], NDArray[np.int_]] = (
                self.encoders[param_name](samples[param_name])
            )
            sample_features.append(param_features)
        sample_features_: NDArray[np.float64] = np.concatenate(sample_features, axis=1)
        return sample_features_


def objective_classifier(
    samples: Dict[str, NDArray[np.float64]],
    trials: List[FrozenTrial],
    groups: List[Dict[str, BaseDistribution]],
    child_group_indices: List[int],
    parent_group_index: int,
) -> NDArray[np.int_]:
    """
    Manual classifier for the benchmark objective.
    """
    parent_group = groups[parent_group_index]

    if set(parent_group) == set(("x", "y")):
        predictions = np.zeros(samples["x"].shape, dtype=int)
        samples_external_repr = np.asarray(
            [parent_group["x"].to_external_repr(sample) for sample in samples["x"]],
            dtype=bool,
        )
        assert len(child_group_indices) == 2
        for i, idx in enumerate(child_group_indices):
            child_group = set(groups[idx])
            if "n" in child_group:
                predictions[samples_external_repr == True] = i
            else:
                assert "m" in child_group
                predictions[samples_external_repr == False] = i
    elif set(parent_group) == set(("n",)):
        predictions = np.zeros(samples["n"].shape, dtype=int)
        samples_external_repr = np.asarray(
            [parent_group["n"].to_external_repr(sample) for sample in samples["n"]],
            dtype=bool,
        )
        for i, idx in enumerate(child_group_indices):
            child_group = set(groups[idx])
            if "a" in child_group:
                predictions[samples_external_repr == True] = i
            else:
                assert "b" in child_group
                predictions[samples_external_repr == False] = i
    else:
        assert set(parent_group) == set(("m",))
        predictions = np.zeros(samples["m"].shape, dtype=int)
        samples_external_repr = np.asarray(
            [parent_group["m"].to_external_repr(sample) for sample in samples["m"]],
            dtype=bool,
        )
        for i, idx in enumerate(child_group_indices):
            child_group = set(groups[idx])
            if "c" in child_group:
                predictions[samples_external_repr == True] = i
            else:
                assert "d" in child_group
                predictions[samples_external_repr == False] = i
    return predictions


class HierarchicalTPESampler(TPESampler):
    """
    The group-decomposed TPESampler, samples each group independently.

    Often parameters in each group can still be correlated.
    This sampler utilizes that the intersection search space
    of the trials containing the parameters of a group can be
    the joint set of both the group itself and other groups as well.
    Each observed group combination is sampled from and an
    additional term is added to the likelihood representing the
    population of each group in below/above.
    """

    def __init__(
        self,
        *args: Unpack[Sequence[Any]],  # type: ignore
        classifier: Optional[
            Union[
                Callable[
                    [
                        Dict[str, NDArray[np.float64]],
                        List[FrozenTrial],
                        List[Dict[str, BaseDistribution]],
                        List[int],
                        int,
                    ],
                    NDArray[np.int_],
                ],
                DefaultGroupClassifier,
            ]
        ] = None,
        **kwargs: Unpack[Dict[str, Any]],  # type: ignore
    ) -> None:
        if classifier is None:
            self._classifier: Union[
                Callable[
                    [
                        Dict[str, NDArray[np.float64]],
                        List[FrozenTrial],
                        List[Dict[str, BaseDistribution]],
                        List[int],
                        int,
                    ],
                    NDArray[np.int_],
                ],
                DefaultGroupClassifier,
            ] = DefaultGroupClassifier()
        else:
            self._classifier = classifier
        self._tpe_sampler = TPESampler(*args, multivariate=False, group=False, **kwargs)

        super().__init__(*args, multivariate=True, group=True, **kwargs)

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: Dict[str, BaseDistribution],
    ) -> Dict[str, Any]:
        assert self._search_space_group is not None

        if search_space == {}:
            return {}

        states = (TrialState.COMPLETE, TrialState.PRUNED)
        trials = study._get_trials(deepcopy=False, states=states, use_cache=True)
        # If the number of samples is insufficient, we run random trial.
        if len(trials) < self._n_startup_trials:
            return {}

        params = self._sample_hierarchy(study, search_space)

        # The unsampled parameters will likely not be used during evaluation,
        # but can be if the trained classifier makes an incorrect prediction
        for param_name, param_distribution in search_space.items():
            if param_name not in params:
                params[param_name] = self._tpe_sampler.sample_independent(
                    study, trial, param_name, param_distribution
                )
        return params

    def _get_trials(self, study: Study) -> List[FrozenTrial]:
        """
        Get complete, pruned and possibly running trials.
        """
        if self._constant_liar:
            states = [TrialState.COMPLETE, TrialState.PRUNED, TrialState.RUNNING]
        else:
            states = [TrialState.COMPLETE, TrialState.PRUNED]
        use_cache = not self._constant_liar
        trials = study._get_trials(deepcopy=False, states=states, use_cache=use_cache)
        return trials

    def _get_group_probabilities(
        self, group_weights: List[List[List[float]]]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Get the group probabilities with prior (symmetrical dirichlet with alpha=1) if required.
        """
        n_groups = len(group_weights[0])
        below_weights = np.asarray(
            functools.reduce(lambda x, y: x + y, group_weights[0])
        )
        above_weights = np.asarray(
            functools.reduce(lambda x, y: x + y, group_weights[1])
        )
        p_group_below = np.zeros(n_groups)
        p_group_above = np.zeros(n_groups)
        assert self._parzen_estimator_parameters.prior_weight is not None
        if self._parzen_estimator_parameters.consider_prior is True:
            n_effective_samples_below = (
                HierarchicalTPESampler._get_effective_sample_size(below_weights)
            )
            n_effective_samples_above = (
                HierarchicalTPESampler._get_effective_sample_size(above_weights)
            )
            for i, weights in enumerate(group_weights[0]):
                p_group_below[i] = (
                    sum(weights) / sum(below_weights) * n_effective_samples_below
                    + self._parzen_estimator_parameters.prior_weight * 1.0
                )
            for i, weights in enumerate(group_weights[1]):
                p_group_above[i] = (
                    sum(weights) / sum(above_weights) * n_effective_samples_above
                    + self._parzen_estimator_parameters.prior_weight * 1.0
                )
        else:
            for i, weights in enumerate(group_weights[0]):
                p_group_below[i] = sum(weights) / sum(below_weights)
            for i, weights in enumerate(group_weights[1]):
                p_group_above[i] = sum(weights) / sum(above_weights)

        # Normalize
        p_group_below /= p_group_below.sum()
        p_group_above /= p_group_above.sum()
        return p_group_below, p_group_above

    def _get_group_trials_and_weights(
        self,
        study: Study,
        trials: List[FrozenTrial],
        search_spaces: List[Dict[str, BaseDistribution]],
        parent_search_space: Optional[Dict[str, BaseDistribution]] = None,
    ) -> Tuple[List[List[List[FrozenTrial]]], List[List[List[float]]]]:
        """
        Get the trial subsets and group weights for the given search spaces.

        :param study: the study
        :param trials: observed trials
        :param search_spaces: the overlapping group search spaces
        :returns: group trials and group weights for below and above
        """
        # Split trials into below/above and get their weights
        below_trials, above_trials, weights_below, weights_above = (
            self._split_trials_weights(trials, study)
        )

        # Gather the trials belonging to each group, with
        # index=0 for below and index=1 for above.
        group_trials: List[List[List[FrozenTrial]]] = [
            [[] for _ in search_spaces],
            [[] for _ in search_spaces],
        ]
        group_weights: List[List[List[float]]] = [
            [[] for _ in search_spaces],
            [[] for _ in search_spaces],
        ]
        if parent_search_space is not None:
            group_trials[0].append([])
            group_trials[1].append([])
            group_weights[0].append([])
            group_weights[1].append([])
        # First assign trials where the intersection search space match a search space exactly
        intersection = intersection_search_space(trials)
        intersection_search_space_index = None
        for k, search_space in enumerate(search_spaces):
            if set(intersection) == set(search_space):
                intersection_search_space_index = k
        n_below = len(below_trials)
        for i, (trial, weight) in enumerate(
            zip(
                below_trials + above_trials,
                weights_below.tolist() + weights_above.tolist(),
            )
        ):
            below_above_index = int(i < n_below)
            # First assign trials where the intersection search space match a search space exactly
            if intersection_search_space_index is not None:
                if set(trial.params) == set(
                    search_spaces[intersection_search_space_index]
                ):
                    group_trials[below_above_index][
                        intersection_search_space_index
                    ].append(trial)
                    group_weights[below_above_index][
                        intersection_search_space_index
                    ].append(weight)
                    assert False
                    continue
            if parent_search_space is not None and set(trial.params) == set(
                parent_search_space
            ):
                group_trials[below_above_index][-1].append(trial)
                group_weights[below_above_index][-1].append(weight)
                assert False
                continue

            n_assignments = 0
            # Then relax to supersets excluding the intersection search space group
            for k, search_space in enumerate(search_spaces):
                if k == intersection_search_space_index:
                    continue
                if set(trial.params).issuperset(search_space):
                    n_assignments += 1
                    group_trials[below_above_index][k].append(trial)
                    group_weights[below_above_index][k].append(weight)
            assert n_assignments < 2
            if parent_search_space is not None:
                assert set(trial.params).issuperset(parent_search_space)
                if n_assignments == 0:
                    group_trials[below_above_index][-1].append(trial)
                    group_weights[below_above_index][-1].append(weight)

        # Check that all trials are assigned
        assert len(
            functools.reduce(
                lambda x, y: x + y,
                functools.reduce(lambda x, y: x + y, group_trials),  # type: ignore
            )
        ) == len(trials)
        return group_trials, group_weights

    def _split_trials_weights(
        self, trials: List[FrozenTrial], study: Study
    ) -> Tuple[
        List[FrozenTrial], List[FrozenTrial], NDArray[np.float64], NDArray[np.float64]
    ]:
        # Divide data into below and above.
        n_trials = sum(
            trial.state != TrialState.RUNNING for trial in trials
        )  # Ignore running trials.
        below_trials, above_trials = optuna.samplers._tpe.sampler._split_trials(
            study,
            trials,
            self._gamma(n_trials),
            self._constraints_func is not None,
        )
        # Calculate weights of trials
        if study._is_multi_objective():
            weights_below = _calculate_weights_below_for_multi_objective(
                study, below_trials, self._constraints_func
            )
        else:
            weights_below = self._parzen_estimator_parameters.weights(len(below_trials))
            assert len(below_trials) == len(weights_below)
        weights_above = self._parzen_estimator_parameters.weights(len(above_trials))

        # Normalize
        weights_below /= weights_below.sum()
        weights_above /= weights_above.sum()
        return below_trials, above_trials, weights_below, weights_above

    def _get_mpes(
        self,
        study: Study,
        trials: List[FrozenTrial],
        search_space: Dict[str, BaseDistribution],
    ) -> Tuple[_ParzenEstimator, _ParzenEstimator]:
        """
        Generate samples with the Parzen estimator, using the trial subset containing all parameters in the
        given search space.

        :param study: the study containing observed trials
        :param trials: the trials with the same intersection search space as the given search space
        :param search_space: the search space to expore
        :returns: the samples and the log-likelihood difference between below and above
        """
        assert all(set(trial.params).issuperset(search_space) for trial in trials)

        # We divide data into below and above.
        n = sum(
            trial.state != TrialState.RUNNING for trial in trials
        )  # Ignore running trials.
        below_trials, above_trials = optuna.samplers._tpe.sampler._split_trials(
            study,
            trials,
            self._gamma(n),
            self._constraints_func is not None,
        )

        below = self._get_internal_repr(below_trials, search_space)
        above = self._get_internal_repr(above_trials, search_space)

        # We then sample by maximizing log likelihood ratio.
        if study._is_multi_objective():
            weights_below = _calculate_weights_below_for_multi_objective(
                study, below_trials, self._constraints_func
            )
            mpe_below = _ParzenEstimator(
                below, search_space, self._parzen_estimator_parameters, weights_below
            )
        else:
            mpe_below = _ParzenEstimator(
                below, search_space, self._parzen_estimator_parameters
            )
        mpe_above = _ParzenEstimator(
            above, search_space, self._parzen_estimator_parameters
        )

        return mpe_below, mpe_above

    def _sample_hierarchy(
        self,
        study: Study,
        search_space: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate samples and select the one with the highest likelihood ratio between below and above.

        :param study: the study containing observed trials
        :param search_space: the full observed search space
        """
        # Get trials
        trials = self._get_trials(study)

        # Determine the hierachy from observed trials
        hierarchy = self._determine_hierarchy(trials)
        assert self._search_space_group is not None
        groups = self._search_space_group.search_spaces

        # Generate samples hierarchically and get the log likelihood difference
        # between below and above
        mask, samples, delta_ll = self._get_hierarchical_samples(
            study, hierarchy, groups, trials, self._n_ei_candidates
        )

        # Select the best overall sample
        ret = {}
        best_sample_index = np.argmax(delta_ll)
        mask_to_idx = np.empty(self._n_ei_candidates, dtype=int)
        for param_name, distribution in search_space.items():
            if mask[param_name][best_sample_index] is np.bool_(True):
                mask_to_idx[mask[param_name]] = np.arange(mask[param_name].sum())
                idx = mask_to_idx[best_sample_index]
                ret[param_name] = distribution.to_external_repr(
                    samples[param_name][idx]
                )
        return ret

    def _get_samples(
        self,
        study: Study,
        trials: List[FrozenTrial],
        group: Dict[str, BaseDistribution],
        n_samples: int,
    ) -> Tuple[Dict[str, NDArray[np.float64]], NDArray[np.float64]]:
        trial_subset = [trial for trial in trials if set(group).issubset(trial.params)]
        mpe_below, mpe_above = self._get_mpes(study, trial_subset, group)
        samples = mpe_below.sample(self._rng.rng, n_samples)
        assert set(samples) == set(group)
        log_likelihoods_below = mpe_below.log_pdf(samples)
        log_likelihoods_above = mpe_above.log_pdf(samples)
        delta_ll = log_likelihoods_below - log_likelihoods_above
        return samples, delta_ll

    def _get_hierarchical_samples(
        self,
        study: Study,
        hierarchy: List[Optional[int]],
        groups: List[Dict[str, BaseDistribution]],
        trials: List[FrozenTrial],
        n_samples: int,
    ) -> Tuple[
        Dict[str, NDArray[np.bool_]],
        Dict[str, NDArray[np.float64]],
        NDArray[np.float64],
    ]:
        samples = {}
        mask = {}
        for group in groups:
            for param_name in group:
                mask[param_name] = np.zeros(n_samples, dtype=bool)
        delta_ll = np.zeros(n_samples)
        n_groups = len(hierarchy)
        # Sample the current top-level groups
        for i in range(n_groups):
            if hierarchy[i] is not None:
                continue
            group_samples, d_ll = self._get_samples(
                study, trials, groups[i], n_samples=n_samples
            )
            for param_name in group_samples:
                assert param_name not in samples
                # if param_name not in samples:
                mask[param_name][:] = True
                samples[param_name] = group_samples[param_name]
            delta_ll += d_ll

            self._get_hierarchical_child_samples(
                study,
                hierarchy,
                groups,
                trials,
                mask,  # type: ignore
                samples,
                delta_ll,
                parent_group_index=i,
            )

            assert all(mask[param_name].all() for param_name in groups[i])
        return mask, samples, delta_ll  # type: ignore

    def _get_hierarchical_child_samples(
        self,
        study: Study,
        hierarchy: List[Optional[int]],
        groups: List[Dict[str, BaseDistribution]],
        trials: List[FrozenTrial],
        mask: Dict[str, NDArray[np.bool_]],
        samples: Dict[str, NDArray[np.float64]],
        delta_ll: NDArray[np.float64],
        parent_group_index: int,
    ) -> None:
        """
        Generate samples for each group having the given group as parent.
        """
        n_groups = len(hierarchy)
        child_group_indices = [
            index for index in range(n_groups) if hierarchy[index] == parent_group_index
        ]
        n_child_groups = len(child_group_indices)
        if n_child_groups == 0:
            return

        parent_group = groups[parent_group_index]
        trials = [
            trial for trial in trials if set(trial.params).issuperset(parent_group)
        ]
        predictions = self._classifier(
            samples, trials, groups, child_group_indices, parent_group_index
        )

        # Get the subsets of below and above trials belonging to each group, and the relative weights
        # of each group
        _, group_weights = self._get_group_trials_and_weights(
            study,
            trials,
            [groups[index] for index in child_group_indices],
            parent_group,
        )
        # Get group counts
        p_group_below, p_group_above = self._get_group_probabilities(group_weights)

        for i, index in enumerate(child_group_indices):
            prediction_mask = predictions == i
            if sum(prediction_mask) == 0:
                continue
            child_hierarchy = [
                -1 if item is None else None if j == index else item
                for j, item in enumerate(hierarchy)
            ]
            child_mask, child_samples, child_delta_ll = self._get_hierarchical_samples(
                study=study,
                hierarchy=child_hierarchy,
                groups=groups,
                trials=trials,
                n_samples=sum(prediction_mask),
            )
            # Add in delta log likelihoods for the group contribution
            delta_ll[prediction_mask] += (
                child_delta_ll + np.log(p_group_below[i]) - np.log(p_group_above[i])
            )
            for param_name in child_samples:
                assert param_name not in samples
                mask[param_name][prediction_mask] = child_mask[param_name]
                samples[param_name] = child_samples[param_name]

        # Add delta ll for when there is no children groups
        prediction_mask = predictions == n_groups
        if sum(prediction_mask) > 0:
            delta_ll[prediction_mask] += np.log(p_group_below[n_groups]) - np.log(
                p_group_above[n_groups]
            )

    def _determine_hierarchy(self, trials: List[FrozenTrial]) -> List[Optional[int]]:
        """
        :param trials: the observed trials
        """
        trials = [trial for trial in trials if trial.state != TrialState.RUNNING]
        assert self._search_space_group is not None
        groups = self._search_space_group.search_spaces
        n_groups = len(groups)
        hierarchy: List[List[Optional[int]]] = [[] for _ in range(n_groups)]
        for j, child_group in enumerate(groups):
            child_group_intersection_search_space = intersection_search_space(
                [trial for trial in trials if set(trial.params).issuperset(child_group)]
            )
            for i, parent_group in enumerate(groups):
                if i == j:
                    continue
                if set(parent_group).issubset(child_group_intersection_search_space):
                    hierarchy[j].append(i)
        # Clean so each group only keeps the direct parent
        finished = False
        counter = 0
        while not finished:
            counter += 1
            if counter >= 1000:
                raise RuntimeError(
                    "Unable to determine a strict hierarchy between parameter groups"
                )
            finished = True
            for item in hierarchy:
                if len(item) <= 1:
                    continue
                finished = False
                for index1 in item:
                    for index2 in item:
                        if index1 == index2:
                            continue
                        assert index2 is not None
                        if index1 in hierarchy[index2]:
                            item.pop(item.index(index1))
                            break
                    else:
                        continue
                    break
                break
        # Finally change the format to make it easier to work with
        hierarchy_ = [item[0] if len(item) == 1 else None for item in hierarchy]
        return hierarchy_

    @staticmethod
    def _get_effective_sample_size(weights: NDArray[np.float64]) -> float:
        """
        Get the effective sample size given a set of weights.

        :param weights: the weights
        :returns: the effective sample size
        """
        effective_sample_size = float(weights.sum() ** 2 / (weights**2).sum())
        return effective_sample_size
