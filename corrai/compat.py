"""
Compatibility patches for SALib with pandas >= 2.x
"""

import numpy as np
import pandas as pd

import SALib.util
import SALib.analyze.morris


# -------------------------------
# Patch compute_groups_matrix
# -------------------------------

_original_compute = SALib.util.compute_groups_matrix


def _compute_groups_matrix_patched(groups):

    # Normalize input for pandas >=2
    if isinstance(groups, list):
        groups = np.asarray(groups)

    elif isinstance(groups, pd.Index):
        groups = groups.to_numpy()

    return _original_compute(groups)


# Apply patch everywhere
SALib.util.compute_groups_matrix = _compute_groups_matrix_patched
SALib.analyze.morris.compute_groups_matrix = _compute_groups_matrix_patched


# -------------------------------
# Patch _define_problem_with_groups
# -------------------------------

_original_define = SALib.util._define_problem_with_groups


def _define_problem_with_groups_patched(problem):

    groups = problem.get("groups")

    # Normalize for SALib logic
    if isinstance(groups, np.ndarray):
        problem["groups"] = groups.tolist()

    return _original_define(problem)


SALib.util._define_problem_with_groups = _define_problem_with_groups_patched
SALib.analyze.morris._define_problem_with_groups = _define_problem_with_groups_patched
