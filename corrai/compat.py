"""
Compatibility patches for SALib with pandas >= 2.x
"""

import numpy as np
import pandas as pd

import SALib.util
import SALib.analyze.morris
import SALib.util.util_funcs
import SALib.analyze.morris
import SALib.analyze.fast
import SALib.analyze.sobol
import SALib.sample.sobol

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

def _check_groups(problem):
    """Check if there is more than 1 group."""
    groups = problem.get("groups")

    if isinstance(groups, np.ndarray):
        groups = groups.tolist()

    if not groups:
        return False

    if len(set(groups)) == 1:
        return False
    return groups

# Patch in source module
SALib.util.util_funcs._check_groups = _check_groups

# Patch in consumers
SALib.analyze.morris._check_groups = _check_groups
SALib.analyze.fast._check_groups = _check_groups
SALib.analyze.sobol._check_groups = _check_groups
SALib.sample.sobol._check_groups = _check_groups
