from typing import Callable, Sequence

import numpy as np
import pandas as pd

from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Integer, Real, Choice, Binary


class Problem(ElementwiseProblem):
    """
    A  Pymoo ElementwiseProblem for real-valued and mixed-variable optimization.

    Parameters
    ----------
    parameters : list
        A list of Parameter objects, each defining a variable's name, type
        ({"Real", "Integer", "Binary", "Choice"}), and domain (interval or values).
    evaluators : Sequence[Callable]
        A sequence of functions or callable objects that take a dictionary of
        parameter values and return results as:
            - a dict mapping metric names to floats
            - a pandas.Series
            - or a single scalar (only if there is exactly one objective or constraint)
    objective_ids : Sequence[str]
        Names of the objectives to extract from the evaluator results, in the order
        they will appear in F.
    constraint_ids : Sequence[str], optional
        Names of the constraints to extract from the evaluator results, in the order
        they will appear in G. If omitted, G will be an empty array.
    """

    def __init__(
        self,
        *,
        parameters: list,
        evaluators: Sequence[Callable] | None = None,
        objective_ids: Sequence[str],
        constraint_ids: Sequence[str] | None = None,
    ):
        if not evaluators:
            raise ValueError("evaluators must be provided")

        self.parameters = list(parameters)
        _check_duplicate_params(parameters)
        self.param_names = [p.name for p in self.parameters]
        self.evaluators = list(evaluators)
        self.objective_ids = list(objective_ids)
        self.constraint_ids = list(constraint_ids) if constraint_ids else []

        self.is_all_real = all(p.ptype == "Real" for p in self.parameters)
        if self.is_all_real:
            xl = np.array([p.interval[0] for p in self.parameters], dtype=float)
            xu = np.array([p.interval[1] for p in self.parameters], dtype=float)
            super().__init__(
                n_var=len(self.parameters),
                n_obj=len(self.objective_ids),
                n_ieq_constr=len(self.constraint_ids),
                xl=xl,
                xu=xu,
            )
            self._mode = "float"
        else:
            vars_dict = {}
            for p in self.parameters:
                if p.ptype == "Real":
                    lo, hi = p.interval
                    vars_dict[p.name] = Real(bounds=(float(lo), float(hi)))
                elif p.ptype == "Integer":
                    lo, hi = p.interval
                    vars_dict[p.name] = Integer(bounds=(int(lo), int(hi)))
                elif p.ptype == "Binary":
                    vars_dict[p.name] = Binary()
                elif p.ptype == "Choice":
                    if p.values is None:
                        raise ValueError(
                            f"Parameter {p.name!r} of type Choice requires 'values'"
                        )
                    vars_dict[p.name] = Choice(options=list(p.values))
                else:
                    raise ValueError(
                        f"Unsupported ptype={p.ptype!r} for parameter {p.name!r}"
                    )
            super().__init__(
                vars=vars_dict,
                n_obj=len(self.objective_ids),
                n_ieq_constr=len(self.constraint_ids),
            )
            self._mode = "mixed"

    def _x_to_param_dict(self, x) -> dict:
        if self._mode == "float":
            return {name: float(val) for name, val in zip(self.param_names, x)}

        if isinstance(x, dict):
            return x

        if isinstance(x, (list, tuple, np.ndarray, pd.Series)):
            out = {}
            for (name, p), val in zip(zip(self.param_names, self.parameters), x):
                if p.ptype == "Integer":
                    out[name] = int(val)
                elif p.ptype == "Real":
                    out[name] = float(val)
                elif p.ptype == "Binary":
                    out[name] = bool(val)
                elif p.ptype == "Choice":
                    out[name] = val
                else:
                    raise ValueError(
                        f"Unsupported ptype={p.ptype!r} for parameter {p.name!r}"
                    )
            return out

        raise TypeError(f"Unsupported x type for mixed mode: {type(x).__name__}")

    def _aggregate(self, param_dict: dict) -> dict[str, float]:
        acc: dict[str, float] = {}
        total_ids = len(self.objective_ids) + len(self.constraint_ids)
        for block in self.evaluators:
            res = (
                block.function(param_dict)
                if hasattr(block, "function")
                else block(param_dict)
            )
            if isinstance(res, dict):
                acc.update({k: float(v) for k, v in res.items()})
            elif hasattr(res, "to_dict"):
                acc.update({k: float(v) for k, v in res.to_dict().items()})
            else:
                if total_ids != 1:
                    raise TypeError(
                        "A scalar was returned, but several objective/constraint IDs are defined"
                    )
                target = (
                    self.objective_ids[0]
                    if self.objective_ids
                    else self.constraint_ids[0]
                )
                acc[target] = float(res)
        return acc

    def _evaluate(self, x, out, *args, **kwargs):
        param_dict = self._x_to_param_dict(x)
        acc = self._aggregate(param_dict)

        F = [acc[name] for name in self.objective_ids]
        G = [acc[name] for name in self.constraint_ids] if self.constraint_ids else []

        out["F"] = F
        out["G"] = G


def _check_duplicate_params(params: list["Parameter"]) -> None:
    seen = set()
    for p in params:
        if p.name in seen:
            raise ValueError(f"Duplicate parameter name: {p.name}")
        seen.add(p.name)
