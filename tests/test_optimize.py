import pandas as pd
import numpy as np
from pathlib import Path
import pytest

from corrai.optimize import (
    MixedProblem,
    RealContinuousProblem,
    PymooModelEvaluator,
    check_duplicate_params,
    ModelEvaluator,
    SciOptimizer,
)
from corrai.base.parameter import Parameter
from corrai.base.model import Ishigami, IshigamiDynamic, PyModel

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize
from pymoo.core.mixed import MixedVariableGA
from pymoo.termination import get_termination


PACKAGE_DIR = Path(__file__).parent / "TestLib"


class X2(PyModel):
    def __init__(self, is_dynamic: bool = False):
        super().__init__(is_dynamic)
        self.x = 10

    def simulate(
        self,
        property_dict: dict[str, str | int | float] = None,
        simulation_options: dict = None,
        **simulation_kwargs,
    ) -> pd.DataFrame | pd.Series:
        self.set_property_values(property_dict)
        return pd.Series({"f_out": self.x**2})


class PyRosen(PyModel):
    def __init__(self, x_init=1.0, y_init=1.0):
        super().__init__(is_dynamic=False)
        self.x = x_init
        self.y = y_init

    def simulate(
        self,
        property_dict: dict[str, str | int | float] = None,
        simulation_options: dict = None,
        **simulation_kwargs,
    ) -> pd.DataFrame | pd.Series:
        self.set_property_values(property_dict)
        return pd.Series({"f1": (1 - self.x) ** 2 + 100 * (self.y - self.x**2) ** 2})


class BinAndKorn1(PyModel):
    def __init__(self, x_init=1.0, y_init=1.0):
        super().__init__(is_dynamic=False)
        self.x = x_init
        self.y = y_init

    def simulate(
        self,
        property_dict: dict[str, str | int | float] = None,
        simulation_options: dict = None,
        **simulation_kwargs,
    ) -> pd.DataFrame | pd.Series:
        self.set_property_values(property_dict)
        f1 = 4 * self.x**2 + 4 * self.y**2
        f2 = (self.x - 5) ** 2 + (self.y - 5) ** 2
        g1 = (self.x - 5) ** 2 + (self.y - 5) ** 2 - 25
        return pd.Series({"f1": f1, "f2": f2, "g1": g1})


class BinAndKorn2(PyModel):
    def __init__(self, x_init=1.0, y_init=1.0):
        super().__init__(is_dynamic=False)
        self.x = x_init
        self.y = y_init

    def simulate(
        self,
        property_dict: dict[str, str | int | float] = None,
        simulation_options: dict = None,
        **simulation_kwargs,
    ) -> pd.DataFrame | pd.Series:
        self.set_property_values(property_dict)
        g2 = 7.7 - (self.x - 8) ** 2 - (self.y + 3) ** 2
        return pd.Series({"g2": g2})


class MixedProblemModel(PyModel):
    def __init__(self, operator_init="add", switch_init=False, y_init=1.0, z_init=10.0):
        super().__init__(is_dynamic=False)
        self.operator = operator_init
        self.switch = switch_init
        self.y = y_init
        self.z = z_init

    def simulate(
        self,
        property_dict: dict[str, str | int | float] = None,
        simulation_options: dict = None,
        **simulation_kwargs,
    ) -> pd.DataFrame | pd.Series:
        self.set_property_values(property_dict)
        if self.operator == "multiply":
            op = np.multiply
        elif self.operator == "add":
            op = np.add

        if self.switch:
            f1 = op(self.z**2, self.y**2)
            f2 = np.sqrt(self.z**2 + self.y**2)

        else:
            f1 = op(self.z**2, self.y**2) * 10
            f2 = np.sqrt(self.z**2 + self.y**2) * 10

        return pd.Series({"f1": f1, "f2": f2})


class TestPymooAPI:
    def test_duplicates(self):
        parameters = [
            Parameter(name="x", interval=(-2, 10)),
            Parameter(name="y", interval=(-2, 10)),
            Parameter(name="x", interval=(-2, 12)),
        ]

        with pytest.raises(ValueError) as exec_info:
            check_duplicate_params(parameters)

        assert "Duplicate parameter name: x" in str(exec_info.value)

    def test_single_evaluator(self):
        parameters = [
            Parameter(name="x_par", interval=(0, 5), model_property="x"),
            Parameter(name="y_par", interval=(0, 3), model_property="y"),
        ]

        pymoo_ev = PymooModelEvaluator(parameters, PyRosen())

        problem = RealContinuousProblem(
            parameters=parameters,
            evaluators=[pymoo_ev],
            objective_ids=["f1"],
        )

        algorithm = DE(pop_size=100, sampling=LHS(), CR=0.3, jitter=False)
        res = minimize(problem, algorithm, seed=1, verbose=False)

        np.testing.assert_allclose(res.X, np.array([1, 1]), rtol=0.01)

    def test_two_evaluators_two_objectives(self):
        param = [
            Parameter(name="x_par", interval=(0, 5), model_property="x"),
            Parameter(name="y_par", interval=(0, 3), model_property="y"),
        ]

        eval1 = PymooModelEvaluator(param, BinAndKorn1())
        eval2 = PymooModelEvaluator(param, BinAndKorn2())

        problem = RealContinuousProblem(
            parameters=param,
            evaluators=[eval1, eval2],
            objective_ids=["f1", "f2"],
            constraint_ids=["g1", "g2"],
        )
        res = minimize(
            problem, NSGA2(pop_size=10), ("n_gen", 10), seed=1, verbose=False
        )

        assert res.F.shape[0] == 10
        assert res.F.shape[1] == 2
        assert np.all(res.G <= 1e-6)
        assert np.all(res.F >= 0)

    def test_mixed_problem(self):
        param = [
            Parameter(
                name="switch",
                values=(True, False),
                ptype="Binary",
                model_property="switch",
            ),
            Parameter(
                name="x",
                values=("add", "multiply"),
                ptype="Choice",
                model_property="operator",
            ),
            Parameter(name="y", interval=(-2, 2), ptype="Integer", model_property="y"),
            Parameter(name="z", interval=(-5.5, 5.5), ptype="Real", model_property="z"),
        ]

        pymoo_ev = PymooModelEvaluator(param, MixedProblemModel())

        problem = MixedProblem(
            parameters=param,
            evaluators=[pymoo_ev],
            objective_ids=["f1"],
        )

        algorithm = MixedVariableGA(pop_size=20)
        termination = get_termination("n_gen", 5)
        res = minimize(problem, algorithm, termination, seed=42, verbose=False)
        assert res.F.shape[0] == 1


class TestModelEvaluator:
    def test_model_evaluator(self):
        param_list = [
            Parameter(
                "par_x1",
                (-3.14159265359, 3.14159265359),
                init_value=1.5,
                model_property="x1",
            ),
            Parameter("par_x2", (-3.14159265359, 3.14159265359), model_property="x2"),
            Parameter(
                "par_x3",
                (-3.14159265359, 3.14159265359),
                relabs="Relative",
                model_property="x3",
            ),
        ]

        modev = ModelEvaluator(model=IshigamiDynamic(), parameters=param_list)

        assert modev.intervals == [
            (-3.14159265359, 3.14159265359),
            (-3.14159265359, 3.14159265359),
            (-3.14159265359, 3.14159265359),
        ]

        assert modev.get_initial_values() == [1.5, 2, 1.0]

        # 3rd relative parameter was just for test
        param_list[-1] = Parameter(
            "par_x3", (-3.14159265359, 3.14159265359), model_property="x3"
        )
        modev = ModelEvaluator(model=IshigamiDynamic(), parameters=param_list)
        res = modev.evaluate(
            [(param_list[0], -3.14 / 2), (param_list[1], 0), (param_list[2], 3.14)],
            indicators_configs=[("res", "mean")],
            simulation_options={
                "start": "2009-01-01 00:00:00",
                "end": "2009-01-01 00:00:00",
                "timestep": "h",
            },
        )

        pd.testing.assert_series_equal(res, pd.Series({"res": -10.721167816657914}))

        assert (
            round(
                modev.scipy_obj_function(
                    np.array([-3.14 / 2, 0, 3.14]),
                    ("res", "mean"),
                    {
                        "start": "2009-01-01 00:00:00",
                        "end": "2009-01-01 00:00:00",
                        "timestep": "h",
                    },
                    None,
                ),
                3,
            )
            == -10.721
        )

        # Static model test
        modev = ModelEvaluator(model=Ishigami(), parameters=param_list)

        res = modev.evaluate(
            [(param_list[0], -3.14 / 2), (param_list[1], 0), (param_list[2], 3.14)],
            indicators_configs=["res"],
        )

        pd.testing.assert_series_equal(res, pd.Series({"res": -10.721167816657914}))

        assert (
            modev.scipy_obj_function(np.array([-3.14 / 2, 0, 3.14]), "res", None, None)
            == -10.721167816657914
        )


class TestSciOptimizer:
    def test_sci_optimizer(self):
        param_list = [
            Parameter(
                "par_x1",
                (-3.14159265359, 3.14159265359),
                model_property="x1",
            ),
            Parameter("par_x2", (-3.14159265359, 3.14159265359), model_property="x2"),
            Parameter(
                "par_x3",
                (-3.14159265359, 3.14159265359),
                model_property="x3",
            ),
        ]
        # Dynamic optimization
        sci_opt = SciOptimizer(parameters=param_list, model=IshigamiDynamic())

        opt_res = sci_opt.diff_evo_minimize(
            indicator_config=("res", "mean"),
            simulation_options={
                "start": "2009-01-01 00:00:00",
                "end": "2009-01-01 00:00:00",
                "timestep": "h",
            },
            rng=42,
        )

        assert round(opt_res.fun, 4) == -10.7409

        # Static optimization
        sci_opt = SciOptimizer(parameters=param_list, model=Ishigami())

        opt_res = sci_opt.diff_evo_minimize(
            indicator_config="res",
            rng=42,
        )

        assert round(opt_res.fun, 4) == -10.7409

        # Scalar optimization
        parameter = Parameter("x_param", interval=(-10, 10), model_property="x")

        sci_opt = SciOptimizer(parameters=[parameter], model=X2())

        res = sci_opt.scalar_minimize("f_out")

        assert res.x == 0
