import pandas as pd
from SALib.sample import fast_sampler, saltelli, latin
from SALib.sample import morris as morris_sampler
from SALib.analyze import fast, morris, sobol, rbd_fast

import numpy as np

METHOD_SAMPLER_DICT = {
    "FAST": {
        "method": fast,
        "sampling": fast_sampler,
    },
    "Morris": {"method": morris, "sampling": morris_sampler},
    "Sobol": {
        "method": sobol,
        "sampling": saltelli,
    },
    "RBD_fast": {
        "method": rbd_fast,
        "sampling": latin,
    },
}


class SAnalysis:
    def __init__(self, model, parameters_list: [dict], method: str):
        if method not in METHOD_SAMPLER_DICT.keys():
            raise ValueError("Specified sensitivity method is not valid")
        else:
            self.method = method
        self.parameters_list = parameters_list
        self._salib_problem = None
        self.set_parameters_list(parameters_list)
        self.sample = None
        self.model = model

    def set_parameters_list(self, parameters_list: list):
        self._salib_problem = {
            "num_vars": len(parameters_list),
            "names": [p["name"] for p in parameters_list],
            "bounds": list(map(lambda p: p["interval"], parameters_list)),
        }

    def draw_sample(self, n:int, sampling_kwargs:dict=None):
        if sampling_kwargs is None:
            sampling_kwargs = {}

        sampler = METHOD_SAMPLER_DICT[self.method]["sampling"]
        sample_temp = sampler.sample(
            N=n, problem=self._salib_problem, **sampling_kwargs
        )

        for index, param in enumerate(self.parameters_list):
            vtype = param["type"]
            if vtype == "Integer":
                sample_temp[:, index] = np.round(sample_temp[:, index])
                sample_temp = np.unique(sample_temp, axis=0)

        self.sample = pd.DataFrame(sample_temp, columns=self._salib_problem["names"])
