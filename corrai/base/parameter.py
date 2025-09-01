from dataclasses import dataclass

TYPES = ["Integer", "Real", "Choice", "Binary"]
RELABS = ["Absolute", "Relative"]


@dataclass
class Parameter:
    name: str
    interval: tuple[int | float, int | float] | None = None
    values: tuple[str | int | float, ...] | None = None
    ptype: str = "Real"
    relabs: str = "Absolute"
    init_value: str | int | float | tuple[str | int | float] | None = None
    min_max_interval: (
        tuple[int | float, int | float] | list[tuple[int | float, int | float]] | None
    ) = None
    model_property: str | tuple[str, ...] = None

    """
    A parameter definition for models. Can Affect a single model property or a list
    of properties

    The parameter can be either defined over a continuous interval (e.g., for real or
    integer parameters) or as a discrete set of possible values (e.g., for categorical
    or binary parameters).
    Exactly one of `interval` or `values` must be provided.

    The `Parameter` class also supports specifying metadata such as type,
    relative/absolute scaling, an initial value, and optional min/max constraints.

    Parameters
    ----------
    name : str
        The name of the parameter.

    model_property : str or list of str
        The model property (or list of property) that this parameter is associated with.
        Usually, a property is a "path" to a value in a model.
        (eg : building.wall.insulation.conductivity)

    interval : tuple of int or float, optional
        A tuple representing the lower and upper bounds of the parameter's domain,
        used for continuous or integer parameters. Must not be provided if `values`
        is specified.

    values : list of str, int, or float, optional
        A list of discrete candidate values the parameter can take. Must not be provided
        if `interval` is specified.

    ptype : str, default="Real"
        The type of parameter.
        Must be one of `"Integer"`, `"Real"`, `"Choice"`, or `"Binary"`.

    relabs : str, default="Absolute"
        Specifies whether the parameter should be treated in `"Relative"`
        or `"Absolute"` terms. Use Relative if the parameter values or interval
        are specified as a percentage of the model property values

    init_value : str, int, or float, optional
        The initial value of the parameter. If `interval` is used, must fall within
        the interval. If `values` is used, must be one of the listed values.

    min_max_interval : tuple of int or float, optional
        Optional min and max bounds used for some checking operations.

    Raises
    ------
    ValueError
        If both `interval` and `values` are specified, or if neither is specified.
        If `init_value` is outside the specified domain.
        If `ptype` or `relabs` are not in the allowed sets.

    Examples
    --------
    >>> # Example using an interval
    >>> p = Parameter(
    ...     name="Conductivity",
    ...     model_property=(
    ...         "building.wall.insulation.conductivity",
    ...         "building.roof.insulation.conductivity",
    ...     ),
    ...     interval=(0.032, 0.040),
    ...     ptype="Real",
    ...     init_value=(0.035, 0.039)
    ... )

    >>> # Example using discrete values
    >>> p = Parameter(
    ...     name="ThermalCapacity",
    ...     model_property="simulation.algorithm",
    ...     values=("TARP", "DOE"),
    ...     ptype="Choice",
    ...     init_value="TARP"
    ... )
    """

    def __post_init__(self):
        if self.interval is not None and self.values is not None:
            raise ValueError("Only one of 'interval' or 'values' may be specified.")
        if self.interval is None and self.values is None and self.ptype != "Binary":
            raise ValueError("One of 'interval' or 'values' must be specified.")

        if self.ptype not in TYPES:
            raise ValueError(f"Invalid type: {self.ptype!r}. Must be one of {TYPES}.")

        if isinstance(self.relabs, str) and self.relabs not in RELABS:
            raise ValueError(
                f"Invalid relabs: {self.relabs!r}. "
                f"Must be one of {RELABS} or a bool."
            )

        if self.init_value is not None:
            list_to_test = (
                self.init_value
                if isinstance(self.init_value, list)
                else [self.init_value]
            )
            if self.interval is not None and self.relabs != "Relative":
                lo, hi = self.interval
                if not all(lo <= i_value <= hi for i_value in list_to_test):
                    raise ValueError(
                        f"init_value {self.init_value} "
                        f"not in interval {self.interval}"
                    )
            elif self.values is not None and self.relabs != "Relative":
                if not all(i_value in self.values for i_value in list_to_test):
                    raise ValueError(
                        f"init_value {self.init_value} " f"not in values {self.values}"
                    )
            self.init_value = list_to_test
