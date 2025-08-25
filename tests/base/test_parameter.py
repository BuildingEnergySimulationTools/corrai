import pytest
from corrai.base.parameter import Parameter


VALID_INTERVAL = (0, 10)
VALID_VALUES = [1, 2, 3]


class TestParameter:
    def test_parameter_with_interval(self):
        param = Parameter(
            name="x",
            model_property="m.x",
            interval=(0, 5),
        )
        assert param.name == "x"
        assert param.interval == (0, 5)

    def test_parameter_with_values(self):
        param = Parameter(
            name="mode",
            model_property="m.mode",
            values=("A", "B"),
            ptype="Choice",
            init_value="A",
        )
        assert all([val in param.values for val in param.init_value])

    def test_error_both_interval_and_values(self):
        with pytest.raises(
            ValueError, match="Only one of 'interval' or 'values' may be specified"
        ):
            Parameter(
                name="bad",
                model_property="m.bad",
                interval=(0, 1),
                values=(0, 1),
                ptype="Binary",
            )

    def test_error_neither_interval_nor_values(self):
        with pytest.raises(
            ValueError, match="One of 'interval' or 'values' must be specified"
        ):
            Parameter(name="bad", model_property="m.bad", ptype="Real")

    def test_invalid_ptype(self):
        with pytest.raises(ValueError, match="Invalid type:"):
            Parameter(
                name="bad", model_property="m.bad", interval=(1, 2), ptype="WrongType"
            )

    def test_invalid_relabs(self):
        with pytest.raises(ValueError, match="Invalid relabs:"):
            Parameter(
                name="bad",
                model_property="m.bad",
                interval=(1, 2),
                relabs="NotAValidMode",
            )

    def test_init_value_out_of_interval(self):
        with pytest.raises(ValueError, match="init_value .* not in interval"):
            Parameter(name="x", model_property="m.x", interval=(0, 5), init_value=10)

    def test_init_value_not_in_values(self):
        with pytest.raises(ValueError, match="init_value .* not in values"):
            Parameter(
                name="choice",
                model_property="m.choice",
                values=("A", "B", "C"),
                init_value="D",
            )
