# Import Pyomo libraries
from pyomo.environ import (
    Var,
    Suffix,
    units as pyunits,
)
from pyomo.common.config import ConfigBlock, ConfigValue, In
from idaes.core.util.tables import create_stream_table_dataframe
from idaes.core.util.exceptions import ConfigurationError
from idaes.models.unit_models.translator import TranslatorData
# Import IDAES cores
from idaes.core import (
    declare_process_block_class,
    UnitModelBlockData,
    useDefault,
)
from idaes.core.util.config import is_physical_parameter_block
import idaes.core.util.scaling as iscale
import idaes.logger as idaeslog

# Set up logger
_log = idaeslog.getLogger(__name__)


# When using this file the name "GenericTranslator" is what is imported
@declare_process_block_class("GenericTranslator")
class GenericTranslatorData(TranslatorData):
    """
    GenericTranslator.

    This is used to translate between two different property packages, and supports dropping compounds that are not present.

    For example, if you have a stream of water/milk, and it's almost all water, this allows you to translate the stream to a water-only stream.

    It works by fixing the temperature and pressure, and flow of each component in the outlet stream to be the same as the inlet stream.
     
    """

    def build(self):
        super().build()

        # Pressure (= inlet pressure)
        @self.Constraint(
            self.flowsheet().time,
            doc="Pressure balance",
        )
        def eq_outlet_pressure(b, t):
            return b.properties_in[t].pressure == b.properties_out[t].pressure

        # Temperature (= inlet temperature)
        @self.Constraint(
            self.flowsheet().time,
            doc="Equivalent temperature balance",
        )
        def eq_outlet_temperature(b, t):
            return (
                b.properties_in[t].temperature == b.properties_out[t].temperature
            )
        
        # Flow = mixed flow
        @self.Constraint(
            self.flowsheet().time,
            self.config.outlet_property_package.component_list,
            doc="Mass balance for the outlet",
        )
        def eq_outlet_composition(b, t, c):
            return 0 == sum(
                b.properties_out[t].get_material_flow_terms(p, c)
                - b.properties_in[t].get_material_flow_terms(p, c)
                for p in b.properties_out[t].phase_list
                if (p, c) in b.properties_out[t].phase_component_set
            )  # handle the case where a component is not in that phase (e.g no milk vapor)
