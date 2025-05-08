from translator import GenericTranslator
import pytest
import pyomo.environ as pyo
from pyomo.network import Arc
from idaes.core import FlowsheetBlock, MaterialBalanceType
from idaes.models.unit_models import Heater, Valve
from idaes.models.properties import iapws95
from idaes.core.util.initialization import propagate_state
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util import DiagnosticsToolbox
from idaes.models.properties.general_helmholtz import (
        HelmholtzParameterBlock,
        HelmholtzThermoExpressions,
        AmountBasis,
        PhaseType,
        StateVars
    )
from idaes.models.properties.modular_properties import GenericParameterBlock
from milk_config import milk_configuration



m = pyo.ConcreteModel()
m.fs = FlowsheetBlock(dynamic=False)
m.fs.steam_properties = HelmholtzParameterBlock(
        pure_component="h2o", amount_basis=AmountBasis.MOLE,
        phase_presentation=PhaseType.LG,
        state_vars=StateVars.PH,
    )
m.fs.milk_properties = GenericParameterBlock(**milk_configuration)
m.fs.translator = GenericTranslator(inlet_property_package=m.fs.milk_properties, 
                                    outlet_property_package=m.fs.steam_properties,
                                    outlet_state_defined=True)

m.fs.translator.inlet.flow_mol.fix(1)
m.fs.translator.inlet.temperature.fix(300 * pyo.units.K)
m.fs.translator.inlet.pressure.fix(101325)
m.fs.translator.inlet.mole_frac_comp[0,"h2o"].fix(0.99)
m.fs.translator.inlet.mole_frac_comp[0,"milk_solid"].fix(0.01)

m.fs.translator.display()
print(degrees_of_freedom(m.fs.translator.properties_in))
print(degrees_of_freedom(m.fs.translator.properties_out))

print(degrees_of_freedom(m.fs.translator))
assert degrees_of_freedom(m.fs) == 0

opt = pyo.SolverFactory("ipopt")
results = opt.solve(m, tee=True)
m.fs.translator.display()

assert results.solver.termination_condition == pyo.TerminationCondition.optimal
assert degrees_of_freedom(m.fs) == 0

m.fs.display()

dt = DiagnosticsToolbox(m.fs.translator)
dt.report_structural_issues()
dt.display_components_with_inconsistent_units()
dt.display_underconstrained_set()
dt.display_overconstrained_set()
dt.display_potential_evaluation_errors()