from direct_steam_injection import Dsi
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
        state_vars= StateVars.PH
    )
m.fs.milk_properties = GenericParameterBlock(**milk_configuration)

m.fs.milk_sb = m.fs.milk_properties.build_state_block(
    m.fs.time,
    defined_state=True,
    doc="Stateblock for the inlet stream",
)
m.fs.helm_sb = m.fs.steam_properties.build_state_block(
    m.fs.time,
    defined_state=True,
    doc="Stateblock for the inlet stream",
)

# Make the milk sb pure water
m.fs.milk_sb[0].mole_frac_comp["h2o"].fix(0.9999999)
m.fs.milk_sb[0].mole_frac_comp["milk_solid"].fix(0.0000001)

# Set flow
m.fs.milk_sb[0].flow_mol.fix(1)
m.fs.helm_sb[0].flow_mol.fix(1)

# Set both stateblocks to the same temperature and pressure
m.fs.milk_sb[0].temperature.fix(300 * pyo.units.K)
m.fs.milk_sb[0].pressure.fix(101325)
m.fs.helm_sb[0].enth_mol.fix(
    m.fs.steam_properties.htpx(p=101325 * pyo.units.Pa, T= 300 * pyo.units.K)
)
m.fs.helm_sb[0].pressure.fix(101325)


print(degrees_of_freedom(m.fs.milk_sb[0]))
print(degrees_of_freedom(m.fs.helm_sb[0]))
assert degrees_of_freedom(m.fs.milk_sb) == 0
assert degrees_of_freedom(m.fs.helm_sb) == 0
assert degrees_of_freedom(m.fs) == 0

opt = pyo.SolverFactory("ipopt")
results = opt.solve(m, tee=False)
assert results.solver.termination_condition == pyo.TerminationCondition.optimal
#m.fs.display()

# Calculate the enthalpy of the milk stateblock
print("milk enth_mol",pyo.value( m.fs.milk_sb[0].enth_mol))
print("milk vapor fraction",pyo.value( m.fs.milk_sb[0].phase_frac["Vap"]))

print("helm enth_mol",pyo.value( m.fs.helm_sb[0].enth_mol))
print("helm vapor fraction",pyo.value( m.fs.helm_sb[0].vapor_frac))

opt = pyo.SolverFactory("ipopt")
results = opt.solve(m, tee=False)
assert results.solver.termination_condition == pyo.TerminationCondition.optimal

