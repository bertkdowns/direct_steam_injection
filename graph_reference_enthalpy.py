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
m.fs.milk_sb[0].mole_frac_comp["h2o"].fix(0.99999)
m.fs.milk_sb[0].mole_frac_comp["milk_solid"].fix(0.00001)

# Set flow
m.fs.milk_sb[0].flow_mol.fix(1)
m.fs.helm_sb[0].flow_mol.fix(1)

opt = pyo.SolverFactory("ipopt")

temperature = [280 + i for i in range(150)]
enthalpy_helm = []
enthalpy_milk = []

# Set both stateblocks to the same temperature and pressure
for temp in temperature:
    m.fs.milk_sb[0].temperature.fix(temp * pyo.units.K)
    m.fs.milk_sb[0].pressure.fix(101325)
    m.fs.helm_sb[0].enth_mol.fix(
        m.fs.steam_properties.htpx(p=101325 * pyo.units.Pa, T=temp * pyo.units.K)
    )
    m.fs.helm_sb[0].pressure.fix(101325)

    assert degrees_of_freedom(m.fs) == 0

    results = opt.solve(m, tee=False)
    assert results.solver.termination_condition == pyo.TerminationCondition.optimal

    enthalpy_milk.append(pyo.value(m.fs.milk_sb[0].enth_mol))
    enthalpy_helm.append(pyo.value(m.fs.helm_sb[0].enth_mol))


# Graph the results
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

plt.plot(temperature, enthalpy_milk, label="milk PP")
plt.plot(temperature, enthalpy_helm, label="steam PP")
plt.xlabel("Temperature (K)")
plt.ylabel("Enthalpy (J/mol)")
plt.title("Enthalpy comparison - milk pp vs helmholtz pp")
plt.legend()
plt.show()