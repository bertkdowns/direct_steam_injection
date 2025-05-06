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
    )
from idaes.models.properties.modular_properties import GenericParameterBlock
from milk_config import milk_configuration


m = pyo.ConcreteModel()
m.fs = FlowsheetBlock(dynamic=False)
m.fs.steam_properties = HelmholtzParameterBlock(
        pure_component="h2o", amount_basis=AmountBasis.MOLE,
        phase_presentation=PhaseType.LG,
    )
m.fs.milk_properties = GenericParameterBlock(**milk_configuration)
m.fs.dsi = Dsi(property_package=m.fs.milk_properties,steam_property_package=m.fs.steam_properties)

m.fs.dsi.inlet.flow_mol.fix(1)
m.fs.dsi.properties_milk_in[0].temperature.fix( 300 * pyo.units.K)
m.fs.dsi.inlet.pressure.fix(101325)
m.fs.dsi.inlet.mole_frac_comp[0,"h2o"].fix(0.99)
m.fs.dsi.inlet.mole_frac_comp[0,"milk_solid"].fix(0.01)

m.fs.dsi.steam_inlet.flow_mol.fix(1)
m.fs.dsi.properties_steam_in[0].enth_mol.fix(
    m.fs.steam_properties.htpx(p=101325 * pyo.units.Pa, T= 300 * pyo.units.K)
)
m.fs.dsi.steam_inlet.pressure.fix(101325)

m.fs.dsi.initialize()
m.fs.dsi.display()
assert degrees_of_freedom(m.fs.dsi.properties_milk_in) == 0
assert degrees_of_freedom(m.fs.dsi.properties_steam_in) == 0
assert degrees_of_freedom(m.fs) == 0

opt = pyo.SolverFactory("ipopt")
results = opt.solve(m, tee=True)
m.fs.dsi.display()

assert results.solver.termination_condition == pyo.TerminationCondition.optimal
assert degrees_of_freedom(m.fs) == 0


m.fs.display()
dsi = m.fs.dsi
enth_mol_phase = dsi.properties_out[0.0].enth_mol_phase
mole_frac_phase_comp = dsi.properties_out[0.0].mole_frac_phase_comp

print("milk_in")
print("Flow (mol):", pyo.value(dsi.properties_milk_in[0.0].flow_mol))
print("Temperature (K):", pyo.value(dsi.properties_milk_in[0.0].temperature))
print("Pressure (Pa):", pyo.value(dsi.properties_milk_in[0.0].pressure))
print("Enthalpy (mol):", pyo.value(dsi.properties_milk_in[0.0].enth_mol))
print("Enthalpy (mol, Liquid Phase):", pyo.value(dsi.properties_milk_in[0.0].enth_mol_phase["Liq"]))
print("Enthalpy (mol, Vapor Phase):", pyo.value(dsi.properties_milk_in[0.0].enth_mol_phase["Vap"]))
print("Mole Fraction (Liquid Phase, Water):", pyo.value(dsi.properties_milk_in[0.0].mole_frac_phase_comp["Liq", "h2o"]))
print("Mole Fraction (Vapor Phase, Water):", pyo.value(dsi.properties_milk_in[0.0].mole_frac_phase_comp["Vap", "h2o"]))
print("Mole Fraction (Liquid Phase, Milk):", pyo.value(dsi.properties_milk_in[0.0].mole_frac_phase_comp["Liq", "milk_solid"]))
print("steam_in")
print("Flow (mol):", pyo.value(dsi.properties_steam_in[0.0].flow_mol))
print("Temperature (K):", pyo.value(dsi.properties_steam_in[0.0].temperature))
print("Pressure (Pa):", pyo.value(dsi.properties_steam_in[0.0].pressure))
print("Enthalpy (mol):", pyo.value(dsi.properties_steam_in[0.0].enth_mol))
print("Enthalpy (mol, Liquid Phase):", pyo.value(dsi.properties_steam_in[0.0].enth_mol_phase["Liq"]))
print("Enthalpy (mol, Vapor Phase):", pyo.value(dsi.properties_steam_in[0.0].enth_mol_phase["Vap"]))
print("Mole Fraction (Liquid Phase, Water):", pyo.value(dsi.properties_steam_in[0.0].mole_frac_phase_comp["Liq", "h2o"]))
print("Mole Fraction (Vapor Phase, Water):", pyo.value(dsi.properties_steam_in[0.0].mole_frac_phase_comp["Vap", "h2o"]))
print("Vapor Fraction:", pyo.value(dsi.properties_steam_in[0.0].vapor_frac))
print("steam_cooled")
print("Flow (mol):", pyo.value(dsi.properties_steam_cooled[0.0].flow_mol))
print("Temperature (K):", pyo.value(dsi.properties_steam_cooled[0.0].temperature))
print("Pressure (Pa):", pyo.value(dsi.properties_steam_cooled[0.0].pressure))
print("Enthalpy (mol):", pyo.value(dsi.properties_steam_cooled[0.0].enth_mol))
print("Enthalpy (mol, Liquid Phase):", pyo.value(dsi.properties_steam_cooled[0.0].enth_mol_phase["Liq"]))
print("Enthalpy (mol, Vapor Phase):", pyo.value(dsi.properties_steam_cooled[0.0].enth_mol_phase["Vap"]))
print("Mole Fraction (Liquid Phase, Water):", pyo.value(dsi.properties_steam_cooled[0.0].mole_frac_phase_comp["Liq", "h2o"]))
print("Mole Fraction (Vapor Phase, Water):", pyo.value(dsi.properties_steam_cooled[0.0].mole_frac_phase_comp["Vap", "h2o"]))
print("Vapor Fraction:", pyo.value(dsi.properties_steam_in[0.0].vapor_frac))
print("steam_delta_h")
print("Delta H:", pyo.value(dsi.steam_delta_h[0.0]))
print("mixed_unheated")
print("Flow (mol):", pyo.value(dsi.properties_mixed_unheated[0.0].flow_mol))
print("Temperature (K):", pyo.value(dsi.properties_mixed_unheated[0.0].temperature))
print("Pressure (Pa):", pyo.value(dsi.properties_mixed_unheated[0.0].pressure))
print("Enthalpy (mol):", pyo.value(dsi.properties_mixed_unheated[0.0].enth_mol))
print("Enthalpy (mol, Liquid Phase):", pyo.value(dsi.properties_mixed_unheated[0.0].enth_mol_phase["Liq"]))
print("Enthalpy (mol, Vapor Phase):", pyo.value(dsi.properties_mixed_unheated[0.0].enth_mol_phase["Vap"]))
print("Mole Fraction (Liquid Phase, Water):", pyo.value(dsi.properties_mixed_unheated[0.0].mole_frac_phase_comp["Liq", "h2o"]))
print("Mole Fraction (Vapor Phase, Water):", pyo.value(dsi.properties_mixed_unheated[0.0].mole_frac_phase_comp["Vap", "h2o"]))
print("Mole Fraction (Liquid Phase, Milk):", pyo.value(dsi.properties_mixed_unheated[0.0].mole_frac_phase_comp["Liq", "milk_solid"]))
print("output")
print("Flow (mol):", pyo.value(dsi.properties_out[0.0].flow_mol))
print("Temperature (K):", pyo.value(dsi.properties_out[0.0].temperature))
print("Pressure (Pa):", pyo.value(dsi.properties_out[0.0].pressure))
print("Enthalpy (mol):", pyo.value(dsi.properties_out[0.0].enth_mol))
print("Enthalpy (mol, Liquid Phase):", pyo.value(dsi.properties_out[0.0].enth_mol_phase["Liq"]))
print("Enthalpy (mol, Vapor Phase):", pyo.value(dsi.properties_out[0.0].enth_mol_phase["Vap"]))
print("Mole Fraction (Liquid Phase, Water):", pyo.value(dsi.properties_out[0.0].mole_frac_phase_comp["Liq", "h2o"]))
print("Mole Fraction (Vapor Phase, Water):", pyo.value(dsi.properties_out[0.0].mole_frac_phase_comp["Vap", "h2o"]))
print("Mole Fraction (Liquid Phase, Milk):", pyo.value(dsi.properties_out[0.0].mole_frac_phase_comp["Liq", "milk_solid"]))

dt = DiagnosticsToolbox(dsi)
dt.report_structural_issues()
dt.display_components_with_inconsistent_units()
dt.display_underconstrained_set()
dt.display_overconstrained_set()
dt.display_potential_evaluation_errors()