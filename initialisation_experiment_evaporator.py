from direct_steam_injection import Dsi
import pytest
import pyomo.environ as pyo
from pyomo.network import Arc
from pyomo.network import SequentialDecomposition, Port
from idaes.core import FlowsheetBlock, MaterialBalanceType
from idaes.models.unit_models import Heater, Valve, Separator
from idaes.models.properties import iapws95
from idaes.core.util.initialization import propagate_state
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util import DiagnosticsToolbox
from idaes.models.unit_models.separator import SplittingType
from property_packages.build_package import build_package
from direct_steam_injection import Dsi
from translator import GenericTranslator
import idaes.logger as idaeslog
# New solver interface: http://pyomo.readthedocs.io/en/6.8.0/developer_reference/solvers.html
from pyomo.contrib.solver.util import assert_optimal_termination, SolutionStatus, TerminationCondition
from pyomo.contrib.solver.ipopt import Ipopt

# Build the model
m = pyo.ConcreteModel()
m.fs = FlowsheetBlock(dynamic=False)
m.fs.steam_properties = build_package("helmholtz",["water"],["Vap","Liq"])
m.fs.milk_properties = build_package("milk",["water","milk_solid"],["Vap","Liq"])


m.fs.dsi = Dsi(
    property_package=m.fs.milk_properties,
    steam_property_package=m.fs.steam_properties
)
m.fs.flash = Valve(
    property_package=m.fs.milk_properties,
)
m.fs.flash_phase_separator = Separator(
    property_package=m.fs.milk_properties,
    split_basis=SplittingType.phaseFlow
)
# m.fs.effect_1 = Heater(
#     property_package=m.fs.properties,
#     has_pressure_change=False,
# )
# m.fs.effect_phase_separator = Separator(
#     property_package=m.fs.properties,
#     splitting_type=SplittingType.phaseFlow
# )
# m.fs.translator = GenericTranslator(
#     inlet_property_package=m.fs.milk_properties,
#     outlet_property_package=m.fs.steam_properties,
# )

# Link them up
m.fs.dsi_to_flash = Arc(source=m.fs.dsi.outlet, destination=m.fs.flash.inlet)
m.fs.flash_to_phase_separator = Arc(
    source=m.fs.flash.outlet, destination=m.fs.flash_phase_separator.inlet
)
# m.fs.flash_phase_separator_to_effect = Arc(
#     source=m.fs.flash_phase_separator.outlet_1,
#     destination=m.fs.effect_1.inlet,
# )
# m.fs.effect_to_phase_separator = Arc(
#     source=m.fs.effect_1.outlet, destination=m.fs.effect_phase_separator.inlet
# )
# m.fs.effect_phase_separator_to_translator = Arc(
#     source=m.fs.effect_phase_separator.outlet_2,
#     destination=m.fs.translator.inlet,
# )

pyo.TransformationFactory("network.expand_arcs").apply_to(m)


# Specify the properties
m.fs.dsi.inlet.flow_mol.fix(50)
m.fs.dsi.inlet.temperature.fix(351.15) # 78.0 C
m.fs.dsi.inlet.pressure.fix(90000) # 90 kPa
m.fs.dsi.inlet.mole_frac_comp[0, "water"].fix(0.95)
m.fs.dsi.inlet.mole_frac_comp[0, "milk_solid"].fix(0.05)


m.fs.dsi.steam_inlet.pressure.fix(1_000_000) # 10 bar
m.fs.dsi.properties_steam_in[0].constrain_component(m.fs.dsi.properties_steam_in[0].temperature, 458.15) # 185 C
m.fs.dsi.outlet.temperature.fix(368.15) # 95 C, this is used to calculate the flowrate of the steam_inlet.

m.fs.flash.valve_opening.fix(1)
# Add a constraint to fix the outlet pressure of the flash, which should calcualte the valve coefficient.
@m.fs.Constraint()
def flash_pressure_constraint(fs):
    return fs.flash.outlet.pressure[0] == 75_000 # 75 kPa
m.fs.flash.Cv.unfix() # IDK why this is fixed by default.



m.fs.flash_phase_separator.split_fraction[0,"outlet_1", "Vap"].fix(0.99)
m.fs.flash_phase_separator.split_fraction[0,"outlet_1", "Liq"].fix(0.01)



# Initialize the model


def init_unit(unit):
    print(f"Initializing unit {unit}")
    unit.initialize(outlvl=idaeslog.DEBUG)

# Use SequentialDecomposition to initialise the model
seq = SequentialDecomposition()
# use create_graph to get the order of sequential decomposition, and also to
# find any units that are not connected to the sequential decomposition
seq.set_tear_set([]) # No tears required.
G = seq.create_graph(m)
order = seq.calculation_order(G)
seq_blocks = []
for o in order:
    seq_blocks.append(o[0])
print("Order of initialisation:", [blk.name for blk in seq_blocks])
seq.options["tol"] = 1e-2
# seq.options["solve_tears"] = False
res = seq.run(m, init_unit)

# Solve the model
print("Degrees of freedom:", degrees_of_freedom(m))
print("Degrees of freedom in dsi:", degrees_of_freedom(m.fs.dsi))
print("Degrees of freedom in flash:", degrees_of_freedom(m.fs.flash))
print("Degrees of freedom in flash_phase_separator:", degrees_of_freedom(m.fs.flash_phase_separator))

assert degrees_of_freedom(m) == 0
opt = Ipopt()
status = opt.solve(m, tee=True)
assert_optimal_termination(status)
status.display()
print(status.iteration_count)
print(status.timing_info.wall_time)
print(status.termination_condition)
print(status.solution_status)