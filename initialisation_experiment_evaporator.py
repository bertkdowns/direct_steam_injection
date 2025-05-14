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
from idaes.core.util.model_serializer import from_json, to_json
import time
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
m.fs.effect_1 = Heater(
    property_package=m.fs.milk_properties,
    has_pressure_change=False,
)
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
m.fs.flash_phase_separator_to_effect = Arc(
    source=m.fs.flash_phase_separator.outlet_1,
    destination=m.fs.effect_1.inlet,
)
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
# Add a constraint to fix the outlet pressure of the flash, which should calculate the valve coefficient.
@m.fs.Constraint()
def flash_pressure_constraint(fs):
    return fs.flash.outlet.pressure[0] == 75_000 # 75 kPa
m.fs.flash.Cv.unfix() # IDK why this is fixed by default.

m.fs.flash_phase_separator.split_fraction[0,"outlet_1", "Vap"].fix(0.02)
m.fs.flash_phase_separator.split_fraction[0,"outlet_1", "Liq"].fix(0.99)


default_values = to_json(m,return_dict=True)


HEAT_DUTY_VALUES = [0,1000,2000, 4000, 8000, 12000, 16000, 20000, 30000,60000]
TEMPERATURE_VALUES = [351.15, 353.15, 355.15, 357.15, 359.15, 361.15, 363.15, 365.15, 330.15,375.15]
time_results = []
iteration_results = []
solve_status = []
indexes = []

global start
global end



def setup():
    global start
    start = time.time()

def restore():
    from_json(m,sd=default_values)


def initialize():
    def init_unit(unit):
        print(f"Initializing unit {unit}")
        unit.initialize()#outlvl=idaeslog.DEBUG

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

def solve():
    assert degrees_of_freedom(m) == 0
    opt = Ipopt()
    opt.config.raise_exception_on_nonoptimal_result = False
    status = opt.solve(m, tee=False)
    global end
    global start
    end = time.time()
    time_results.append(end - start)
    iteration_results.append(status.iteration_count)
    solve_status.append(status.solution_status) 

for heat_duty in HEAT_DUTY_VALUES:
    restore()
    setup()
    m.fs.effect_1.heat_duty.fix(heat_duty)
    initialize()
    solve()
    indexes.append("heat duty of " + str(heat_duty) + " with initialisation")

for heat_duty in HEAT_DUTY_VALUES:
    setup()
    m.fs.effect_1.heat_duty.fix(heat_duty)
    solve()
    indexes.append("heat duty of " + str(heat_duty) + " from previous solve")

m.fs.effect_1.heat_duty.unfix()

for temperature in TEMPERATURE_VALUES:
    setup()
    m.fs.effect_1.outlet.temperature.fix(temperature)
    solve()
    indexes.append("temperature of " + str(temperature) +  " from previous solve")


for temperature in TEMPERATURE_VALUES:
    restore()
    setup()
    m.fs.effect_1.outlet.temperature.fix(temperature)
    initialize()
    solve()
    indexes.append("temperature of " + str(temperature) + " with initialisation")



for results in zip(indexes, time_results, iteration_results, solve_status):
    print(results)


# RESULTS:
# ('heat duty of 0 with initialisation', 1.0205872058868408, 54, <SolutionStatus.optimal: 30>)
# ('heat duty of 1000 with initialisation', 0.95733642578125, 41, <SolutionStatus.optimal: 30>)
# ('heat duty of 2000 with initialisation', 0.9359180927276611, 45, <SolutionStatus.optimal: 30>)
# ('heat duty of 4000 with initialisation', 0.9301490783691406, 37, <SolutionStatus.optimal: 30>)
# ('heat duty of 8000 with initialisation', 0.9101030826568604, 41, <SolutionStatus.optimal: 30>)
# ('heat duty of 12000 with initialisation', 0.9309337139129639, 49, <SolutionStatus.optimal: 30>)
# ('heat duty of 16000 with initialisation', 0.997211217880249, 43, <SolutionStatus.optimal: 30>)
# ('heat duty of 20000 with initialisation', 0.8305227756500244, 38, <SolutionStatus.optimal: 30>)
# ('heat duty of 30000 with initialisation', 0.8931007385253906, 34, <SolutionStatus.optimal: 30>)
# ('heat duty of 60000 with initialisation', 1.0755863189697266, 50, <SolutionStatus.optimal: 30>)
# ('heat duty of 0 from previous solve', 0.07674407958984375, 5, <SolutionStatus.optimal: 30>)
# ('heat duty of 1000 from previous solve', 0.06808257102966309, 3, <SolutionStatus.optimal: 30>)
# ('heat duty of 2000 from previous solve', 0.12989449501037598, 3, <SolutionStatus.optimal: 30>)
# ('heat duty of 4000 from previous solve', 0.1135873794555664, 3, <SolutionStatus.optimal: 30>)
# ('heat duty of 8000 from previous solve', 0.08837270736694336, 3, <SolutionStatus.optimal: 30>)
# ('heat duty of 12000 from previous solve', 0.07968568801879883, 3, <SolutionStatus.optimal: 30>)
# ('heat duty of 16000 from previous solve', 0.08147954940795898, 3, <SolutionStatus.optimal: 30>)
# ('heat duty of 20000 from previous solve', 0.07163739204406738, 3, <SolutionStatus.optimal: 30>)
# ('heat duty of 30000 from previous solve', 0.06998658180236816, 3, <SolutionStatus.optimal: 30>)
# ('heat duty of 60000 from previous solve', 0.06609487533569336, 3, <SolutionStatus.optimal: 30>)
# ('temperature of 351.15 from previous solve', 0.07333040237426758, 4, <SolutionStatus.optimal: 30>)
# ('temperature of 353.15 from previous solve', 0.06981587409973145, 3, <SolutionStatus.optimal: 30>)
# ('temperature of 355.15 from previous solve', 0.06970381736755371, 3, <SolutionStatus.optimal: 30>)
# ('temperature of 357.15 from previous solve', 0.06816983222961426, 3, <SolutionStatus.optimal: 30>)
# ('temperature of 359.15 from previous solve', 0.07016396522521973, 3, <SolutionStatus.optimal: 30>)
# ('temperature of 361.15 from previous solve', 0.06571102142333984, 3, <SolutionStatus.optimal: 30>)
# ('temperature of 363.15 from previous solve', 0.05827641487121582, 3, <SolutionStatus.optimal: 30>)
# ('temperature of 365.15 from previous solve', 0.05720949172973633, 3, <SolutionStatus.optimal: 30>)
# ('temperature of 330.15 from previous solve', 0.056287527084350586, 3, <SolutionStatus.optimal: 30>)
# ('temperature of 375.15 from previous solve', 0.05764889717102051, 5, <SolutionStatus.optimal: 30>)
# ('temperature of 351.15 with initialisation', 1.3332476615905762, 59, <SolutionStatus.infeasible: 10>)
# ('temperature of 353.15 with initialisation', 1.3413028717041016, 40, <SolutionStatus.infeasible: 10>)
# ('temperature of 355.15 with initialisation', 1.2737398147583008, 39, <SolutionStatus.infeasible: 10>)
# ('temperature of 357.15 with initialisation', 1.333902359008789, 43, <SolutionStatus.infeasible: 10>)
# ('temperature of 359.15 with initialisation', 1.2751681804656982, 47, <SolutionStatus.infeasible: 10>)
# ('temperature of 361.15 with initialisation', 1.3430328369140625, 51, <SolutionStatus.infeasible: 10>)
# ('temperature of 363.15 with initialisation', 1.2962055206298828, 53, <SolutionStatus.infeasible: 10>)
# ('temperature of 365.15 with initialisation', 1.364783763885498, 49, <SolutionStatus.infeasible: 10>)
# ('temperature of 330.15 with initialisation', 1.23244047164917, 42, <SolutionStatus.infeasible: 10>)
# ('temperature of 375.15 with initialisation', 1.3908917903900146, 140, <SolutionStatus.infeasible: 10>)