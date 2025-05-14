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


HEAT_DUTY_VALUES = [0,1000,2000, 4000, 8000, 12000, 16000, 20000, 30000,4000]
TEMPERATURE_VALUES = [351.15, 353.15, 355.15, 357.15, 359.15, 361.15, 363.15, 365.15, 330.15]
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
# ('heat duty of 0 with initialisation', 0.9865739345550537, 54, <SolutionStatus.optimal: 30>)
# ('heat duty of 1000 with initialisation', 1.3341705799102783, 41, <SolutionStatus.optimal: 30>)
# ('heat duty of 2000 with initialisation', 2.3008036613464355, 45, <SolutionStatus.optimal: 30>)
# ('heat duty of 4000 with initialisation', 1.2961623668670654, 37, <SolutionStatus.optimal: 30>)
# ('heat duty of 8000 with initialisation', 1.3963162899017334, 41, <SolutionStatus.optimal: 30>)
# ('heat duty of 12000 with initialisation', 1.2948670387268066, 49, <SolutionStatus.optimal: 30>)
# ('heat duty of 16000 with initialisation', 1.1932551860809326, 43, <SolutionStatus.optimal: 30>)
# ('heat duty of 20000 with initialisation', 1.2919673919677734, 38, <SolutionStatus.optimal: 30>)
# ('heat duty of 30000 with initialisation', 1.2608609199523926, 34, <SolutionStatus.optimal: 30>)
# ('heat duty of 4000 with initialisation', 1.2732441425323486, 37, <SolutionStatus.optimal: 30>)
# ('heat duty of 0 from previous solve', 0.1572403907775879, 4, <SolutionStatus.optimal: 30>)
# ('heat duty of 1000 from previous solve', 0.09160971641540527, 3, <SolutionStatus.optimal: 30>)
# ('heat duty of 2000 from previous solve', 0.11079621315002441, 3, <SolutionStatus.optimal: 30>)
# ('heat duty of 4000 from previous solve', 0.10798478126525879, 3, <SolutionStatus.optimal: 30>)
# ('heat duty of 8000 from previous solve', 0.07496857643127441, 3, <SolutionStatus.optimal: 30>)
# ('heat duty of 12000 from previous solve', 0.06638979911804199, 3, <SolutionStatus.optimal: 30>)
# ('heat duty of 16000 from previous solve', 0.061025142669677734, 3, <SolutionStatus.optimal: 30>)
# ('heat duty of 20000 from previous solve', 0.05940508842468262, 3, <SolutionStatus.optimal: 30>)
# ('heat duty of 30000 from previous solve', 0.05750107765197754, 3, <SolutionStatus.optimal: 30>)
# ('heat duty of 4000 from previous solve', 0.0580286979675293, 4, <SolutionStatus.optimal: 30>)
# ('temperature of 351.15 from previous solve', 0.05791807174682617, 3, <SolutionStatus.optimal: 30>)
# ('temperature of 353.15 from previous solve', 0.05661654472351074, 3, <SolutionStatus.optimal: 30>)
# ('temperature of 355.15 from previous solve', 0.05655193328857422, 3, <SolutionStatus.optimal: 30>)
# ('temperature of 357.15 from previous solve', 0.056191205978393555, 3, <SolutionStatus.optimal: 30>)
# ('temperature of 359.15 from previous solve', 0.05602288246154785, 3, <SolutionStatus.optimal: 30>)
# ('temperature of 361.15 from previous solve', 0.056138038635253906, 3, <SolutionStatus.optimal: 30>)
# ('temperature of 363.15 from previous solve', 0.055830955505371094, 3, <SolutionStatus.optimal: 30>)
# ('temperature of 365.15 from previous solve', 0.08025932312011719, 3, <SolutionStatus.optimal: 30>)
# ('temperature of 330.15 from previous solve', 0.05438590049743652, 3, <SolutionStatus.optimal: 30>)
# ('temperature of 351.15 with initialisation', 1.3014867305755615, 59, <SolutionStatus.infeasible: 10>)
# ('temperature of 353.15 with initialisation', 1.2377779483795166, 40, <SolutionStatus.infeasible: 10>)
# ('temperature of 355.15 with initialisation', 1.3015146255493164, 39, <SolutionStatus.infeasible: 10>)
# ('temperature of 357.15 with initialisation', 1.325639009475708, 43, <SolutionStatus.infeasible: 10>)
# ('temperature of 359.15 with initialisation', 1.315643310546875, 47, <SolutionStatus.infeasible: 10>)
# ('temperature of 361.15 with initialisation', 1.4199199676513672, 51, <SolutionStatus.infeasible: 10>)
# ('temperature of 363.15 with initialisation', 1.403618574142456, 53, <SolutionStatus.infeasible: 10>)
# ('temperature of 365.15 with initialisation', 1.429168939590454, 49, <SolutionStatus.infeasible: 10>)
# ('temperature of 330.15 with initialisation', 1.2610621452331543, 42, <SolutionStatus.infeasible: 10>)