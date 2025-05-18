
import pyomo.environ as pyo
from idaes.core import FlowsheetBlock, MaterialBalanceType
from idaes.models.unit_models import Heater, Valve, Separator
from property_packages.build_package import build_package
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

m.fs.heater = Heater(
    property_package=m.fs.milk_properties,
    has_pressure_change=False,
)

m.fs.heater.inlet.flow_mol[0].fix(1)
m.fs.heater.inlet.temperature[0].fix(300)
m.fs.heater.inlet.pressure[0].fix(101325)
m.fs.heater.inlet.mole_frac_comp[0, "water"].fix(0.9)
m.fs.heater.inlet.mole_frac_comp[0, "milk_solid"].fix(0.1)

opt = pyo.SolverFactory("ipopt")
opt.options["max_iter"] = 1000
result = opt.solve(m, tee=True)


default_values = to_json(m.fs.heater,fname="example_json_heater.json")


