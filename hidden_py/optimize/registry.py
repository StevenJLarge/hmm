from hidden_py.optimize.base import OptClass
from hidden_py.optimize import optimization

OPTIMIZER_REGISTRY = {
    OptClass.Local: optimization.LocalLikelihoodOptimizer,
    OptClass.Global: optimization.GlobalLikelihoodOptimizer,
    OptClass.ExpMax: optimization.EMOptimizer
}
