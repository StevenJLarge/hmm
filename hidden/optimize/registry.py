from optimize.base import OptClass
from optimize import optimization

OPTIMIZER_REGISTRY = {
    OptClass.Local: optimization.LocalLikelihoodOptimizer,
    OptClass.Global: optimization.GlobalLikelihoodOptimizer,
}
