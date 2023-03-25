from abc import ABC, abstractmethod


class BaseOptimizationResult(ABC):
    def __init__(self):
        self._success = None
        self._msg = None
        self._results

    def __repr__(self):
        return f"{self.__name__}()"

    @abstractmethod
    def package_results(self):
        pass

    @property
    def success(self):
        return self._success

    @property
    def result(self):
        if self._result is None:
            return self.package_results()
        else:
            return self._results


class LocalOptimizationResult(BaseOptimizationResult):
    def __init__(self):
        pass

    def package_results(self):
        pass


class GlobalOptimizationResult(BaseOptimizationResult):
    def __init__(self):
        pass

    def package_results(self):
        pass


class EMOptimizationResult(BaseOptimizationResult):
    def __init__(self):
        pass

    def package_results(self):
        pass
