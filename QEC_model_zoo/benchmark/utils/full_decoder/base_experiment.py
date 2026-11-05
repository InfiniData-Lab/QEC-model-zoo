from abc import ABC, abstractmethod


class BaseExperiment(ABC):
    @abstractmethod
    def run(self) -> None:
        """
        Run the experiment.
        """
        pass
