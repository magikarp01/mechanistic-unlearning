from abc import ABC, abstractmethod
from masks.masks import AbstractMask
from tasks.task import Task

class AbstractLocalizer(ABC):
    @abstractmethod
    def get_mask(self, model, task: Task) -> AbstractMask:
        pass
