from abc import ABC, abstractmethod


class Ocr(ABC):
    def __init__(self, name):
        self.name = name
        self.model = None

    @abstractmethod
    def init_model(
        self,
    ):
        pass

    @abstractmethod
    def process_image(
        self,
    ):
        pass

    @abstractmethod
    def run_benchmark(
        self,
    ):
        pass
