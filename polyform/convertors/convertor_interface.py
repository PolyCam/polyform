from abc import ABC, abstractmethod
from polyform.core.capture_folder import CaptureFolder


class ConvertorInterface(ABC):
    @abstractmethod
    def convert(self, folder: CaptureFolder, output_path: str = ""):
        """
        All data convertors must implement the covnert method
        """
        pass
