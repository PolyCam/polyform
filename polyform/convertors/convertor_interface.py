'''
File: convertor_interface.py
Polycam Inc.

Created by Chris Heinrich on Tuesday, 1st November 2022
Copyright © 2022 Polycam Inc. All rights reserved.
'''
from abc import ABC, abstractmethod
from polyform.core.capture_folder import CaptureFolder


class ConvertorInterface(ABC):
    @abstractmethod
    def convert(self, folder: CaptureFolder, output_path: str = ""):
        """
        All data convertors must implement the covnert method
        """
        pass
