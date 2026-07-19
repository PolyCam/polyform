'''
File: convert.py
Polycam Inc.

Created by Chris Heinrich on Tuesday, 1st November 2022
Copyright © 2022 Polycam Inc. All rights reserved.
'''
import fire
from polyform.utils.logging import logger
from polyform.core.capture_folder import CaptureFolder
from polyform.convertors.instant_ngp import InstantNGPConvertor
from polyform.convertors.colmap import COLMAPConvertor


def convert(data_folder_path: str, format: str = "ingp", shared_camera: bool = False):
    """
    Main entry point for the command line convertor
    Args:
        data_folder_path: path to the unzipped Polycam data folder
        format: Output format time. Supported values are [ingp, colmap]
        shared_camera: (colmap format only) if True, write a single shared COLMAP
            camera for all images instead of one camera per image
    """
    folder = CaptureFolder(data_folder_path)
    if format.lower() == "ingp" or format.lower() == "instant-ngp":
        convertor = InstantNGPConvertor()
    elif format.lower() == "colmap":
        convertor = COLMAPConvertor(shared_camera=shared_camera)
    else:
        logger.error("Format {} is not curently supported. Consider adding a convertor for it".format(format))
        exit(1)
    convertor.convert(folder)

if __name__ == '__main__':
    fire.Fire(convert)
