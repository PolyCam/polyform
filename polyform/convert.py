import fire
from polyform.utils.logging import logger
from polyform.core.capture_folder import CaptureFolder
from polyform.convertors.instant_ngp import InstantNGPConvertor


def convert(data_folder_path: str, format: str = "ingp"):
    """
    Main entry point for the command line convertor
    Args:
        data_folder_path: path to the unzipped Polycam data folder
        format: Output format time. Supported values are [ingp]
    """
    folder = CaptureFolder(data_folder_path)
    if format.lower() == "ingp" or format.lower() == "instant-ngp":
        convertor = InstantNGPConvertor()
    else:
        logger.error("Format {} is not curently supported. Consider adding a convertor for it".format(format))
        exit(1)
    convertor.convert(folder)

if __name__ == '__main__':
    fire.Fire(convert)