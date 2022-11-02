import fire
from polyform.core.capture_folder import CaptureFolder
from polyform.convertors.instant_ngp import InstantNGPConvertor


def convert(data_folder_path: str):
    folder = CaptureFolder(data_folder_path)
    convertor = InstantNGPConvertor()
    convertor.convert(folder)

if __name__ == '__main__':
    fire.Fire(convert)