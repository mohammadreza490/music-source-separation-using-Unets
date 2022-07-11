import os
import sys

from pathlib import Path


class Config_Handler:
    """
    This utility class is used to handle the configuration of the project and
    paths to folders of the project are stored in the class fields which will be used throughout the project
    Run the init function using the appropriate path before starting the project .
    """

    __PROJECT_FOLDER_PATH = None
    __PATH_TO_DATASET_WAVS = None
    __PATH_TO_DATASET_ZIPPED = None
    __PATH_TO_MODELS = None
    __PATH_TO_DATASET_STEMS = None
    __PATH_TO_TRAIN_DATA_DIR = None
    __PATH_TO_TEST_DATA_DIR = None
    __PATH_TO_TEMP_DATA_DIR = None
    __PATH_TO_TEMP_SPECTROGRAMS_FOR_TRAINING_DIR = None
    __ZIP_FILE_URL = "https://zenodo.org/record/3338373/files/musdb18hq.zip?download=1"

    # These static methods act as getters of the class field
    @staticmethod
    def PROJECT_FOLDER_PATH():
        return Config_Handler.__PROJECT_FOLDER_PATH

    @staticmethod
    def PATH_TO_DATASET_WAVS():
        return Config_Handler.__PATH_TO_DATASET_WAVS

    @staticmethod
    def PATH_TO_DATASET_ZIPPED():
        return Config_Handler.__PATH_TO_DATASET_ZIPPED

    @staticmethod
    def PATH_TO_MODELS():
        return Config_Handler.__PATH_TO_MODELS

    @staticmethod
    def PATH_TO_DATASET_STEMS():
        return Config_Handler.__PATH_TO_DATASET_STEMS

    @staticmethod
    def PATH_TO_TRAIN_DATA_DIR():
        return Config_Handler.__PATH_TO_TRAIN_DATA_DIR

    @staticmethod
    def PATH_TO_TEST_DATA_DIR():
        return Config_Handler.__PATH_TO_TEST_DATA_DIR

    @staticmethod
    def PATH_TO_TEMP_DATA_DIR():
        return Config_Handler.__PATH_TO_TEMP_DATA_DIR

    @staticmethod
    def PATH_TO_TEMP_SPECTROGRAMS_FOR_TRAINING_DIR():
        return Config_Handler.__PATH_TO_TEMP_SPECTROGRAMS_FOR_TRAINING_DIR

    @staticmethod
    def ZIP_FILE_URL():
        return Config_Handler.__ZIP_FILE_URL

    @staticmethod
    def __setup_project_folder_path(project_folder_path: str):
        """
        Checks whether the project folder path is valid and sets the class field

        Parameters
        ----------
        project_folder_path : str

        Returns
        -------
        None

        """

        Config_Handler.__PROJECT_FOLDER_PATH = project_folder_path
        Config_Handler.__PATH_TO_DATASET_WAVS = os.path.join(
            Config_Handler.__PROJECT_FOLDER_PATH, "Dataset", "dataset-wav"
        )
        Config_Handler.__PATH_TO_DATASET_ZIPPED = os.path.join(
            Config_Handler.__PROJECT_FOLDER_PATH, "Dataset", "musdb18.zip"
        )
        Config_Handler.__PATH_TO_MODELS = os.path.join(
            Config_Handler.__PROJECT_FOLDER_PATH, "Models"
        )
        Config_Handler.__PATH_TO_DATASET_STEMS = os.path.join(
            Config_Handler.__PROJECT_FOLDER_PATH, "Dataset", "musdb18"
        )
        Config_Handler.__PATH_TO_TRAIN_DATA_DIR = os.path.join(
            Config_Handler.__PATH_TO_DATASET_WAVS, "train"
        )
        Config_Handler.__PATH_TO_TEST_DATA_DIR = os.path.join(
            Config_Handler.__PATH_TO_DATASET_WAVS, "test"
        )
        Config_Handler.__PATH_TO_TEMP_DATA_DIR = os.path.join(
            Config_Handler.__PATH_TO_DATASET_WAVS, "temp"
        )
        Config_Handler.__PATH_TO_TEMP_SPECTROGRAMS_FOR_TRAINING_DIR = os.path.join(
            Config_Handler.__PATH_TO_TEMP_DATA_DIR, "model_spec"
        )

    @staticmethod
    def make_dir(dir_path: str):

        """
        Creates a directory if it does not exist.

        Parameters
        ----------
        dir_path : str

        Returns
        -------
        None

        Raises
        ------
        None
        """

        if os.path.exists(dir_path):
            print(f"{dir_path} already exists")
        else:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            print(f"directory created at {dir_path}")

    @staticmethod
    def init(project_folder_path: str):
        """
        Initializes the project folder path and creates the necessary folders.

        Parameters
        ----------
        project_folder_path : str

        Returns
        -------
        None

        Raises
        ------
        AssertionError

        """
        assert (
            project_folder_path != None
            and isinstance(project_folder_path, str)
            and len(project_folder_path.replace(" ", "")) != 0
        ), "Path should be a non-empty string!"
        assert os.path.exists(
            project_folder_path
        ), "Project folder path does not exist or file path is not formatted properly!"
        Config_Handler.__setup_project_folder_path(project_folder_path)
        Config_Handler.make_dir(Config_Handler.PATH_TO_MODELS())
        src_dir_path = os.path.join(Config_Handler.PROJECT_FOLDER_PATH(), "src")
        if src_dir_path not in sys.path:
            sys.path.append(
                src_dir_path
            )  # appends the src dir path to the sys path so that the modules can be imported
