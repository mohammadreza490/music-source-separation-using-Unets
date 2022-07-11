import os
from random import sample
import zipfile
import subprocess
from printer import Printer
from config_handler import Config_Handler
from model_handler import Model_Blueprint, Model_Handler
from distutils.dir_util import copy_tree
import soundfile as sf
import numpy as np
import requests
import shutil

from wav_file_handler import Wav_File_Handler


class Dataset_Handler:
    """
    Instances of this class are used to perform operations on the dataset such as:
    - loading the data
    - storing the data
    - removing the data
    - extracting the zipped dataset
    - converting the stems to wav files
    - etc
    """

    def __extract_zip_file(self) -> None:
        """
        Extracts the content of the dataset zip file if the unzipped file does not exist.
        If the zipfile does not exist, it will download the dataset from zip file URL and then unzip it.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Raises
        ------
        None

        """

        if os.path.exists(Config_Handler.PATH_TO_DATASET_STEMS()):
            print(
                f"extracted stem files already exists at {Config_Handler.PATH_TO_DATASET_STEMS()}"
            )
        else:
            if not os.path.exists(
                Config_Handler.PATH_TO_DATASET_ZIPPED
            ):  # if the zip file does not exist
                zip = requests.get(
                    Config_Handler.ZIP_FILE_URL()
                )  # download the zip file
                zip = zipfile.ZipFile(zip)
            else:
                zip = zipfile.ZipFile(Config_Handler.PATH_TO_DATASET_ZIPPED)

            zip.extractall(
                Config_Handler.PATH_TO_DATASET_STEMS()
            )  # extract the zip file to the dataset directory (but in stem format)
            print(
                f"file extracted successfully at {Config_Handler.PATH_TO_DATASET_STEMS()}"
            )
            zip.close()

    def __remove_extra_files(self, files_to_keep_names: list) -> None:

        """
        This method removes all the files that are not in the `files_to_keep_names list`.
        It is used to remove the sources that are not needed for the model training (such as bass and drums)

        Parameters
        ----------
        files_to_keep_names: list
                list of files that should be kept

        Returns
        -------
        None

        Raises
        ------
        None

        """

        number_of_files_removed = 0
        print("removing extra files...")
        for directory in [
            "train",
            "test",
        ]:  # for each directory, remove the files that are not in the `files_to_keep_names list`
            for parent, _, files in os.walk(
                os.path.join(Config_Handler.PATH_TO_DATASET_WAVS(), directory)
            ):
                for f in files:
                    if f not in files_to_keep_names:
                        number_of_files_removed += 1
                        path_to_remove = os.path.join(parent, f)
                        os.remove(path_to_remove)

        if number_of_files_removed:
            print(f"{number_of_files_removed} files removed")
        else:
            print("found no extra files to remove!")

    def __convert_stems_to_wavs(self) -> None:

        """
        Converts the original stem files to wav files using musdbconvert if the files are not already converted.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Raises
        ------
        None

        """

        if self.__check_wav_dataset_exists():
            print(
                f"converted files already exist at {Config_Handler.PATH_TO_DATASET_WAVS()}"
            )
        else:
            print("converting stem files to wav files...")
            # run the musdbconvert command to convert the stems in stem directory and store the wav files in wav directory
            subprocess.run(
                [
                    "musdbconvert",
                    Config_Handler.PATH_TO_DATASET_STEMS(),
                    Config_Handler.PATH_TO_DATASET_WAVS(),
                ]
            )
            print("conversion completed!")

    def __load_data_from_zip(self, files_to_keep_names: list) -> None:

        """
        this method loads the data from the zip file (if not already extracted), converts the stems to wav files (if not already converted) and removes the extra files

        Parameters
        ----------
        None

        Returns
        -------
        None

        Raises
        ------
        None

        """
        if os.path.exists(Config_Handler.PATH_TO_DATASET_WAVS()):
            print(
                f"converted data already exists at {Config_Handler.PATH_TO_DATASET_WAVS()}"
            )
        else:
            self.__extract_zip_file()
            self.__convert_stems_to_wavs()
        self.__remove_extra_files(files_to_keep_names=files_to_keep_names)

    def __check_wav_dataset_exists(self) -> bool:

        """
        This method checks if the data exists in the wav dataset path

        Parameters
        ----------
        None

        Returns
        -------
        bool

        Raises
        ------
        None

        """

        return os.path.exists(Config_Handler.PATH_TO_DATASET_WAVS())

    def __check_temp_dir_train_spectrogram_exists(self) -> bool:

        """
        This method checks if the temp directory for the train spectrograms exists

        Parameters
        ----------
        None

        Returns
        -------
        bool

        Raises
        ------
        None
        """

        return os.path.exists(Config_Handler.PATH_TO_TEMP_DIR_TRAIN_SPECTROGRAM)

    def __create_train_data_dir(self) -> None:
        """
        This method creates a directory at `Config_Handler.PATH_TO_TEMP_SPECTROGRAMS_FOR_TRAINING_DIR()` for storing spectrograms of training songs' segments.
        The data will be used for generating training data of the models

        Parameters
        ----------
        None

        Returns
        -------
        None

        Raises
        ------
        None
        """
        if os.path.exists(Config_Handler.PATH_TO_TEMP_SPECTROGRAMS_FOR_TRAINING_DIR()):
            print(
                f"{Config_Handler.PATH_TO_TEMP_SPECTROGRAMS_FOR_TRAINING_DIR()} exists!"
            )
            return

        music_names = os.listdir(
            Config_Handler.PATH_TO_TRAIN_DATA_DIR()
        )  # get the music names of the training directory

        counter = 0
        for music_name in music_names:

            counter += 1

            music_dir_path = os.path.join(
                Config_Handler.PATH_TO_TRAIN_DATA_DIR(), music_name
            )
            spectrogram_dict = {"available_spec_ids": []}
            destination_dir = os.path.join(
                Config_Handler.PATH_TO_TEMP_SPECTROGRAMS_FOR_TRAINING_DIR(), music_name
            )

            if os.path.exists(
                os.path.join(destination_dir, music_name)
            ):  # if the directory for this music name already exists, skip it
                continue

            if not os.path.exists(destination_dir):
                Config_Handler.make_dir(
                    destination_dir
                )  # create the directory for this music name at destination_dir

            for file_name in ["mixture.wav", "vocals.wav", "accompaniment.wav"]:

                wav_file_handler = Wav_File_Handler(
                    audio_path=os.path.join(music_dir_path, file_name)
                )
                _, spectrograms = wav_file_handler.generate_segments_and_spectrograms(
                    segment_length_in_seconds=Model_Blueprint.SEGMENT_LENGTH_IN_SECONDS
                )  # generate the spectrograms for wav files of each file_name

                for i, spectrogram in enumerate(
                    spectrograms
                ):  # generates an id for each of the segment spectrograms and add it to the spectrogram_dict

                    spec_id = str(i)

                    if spec_id not in spectrogram_dict.keys():
                        spectrogram_dict[spec_id] = {}

                    if spec_id not in spectrogram_dict["available_spec_ids"]:
                        spectrogram_dict["available_spec_ids"].append(spec_id)

                    spectrogram_dict[spec_id][
                        file_name[:-4]
                    ] = spectrogram  # add the spectrogram to the spectrogram_dict and removes the .wav extension from the file name

                """the dict being stored has the following format:
					{"available_spec_ids": [list of available ids], 
					"0": {
						"mixture": [mixture data for segments id 0],
						"vocals": [vocals data for segment id 0],
						"accompaniment": [accompaniemnt data for segment id 0]
					}, 
					"1": {same as 0 but for id 1} ...}
			"""
            destination_path = os.path.join(destination_dir, f"spectrograms.npy")
            np.save(
                destination_path, spectrogram_dict
            )  # save dict which contains vocals, mixture and accompaniment keys and their respective spectrograms

        print(f"stored spectrograms for {counter} songs successfully!")

    def store_new_wav(
        self, data: np.array, path: os.path, sample_rate: int = 44100
    ) -> None:
        """
        This method stores a new wav file in specified path with the specified sample rate.

        Parameters
        ----------
        data: np.array
                the data to store in the wav file
        path: os.path.path
                the path to store the wav file
        sample_rate: int
                the sample rate of the wav file

        Returns
        -------
        None

        Raises
        ------
        Exception
        AssertionError
        """

        assert (
            isinstance(sample_rate, int) and sample_rate > 0
        ), "sample_rate must be a positive integer number!"
        assert (
            isinstance(data, np.ndarray)
            and len(data) > 0
            and np.all(isinstance(d, int) or isinstance(d, float) for d in data)
        ), "data must be a non-empty numpy array with int or float values"
        assert (
            isinstance(path, str)
            and len(path.replace(" ", "")) > 0
            and path.replace(" ", "").isnumeric() == False
        ), "path must be a non empty string and in valid format"

        if ".wav" not in path:  # adds the .wav extension if it is not already there
            path += ".wav"
        try:
            sf.write(path, data, sample_rate)
        except:
            raise Exception(
                f"Error occured while storing new wav file to {path}. Check the path and try again!"
            )

    def load_data(
        self, files_to_keep_names=["vocals.wav", "accompaniment.wav", "mixture.wav"]
    ) -> None:

        """
        This method handles loading the wav files into specified path in the Config_Handler.

        Parameters
        ----------
        files_to_keep_names: list
                list of files that should be kept

        Returns
        -------
        None

        Raises
        ------
        AssertionError
        """
        assert isinstance(
            files_to_keep_names, list
        ), "files_to_keep_names must be a list!"
        assert (
            len(files_to_keep_names) > 0
        ), f"files_to_keep_names must be a list with at least one element. received: {len(files_to_keep_names)} elements"
        assert all(isinstance(n, str) for n in files_to_keep_names) and all(
            len(n.replace(" ", "")) > 0 for n in files_to_keep_names
        ), "files_to_keep_names must be a list of non empty strings"

        if self.__check_wav_dataset_exists():
            if not self.__check_temp_dir_train_spectrogram_exists():
                print("creating temp dir and storing training spectrograms...")
                self.__create_train_data_dir()

            print(f"data exists at {Config_Handler.PATH_TO_DATASET_WAVS()}")
        else:
            self.__load_data_from_zip(files_to_keep_names)
