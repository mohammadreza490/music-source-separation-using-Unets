from abc import ABC, abstractmethod
from doctest import FAIL_FAST
from inspect import signature
import random
import numpy as np
from rsa import sign
import tensorflow as tf
from tensorflow.keras.models import Model
from wav_file_handler import Wav_File_Handler
from config_handler import Config_Handler
from visualiser import Visualiser
from model_architecture_builder import Model_Architecture_Builder
import tensorflow_addons as tfa
from tensorflow_addons.optimizers import AdamW
from tensorflow.keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt
from typing import Callable, Generator, Iterator
import os
import scipy as sp
import shutil


class Model_Blueprint(ABC):
    """
    Abstract class that defines the basic structure of a model_handler object.
    """

    # input shape has a default value of (1025, 173, 1) which cannot be changed (as the whole network architecture has to change for different shapes to work)
    INPUT_SHAPE = (1025, 173, 1)
    # to create that input_shape, the length of music segmetns should be 2.0 seconds
    SEGMENT_LENGTH_IN_SECONDS = 2.0

    def __init__(self, model_name: str):

        assert (
            model_name != None
            and isinstance(model_name, str)
            and len(model_name.replace(" ", "")) > 0
        ), "model_name must be a non-empty string"

        self._model_name = model_name

        # setting up required dirs and paths for the model
        self._model_dir = os.path.join(
            Config_Handler.PATH_TO_MODELS(), self._model_name
        )
        Config_Handler.make_dir(self._model_dir)
        self._trained_model_path = os.path.join(
            self._model_dir, f"{self._model_name}-trained-model"
        )
        Config_Handler.make_dir(self._trained_model_path)
        self._checkpoint_dir = os.path.join(self._model_dir, "checkpoint")
        Config_Handler.make_dir(self._checkpoint_dir)
        self._checkpoint_path = os.path.join(
            self._checkpoint_dir, f"{self._model_name}-checkpoint"
        )
        self._tensorboard_logs_dir = os.path.join(self._model_dir, "tensorboard_logs")
        Config_Handler.make_dir(self._tensorboard_logs_dir)

        # either the trained model, model from the last checkpoint or a new model will be loaded
        self._model: tf.keras.Model = (
            self.load_model()
            if os.path.exists(self._trained_model_path)
            and len(os.listdir(self._trained_model_path)) != 0
            else self.load_checkpoint()
            if os.path.exists(self._checkpoint_path)
            and len(os.listdir(self._checkpoint_path)) != 0
            else self._create_model()
        )

    def _create_model(self) -> tf.keras.Model:
        """
        Creates the model architecture using `Model_Architecture_Builder.create_default_model()`

        Returns:
          tf.keras.Model
        """
        return Model_Architecture_Builder.create_default_model()

    def load_model(self) -> tf.keras.Model:
        """
        Loads the pre-trained model from the `self.trained_model_path`.

        Returns:
          tf.keras.Model

        Raises:
        ------
        Exception - if there is an error loading the model

        """
        print("Loading the pre-trained model!")
        # compile=False doesn't ask for custom loss functions source: https://stackoverflow.com/a/69791056/12828249
        try:
            return tf.keras.models.load_model(self._trained_model_path, compile=False)
        except Exception as e:
            raise Exception(e)

    def load_checkpoint(self) -> tf.keras.Model:
        """
        creates a new model architecture and tells user that the actual model parameters will be loaded from the `self._checkpoint_path` during training

        Returns:
          tf.keras.Model

        """
        print(
            f"Last checkpoint will be loaded from {self._checkpoint_path} during training"
        )
        return self._create_model()

    def _finalise_training(self):
        """
        This method removes the checkpoints from `self._checkpoint_dir` after
        training is complete so that a trained model doesn't have any checkpoints dir
        and saves the final model in `self._trained_model_path`
        """

        self._model.save(self._trained_model_path, save_format="tf")
        if os.path.exists(self._checkpoint_dir):
            shutil.rmtree(self._checkpoint_dir)

    def get_model_summary(self):
        """
        This method returns the model summary

        Returns:
          str - model summary
        """
        return self._model.summary()

    @property
    def model(self):
        return self._model

    @property
    def tensorboard_logs_path(self):
        return self._tensorboards_logs_path

    @property
    def model_name(self):
        return self._model_name

    @property
    def model_dir(self):
        return self._model_dir

    @property
    def checkpoint_dir(self):
        return self._checkpoint_dir

    @property
    def checkpoint_path(self):
        return self._checkpoint_path

    @property
    def trained_model_path(self):
        return self._trained_model_path

    @property
    def model_train_data_path(self):
        return Config_Handler.PATH_TO_TEMP_SPECTROGRAMS_FOR_TRAINING_DIR()

    @property
    def segment_length_in_seconds(self):
        return Model_Blueprint.SEGMENT_LENGTH_IN_SECONDS

    def _generate_data(self, batch_size=8):
        """
        This method defines the default data generator for the model.
        It is used to generate the data for training and generates batches on the fly.
        It uses the spectrograms stored at `Config_Handler.PATH_TO_TEMP_SPECTROGRAMS_FOR_TRAINING_DIR()`.
        The musics are itertated in sequence and batches are segments of data from start of the song to finish (no randomness and shuffling)

        Parameters:
        ----------
        batch_size (int): number of samples in each batch

        Generates:
        ----------
        tuple - (X, y) - X is the batch of mixture spectrograms and y is the batch of true seperated sources (vocals and accompaniment). Shape of X is (number_of_samples_in_batch, 1025, 173, 1) and shape of y is (number_of_samples_in_batch, 1025, 173, 2, 1)

        Raises:
        ------
        IOError - if there is an error reading the spectrogram array from `Config_Handler.PATH_TO_TEMP_SPECTROGRAMS_FOR_TRAINING_DIR()`

        """
        available_musics = os.listdir(
            Config_Handler.PATH_TO_TEMP_SPECTROGRAMS_FOR_TRAINING_DIR()
        )
        music_dict = {}
        # creates a dict for each music and stores a list of available spectrogram ids for each music (ids refer to spectrogram of segments)
        for music_name in available_musics:
            number_of_available_segments = Wav_File_Handler(
                audio_path=os.path.join(
                    Config_Handler.PATH_TO_TRAIN_DATA_DIR(), music_name, "mixture.wav"
                )
            ).get_number_of_possible_segments()
            music_dict[music_name] = {
                "available_spectrograms_ids": [
                    str(spec_id) for spec_id in range(number_of_available_segments)
                ]
            }

        batch_X = []
        batch_y = {"vocal_spectrograms": [], "ac_spectrograms": []}

        while len(available_musics) > 0:
            # reverseed because we want to remove the music from the list after we use it and reversing deletes the music from the end of the list therefore not affecting the loop order
            for music_name in reversed(available_musics):

                spec_dir_path = os.path.join(
                    Config_Handler.PATH_TO_TEMP_SPECTROGRAMS_FOR_TRAINING_DIR(),
                    music_name,
                )
                try:
                    specs = np.load(
                        os.path.join(spec_dir_path, "spectrograms.npy"),
                        allow_pickle=True,
                    ).item()
                except IOError as e:
                    raise IOError(e.strerror)

                while len(music_dict[music_name]["available_spectrograms_ids"]) > 0:

                    # depending on the number of ids left, the spectrograms in a batch will be different
                    if (
                        len(music_dict[music_name]["available_spectrograms_ids"])
                        < batch_size
                    ):
                        spectrogram_ids_to_select = music_dict[music_name][
                            "available_spectrograms_ids"
                        ]
                    else:
                        spectrogram_ids_to_select = music_dict[music_name][
                            "available_spectrograms_ids"
                        ][:batch_size]

                    segments_ids_to_select = [
                        spec_id for spec_id in spectrogram_ids_to_select
                    ]
                    music_dict[music_name]["available_spectrograms_ids"] = music_dict[
                        music_name
                    ]["available_spectrograms_ids"][batch_size:]
                    batch_X = np.array(
                        [
                            specs[spec_id]["mixture"]
                            for spec_id in segments_ids_to_select
                        ]
                    )
                    batch_y["vocal_spectrograms"] = np.array(
                        [specs[spec_id]["vocals"] for spec_id in segments_ids_to_select]
                    )
                    batch_y["ac_spectrograms"] = np.array(
                        [
                            specs[spec_id]["accompaniment"]
                            for spec_id in segments_ids_to_select
                        ]
                    )
                    X = np.array(batch_X)
                    y = np.array(
                        np.stack(
                            [batch_y["vocal_spectrograms"], batch_y["ac_spectrograms"]],
                            axis=-1,
                        )
                    )  # stacks the arrays along the last axis to create one array with last shape = 2 which is the number of sources to separate
                    X = tf.squeeze(X)
                    y = tf.squeeze(y)
                    X = tf.expand_dims(
                        X, -1
                    )  # this is for the input channel numbers (the input layer of cnn is has only one channel (look at the structure in the paper))
                    y = tf.expand_dims(y, -1)
                    if len(X.shape) == 3:
                        # if for example there is only one element (one spectrogram), we add a batch size of one at the beggining
                        X = tf.expand_dims(
                            X, 0
                        )  # this is for the input channel numbers (the input layer of cnn is has only one channel (look at the structure in the paper))
                        y = tf.expand_dims(y, 0)
                    batch_X = []
                    batch_y = {"vocal_spectrograms": [], "ac_spectrograms": []}

                    print(X.shape, y.shape)
                    yield (X, y)

                    # if the music is empty, we remove it from the list
                    if len(music_dict[music_name]["available_spectrograms_ids"]) == 0:
                        available_musics.remove(music_name)

                    # if there are no more musics, break the loop
                    if len(available_musics) == 0:
                        break

    @abstractmethod
    def train(
        self,
        loss_function: Callable,
        learning_rate_scheduler_callback: tf.keras.callbacks = None,
        learning_rate: float = 1e-3,
        epochs: int = 40,
        batch_size: int = 8,
        shuffle_data: bool = True,
        seed: int = 0,
    ) -> dict:
        pass

    @abstractmethod
    def _generate_pred_data(self, song_path: str, batch_size: int = 8):
        pass

    @abstractmethod
    def _predict(self, song_path: str, batch_size: int = 8):
        pass

    @abstractmethod
    def predict(self, path_to_song: str, visualise_graphs: bool = True) -> tuple:
        pass


class Model_Handler(Model_Blueprint):

    """
    This class handles all the operations related to model such as:
    - loading the model
    - saving the model
    - generating training data and training the model
    - generating testing data and testing the model
    - predicting sources for a song
    - etc
    """

    def __init__(self, model_name: str):
        super().__init__(model_name)

    def train(
        self,
        loss_function: Callable,
        data_generator: Callable = None,
        learning_rate_scheduler: tf.keras.callbacks = None,
        learning_rate: float = 1e-3,
        epochs: int = 40,
        batch_size: int = 8,
    ) -> tf.keras.callbacks.History():
        """
        This method is used to training the model and create backups after each epoch and at the end, stores the trained model.

        Parameters:
        -----------

        loss_function: Callable
            The loss function to use for training. it should accept two arguments: y_true and y_pred and return a value of the loss function.

        data_generator: Callable
            The data generator to use for training. if the default is provided, it should accept two args: a Model_Handler object and a batch_size:int. if not provided, the default data generator will be used.

        learning_rate_scheduler: tf.keras.callbacks (optional)
            The learning rate scheduler callback to change the learning rate during training. if not provided, the default learning rate will be used without any change.

        learning_rate: float (optional)
            The learning rate to use for training. if not provided, the default learning rate (1e-3) will be used.

        epochs: int (optional)
            The number of epochs to train the model. if not provided, the default number of epochs (40) will be used.

        batch_size: int (optional)
            The batch size to use for training. if not provided, the default batch size (8) will be used.

        Returns:
        --------
        tf.keras.callbacks.History

        Raises:
        -------
        None

        """
        # if the model is already trained, it asks the users if they want to train it for more epochs.
        assert (
            data_generator != None
            and isinstance(data_generator, Callable)
            and len(signature(data_generator).parameters) == 2
        ), "The data generator must be a callable with two arguments: a Model_Handler object and a batch_size:int"
        assert (
            isinstance(loss_function, Callable)
            and len(signature(loss_function).parameters) >= 2
        ), f"The loss function should be a Callable object with at least two arguments: y_true and y_pred. Found {len(signature(loss_function).parameters)} argument(s) and type {type(loss_function).__name__}."
        assert (
            isinstance(learning_rate, (int, float)) and learning_rate > 0
        ), f"The learning rate should be a positive number."
        assert (
            isinstance(epochs, int) and epochs > 0
        ), f"The number of epochs should be a positive integer."
        assert (
            isinstance(batch_size, int) and batch_size > 0
        ), f"The batch size should be a positive integer."
        assert data_generator == None or (
            isinstance(data_generator, Callable)
            and len(signature(data_generator).parameters) == 2
        ), f"The data generator should be a Callable object with two arguments."
        if (
            os.path.exists(self._trained_model_path)
            and len(os.listdir(self._trained_model_path)) != 0
        ):
            train_more = str.lower(
                input(
                    f"Model is already trained, are you sure you want to train it for {epochs} more epochs? \n (y for yes and n for no:)"
                )
            ).replace(" ", "")
            while train_more not in ["y", "n"]:
                print("invalid answer!")
                train_more = input(
                    f"Model is already trained, are you sure you want to train it for {epochs} more epochs? \n (y for yes and n for no:)"
                )
            if train_more == "n":
                print("cancelling training!")
                return
            else:
                print(f"model will be trained for {epochs} more epochs")

        # this is used to generate data either from a data_generator function or from the default data generator if no data_generator is provided
        train_dataset = tf.data.Dataset.from_generator(
            lambda: data_generator(self, batch_size)
            if data_generator
            else lambda: self._generate_data(batch_size=batch_size),
            output_signature=(
                tf.TensorSpec(
                    shape=(
                        None,
                        Model_Blueprint.INPUT_SHAPE[0],
                        Model_Blueprint.INPUT_SHAPE[1],
                        Model_Blueprint.INPUT_SHAPE[2],
                    ),
                    dtype=tf.float32,
                ),
                tf.TensorSpec(
                    shape=(
                        None,
                        Model_Blueprint.INPUT_SHAPE[0],
                        Model_Blueprint.INPUT_SHAPE[1],
                        2,
                        Model_Blueprint.INPUT_SHAPE[2],
                    ),
                    dtype=tf.float32,
                ),
            ),
        )

        optimizer = AdamW(weight_decay=1e-6, learning_rate=learning_rate)

        # used to store tensorboard logs
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=self._tensorboard_logs_dir,
            histogram_freq=0,
            write_graph=True,
            write_images=False,
            write_steps_per_second=False,
            update_freq="epoch",
            profile_batch=0,
            embeddings_freq=0,
            embeddings_metadata=None,
        )

        # user for backing up the data after each epoch
        backup_and_restore_callback = tf.keras.callbacks.BackupAndRestore(
            backup_dir=self._checkpoint_path
        )

        callbacks = (
            [tensorboard_callback, backup_and_restore_callback]
            + [tf.keras.callbacks.LearningRateScheduler(learning_rate_scheduler)]
            if learning_rate_scheduler != None
            else []
        )

        # compiled the model with the loss function and the optimizer provided
        self._model.compile(loss=loss_function, optimizer=optimizer)

        print("starting training!")
        history = self._model.fit(
            train_dataset,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )

        self._finalise_training()  # saves the model and delete the checkpoints after training is done

        return history

    def _generate_pred_data(
        self, song_data: np.array = [], song_path: str = None, batch_size: int = 8
    ):
        """
        This method generates the predicting X batches for a song.

        Parameters:
        -----------
        song_data: np.array (optional)
            The song data to use for generating the predicting X batches. if not provided, the song data will be loaded from the song path.

        song_path: str (optional)
            The path to the song to use for generating the predicting X batches.

        batch_size: int (optional)
            The batch size to use for generating the predicting X batches. if not provided, the default batch size (8) will be used.

        Generates:
        ----------
        np.array

        """

        wav_file = (
            Wav_File_Handler(song_data)
            if isinstance(song_data, np.ndarray) and len(song_data) > 0
            else Wav_File_Handler(audio_path=song_path)
        )  # create a wav file handler for the song either based on data or path
        _, X = wav_file.generate_segments_and_spectrograms(
            segment_length_in_seconds=Model_Blueprint.SEGMENT_LENGTH_IN_SECONDS
        )
        X = tf.expand_dims(X, -1)
        # the following part batches the data and yields the batches
        number_of_possible_batches = int(np.ceil(len(X) / batch_size))
        for batch_num in range(number_of_possible_batches):
            batch_X = X[batch_size * batch_num : batch_size * (batch_num + 1)]
            yield batch_X

    def _predict(
        self,
        song_data: np.array = None,
        song_path: str = None,
        batch_size: int = 8,
        verbose: bool = False,
    ) -> tuple:
        """
        This method gets a path to the song, generate the segments, predict the vocal and accompaniments for that segment.
        puts all the segments together and returns the predicted song (the segments all in one array)

        Parameters:
        -----------
        song_data: np.array (optional)
            The song data to use for generating the predicting X batches. if not provided, the song data will be loaded from the song path.

        song_path: str (optional)
            The path to the song to use for generating the predicting X batches.

        batch_size: int (optional)
            The batch size to use for generating the predicting X batches. if not provided, the default batch size (8) will be used.

        verbose: bool (optional)
            Whether to print more information for predicting the sources of the song. if not provided, the default value (False) will be used.

        Returns:
        --------
        tuple (np.array, np.array)
            The predicted vocals (first element) and accompaniments (second element) for the song.


        """
        if isinstance(song_data, np.ndarray) and not np.any(song_data) == None:
            song_data_generator = tf.data.Dataset.from_generator(
                lambda: self._generate_pred_data(
                    song_data=song_data, batch_size=batch_size
                ),
                output_signature=(
                    tf.TensorSpec(
                        shape=(
                            None,
                            Model_Blueprint.INPUT_SHAPE[0],
                            Model_Blueprint.INPUT_SHAPE[1],
                            1,
                        ),
                        dtype=tf.float64,
                    )
                ),
            )
        elif song_path != None and song_path.replace(" ", "") != "":
            song_data_generator = tf.data.Dataset.from_generator(
                lambda: self._generate_pred_data(
                    song_path=song_path, batch_size=batch_size
                ),
                output_signature=(
                    tf.TensorSpec(
                        shape=(
                            None,
                            Model_Blueprint.INPUT_SHAPE[0],
                            Model_Blueprint.INPUT_SHAPE[1],
                            1,
                        ),
                        dtype=tf.float64,
                    )
                ),
            )
        vocal_predicted = []
        accompaniment_predicted = []
        batch_counter = 0
        # for each batch generated from the song data generator, predicts the vocals and accompaniments and appends them to the lists to create prediction array for sources of the whole song
        for mixture_segment_batch in song_data_generator:
            batch_counter += 1
            if verbose:
                print(f"predicting batch {batch_counter}")
            prediction_batch = self._model.predict_on_batch(mixture_segment_batch)
            vocal_predicted_segment_batch, accompaniment_predicted_segment_batch = (
                prediction_batch[..., 0],
                prediction_batch[..., 1],
            )

            if verbose:
                print(vocal_predicted_segment_batch.shape)
            vocal_predicted.append(vocal_predicted_segment_batch)
            accompaniment_predicted.append(accompaniment_predicted_segment_batch)
            if verbose:
                print(f"- predicting of batch {batch_counter} completed!")

        return vocal_predicted, accompaniment_predicted

    def predict(
        self,
        song_data: np.array = None,
        song_path: str = None,
        visualise_graphs: bool = False,
        verbose: bool = False,
        batch_size: int = 8,
        sample_rate: int = 44100,
    ) -> tuple:
        """
        This method predicts the sources of the songs and visualises the graphs if the `visualise_graphs parameter` is True.

        Parameters:
        -----------
        song_data: np.array (optional)
            The song data to use for generating the predicting X batches. if not provided, the song data will be loaded from the song path.

        song_path: str (optional)
            The path to the song to use for generating the predicting X batches.

        visualise_graphs: bool (optional)
            Whether to visualise the graphs of the predicted sources. if not provided, the default value (False) will be used.

        verbose: bool (optional)
            Whether to print more information for predicting the sources of the song. if not provided, the default value (False) will be used.

        batch_size: int (optional)
            The batch size to use for generating the predicting X batches. if not provided, the default batch size (8) will be used.

        sample_rate: int (optional)
            The sample rate of the audio to separate the sources from. if not provided, the default value (44100) will be used.

        Notes
        -----
        either song_data or song_path must be provided. if both are not provided, this method will raise an error.
        if both are provided, song_data will be used.

        Returns
        -------
        tuple(predicted vocals Wave_File_Handler, predicted accompaniments Wave_File_Handler, original song Wave_File_Handler)

        Raises
        ------
        AssertionError
        """
        assert (
            (not song_data is None and song_path is None)
            or (song_data is None and not song_path is None)
            or (not song_data is None and not song_path is None)
        ), "either song_data or song_path must be provided. if both are not provided, this method will raise an error. if both are provided, song_data will be used."

        if not song_data is None:
            assert (
                isinstance(song_data, np.ndarray)
                and np.any(song_data) != None
                and len(song_data) > 0
                and song_data.dtype in (np.float16, np.float32, np.float64, float)
            ), "song_data must be a valid non-null np.array with type float!"
        if not song_path is None:
            assert (
                isinstance(song_path, str)
                and os.path.exists(song_path)
                and song_path[-4:] in [".wav", ".mp3", ".mp4"]
            ), "song_path must be a valid path to a song with a valid format (.mp3, .mp4 or .wav)!"
        assert (
            isinstance(batch_size, int) and batch_size > 0
        ), "batch_size must be a positive integer!"
        assert isinstance(visualise_graphs, bool), "visualise_graphs must be a boolean!"
        assert isinstance(verbose, bool), "verbose must be a boolean!"
        assert (
            isinstance(sample_rate, int) and sample_rate > 0
        ), "sample_rate must be a positive integer number!"

        v_predict, acc_predict = (
            self._predict(song_data, batch_size=batch_size, verbose=verbose)
            if isinstance(song_data, np.ndarray) and len(song_data) > 0
            else self._predict(
                song_data=None,
                song_path=song_path,
                batch_size=batch_size,
                verbose=verbose,
            )
        )  # uses either the song_path or song_data to predict the sources
        v_predict = [
            np.hstack(vocal) for vocal in v_predict
        ]  # merges the predicted vocals for each segment into a one dimensional array
        acc_predict = [
            np.hstack(acc) for acc in acc_predict
        ]  # merges the accompaniment vocals for each segment into one dimension
        wav_file = (
            Wav_File_Handler(song_data, sample_rate=sample_rate)
            if isinstance(song_data, np.ndarray) and len(song_data) > 0
            else Wav_File_Handler(audio_path=song_path, sample_rate=sample_rate)
        )
        _, spectrogram = wav_file.generate_segments_and_spectrograms(
            segment_length_in_seconds=Model_Blueprint.SEGMENT_LENGTH_IN_SECONDS
        )  # generates the spectrograms for the original song
        reconstructed_v_wav_file = wav_file.reconstruct_audio_from_spectrogram_using_inverse_stft(
            np.hstack(v_predict),
            segment_length_in_seconds=Model_Blueprint.SEGMENT_LENGTH_IN_SECONDS,
        )  # reconstructs the vocals from the spectrograms based on the original song. np.hstack(v_predict) stacks any extra dimensions to createa a one dimensional array
        (
            _,
            v_segments_spectrogram,
        ) = reconstructed_v_wav_file.generate_segments_and_spectrograms(
            segment_length_in_seconds=Model_Blueprint.SEGMENT_LENGTH_IN_SECONDS
        )  # generates the spectrogram for the reconstructed vocals
        reconstructed_acc_wav_file = (
            wav_file.reconstruct_audio_from_spectrogram_using_inverse_stft(
                np.hstack(acc_predict),
                segment_length_in_seconds=Model_Blueprint.SEGMENT_LENGTH_IN_SECONDS,
            )
        )
        (
            _,
            acc_segments_spectrogram,
        ) = reconstructed_acc_wav_file.generate_segments_and_spectrograms(
            segment_length_in_seconds=Model_Blueprint.SEGMENT_LENGTH_IN_SECONDS
        )

        if visualise_graphs:
            plt.figure(figsize=(10, 15))
            plt.subplot(3, 1, 1)
            Visualiser.visualise_wav(reconstructed_v_wav_file)
            plt.title("vocal prediction")
            plt.subplot(3, 1, 2)
            Visualiser.visualise_wav(reconstructed_acc_wav_file)
            plt.title("accompaniment prediction")
            plt.subplot(3, 1, 3)
            Visualiser.visualise_wav(wav_file)
            plt.title("original song")
            Visualiser.visualise_spectrogram_for_whole_song(
                np.hstack(v_segments_spectrogram), title="spectrogram of vocals"
            )
            Visualiser.visualise_spectrogram_for_whole_song(
                np.hstack(acc_segments_spectrogram),
                title="spectrogram of accompaniment",
            )
            Visualiser.visualise_spectrogram_for_whole_song(
                np.hstack(spectrogram), title="spectrogram of original song"
            )
        return (reconstructed_v_wav_file, reconstructed_acc_wav_file, wav_file)

    def evaluate_on_test_dir(self) -> np.array:
        """
        This method evaluates the model on the test directory using a similarity metric.

        Returns
        -------
        tuple (vocals_mean_metric, accompaniments_mean_metric, vocals_median_metric, accompaniments_median_metric)
        """

        def compute_difference(
            ref_rec: np.array, est_rec: np.array, weightage=[0.33, 0.33, 0.33]
        ):
            """
            This method is based on this post: https://stackoverflow.com/questions/20644599/similarity-between-two-signals-looking-for-simple-measure
            It calculates the difference between the two signals in time domain, frequency domain and their respective power difference.

            The method calculates the total difference metric in a weighted fashion.

            Parameters
            ----------
            ref_rec : np.ndarray -> the reference recording

            est_rec : np.ndarray -> the estimated recording

            weightage : list
            """
            ## Time domain difference
            ref_time = np.correlate(ref_rec, ref_rec)
            inp_time = np.correlate(ref_rec, est_rec)
            diff_time = abs(ref_time - inp_time) / ref_time
            ## Freq domain difference
            ref_freq = np.correlate(np.fft.fft(ref_rec), np.fft.fft(ref_rec))
            inp_freq = np.correlate(np.fft.fft(ref_rec), np.fft.fft(est_rec))
            diff_freq = complex(abs(ref_freq - inp_freq) / ref_freq).real

            ## Power difference
            ref_power = np.sum(ref_rec ** 2)
            inp_power = np.sum(est_rec ** 2)
            diff_power = abs(ref_power - inp_power) / ref_power

            return float(
                weightage[0] * diff_time
                + weightage[1] * diff_freq
                + weightage[2] * diff_power
            )

        print("calculating similarity metrics...")
        vocal_metrics = []
        accomapniment_metrics = []
        counter = 0
        for music_name in os.listdir(Config_Handler.PATH_TO_TEST_DATA_DIR()):
            counter += 1
            mixture_path = os.path.join(
                Config_Handler.PATH_TO_TEST_DATA_DIR(), music_name, "mixture.wav"
            )
            vocal_path = os.path.join(
                Config_Handler.PATH_TO_TEST_DATA_DIR(), music_name, "vocals.wav"
            )
            accompaniment_path = os.path.join(
                Config_Handler.PATH_TO_TEST_DATA_DIR(), music_name, "accompaniment.wav"
            )
            vocal_wav_file_handler = Wav_File_Handler(
                audio_path=vocal_path, sample_rate=44100
            )
            actual_vocals_normalised = vocal_wav_file_handler.wav_file / np.max(
                vocal_wav_file_handler.wav_file
            )
            accompaniment_handler = Wav_File_Handler(
                audio_path=accompaniment_path, sample_rate=44100
            )
            actual_accompaniment_normalised = accompaniment_handler.wav_file / np.max(
                accompaniment_handler.wav_file
            )

            predicted_vocals, predicted_accompaniment, original_song = self.predict(
                song_path=mixture_path
            )

            predicted_vocals_cropped_to_original = predicted_vocals.wav_file[
                : len(vocal_wav_file_handler.wav_file)
            ]
            predicted_vocals_normalised = predicted_vocals_cropped_to_original / np.max(
                predicted_vocals_cropped_to_original
            )

            predicted_accompaniment_cropped_to_original = (
                predicted_accompaniment.wav_file[: len(accompaniment_handler.wav_file)]
            )
            predicted_accompaniment_normalised = (
                predicted_accompaniment_cropped_to_original
                / np.max(predicted_accompaniment_cropped_to_original)
            )

            vocal_metrics.append(
                compute_difference(
                    actual_vocals_normalised, predicted_vocals_normalised
                )
            )
            accomapniment_metrics.append(
                compute_difference(
                    actual_accompaniment_normalised, predicted_accompaniment_normalised
                )
            )

            print(
                "vocal_mean: ",
                np.mean(vocal_metrics) * 1000,
                "accompaniment mean: ",
                np.mean(accomapniment_metrics) * 1000,
                "vocal median:",
                np.median(vocal_metrics) * 1000,
                "accompaniment median:",
                np.median(accomapniment_metrics) * 1000,
            )

        # times 1000 to see the differces better
        return (
            np.mean(vocal_metrics) * 1000,
            np.mean(accomapniment_metrics) * 1000,
            np.median(vocal_metrics) * 1000,
            np.median(accomapniment_metrics) * 1000,
        )
