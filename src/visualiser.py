import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from wav_file_handler import Wav_File_Handler


class Visualiser:
    """
    A utility class for performing visualisation tasks.
    """

    @staticmethod
    def visualise_spectrogram_for_whole_song(
        spectrogram: np.array,
        plot_title: str = None,
        hop_length: int = 512,
        window_length: int = 2048,
        sample_rate: int = 44100,
    ):
        """
        This static method visualises the spectrogram for a song.

        Parameters
        ----------
        spectrogram : np.array
                this should be the power spectrogram (result of a `librosa.power_to_db(np.abs(stft)**2)`) of the wav data and has the shape (#frequency_bins, #frames)

        plot_title : str
                the title of the plot(optional)

        hop_length : int
                the hop length of the stft(optional)
                default is 512 frames

        window_length : int
                the window length of the stft

        sample_rate : int
                the sample rate of song (optional)
                default is 44100 samples per second

        Returns
        -------
        None

        Raises
        ------
        AssertionError, TypeError

        """
        assert isinstance(
            spectrogram, np.ndarray
        ), f"spectrogram must be a numpy array. Found {type(spectrogram).__name__}"
        assert (
            len(spectrogram.shape) == 2
        ), f"the spectrogram has {len(spectrogram.shape)} dimensions but 2 dimenstions were expected"
        assert (
            spectrogram.dtype == float
        ), "the spectrogram must be a numpy array of float values."
        assert (
            isinstance(sample_rate, int) and sample_rate > 0
        ), "sample_rate must be a positive integer number!"
        assert (
            isinstance(hop_length, int) and hop_length > 0
        ), "hop_length must be a positive integer"
        assert (
            isinstance(window_length, int)
            and window_length > 0
            and window_length > hop_length
        ), "window_length must be a positive integer and greater than hop_length"
        assert plot_title == None or isinstance(
            plot_title, str
        ), f"plot_title must be str or None. Found {type(plot_title).__name__}"

        plt.figure(figsize=(15, 10))
        try:
            librosa.display.specshow(
                spectrogram,
                x_axis="time",
                y_axis="log",
                sr=sample_rate,
                hop_length=hop_length,
            )  # visualises the spectrogram
        except TypeError as e:
            raise TypeError(e.message)

        colorbar = plt.colorbar(
            format="%2.f",
        )  # adds a colorbar for different intensity levels
        colorbar.ax.set_title("db")
        if plot_title:
            plt.title(plot_title)
        plt.show()

    @staticmethod
    def visualise_wav(
        wav: Wav_File_Handler, alpha: float = 0.5, plot_title: str = None
    ):
        """
        Visualises the wave plot of the wav file.

        Parameters
        ----------
        wav : Wav_File_Handler

        alpha : float (optional)
                the transparency of the wave plot
                default is 0.5

        plot_title : str (optional)
                the title of the plot

        Returns
        -------
        None

        Raises
        ------
        AssertionError, TypeError

        """

        assert (
            isinstance(alpha, float) and 0.0 < alpha <= 1.0
        ), "alpha must be a float between 0.0 and 1.0"
        assert plot_title == None or isinstance(
            plot_title, str
        ), f"plot_title must be str or None. Found {type(plot_title).__name__}"

        librosa.display.waveshow(wav.wav_file, sr=wav.sample_rate, alpha=alpha)

        if plot_title:
            plt.title(plot_title)
