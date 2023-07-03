from __future__ import annotations
import librosa
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from config_handler import Config_Handler
from pydub import AudioSegment
import scipy as sp
import random

class Wav_File_Handler:

    '''
    Instances of this class are responsible for handling all the operations on a single wav file
    '''
    @staticmethod
    def __random_path_generator()->str:
        '''
        a static method to generate a random song path

        Parameters
        ----------
        none

        Returns
        ----------
        random_audio_path:str 
            a random path to a wav file in test or train directory
        '''

        print("generating a random song path!")
        dataset_path = os.path.join(Config_Handler.PATH_TO_DATASET_WAVS(), random.choice(["train", "test"]))
        random_music_path = random.choice(os.listdir(dataset_path))
        random_wav_file_name = random.choice(["mixture.wav", "vocals.wav", "accompaniment.wav"])
        random_audio_path = os.path.join(dataset_path ,random_music_path, random_wav_file_name)
        print(f"generated a random path: {random_audio_path}")
        return random_audio_path

    def __init__(self, wav_array:np.array=None, audio_path:os.path = None, sample_rate:int=44100):
        #either audio_path or wav_array should be provided. if both are provided, wav_array will be used

        assert(not wav_array is None and audio_path is None ) or  (wav_array is None and not audio_path is None) or (not wav_array is None and not audio_path is None),  "either wav_array or audio_path must be provided. if both are not provided, this method will raise an error. if both are provided, wav_array will be used."
        if not wav_array is None:
            assert isinstance(wav_array, np.ndarray) and len(wav_array) > 0 and wav_array.dtype in (np.float16, np.float32, np.float64), "wav_array must be a valid non-null np.array with type float!"
        if not audio_path is None:
            assert  isinstance(audio_path, str) and os.path.exists(audio_path) and audio_path[-4:] in [".wav", ".mp3", ".mp4"], "audio_path must be a valid path to a song with a valid format (.mp3, .mp4 or .wav)!"
        assert isinstance(sample_rate, int) and sample_rate > 0, "sample_rate must be a positive integer number!"

        self.__random_path = Wav_File_Handler.__random_path_generator() if (not isinstance(wav_array, np.ndarray) or (isinstance(wav_array, np.ndarray) and len(wav_array) == 0)) and (audio_path == None or not os.path.exists(audio_path)) else None
        self.__wav_file = wav_array if isinstance(wav_array, np.ndarray) and len(wav_array) != 0 and isinstance(wav_array, np.ndarray)\
                                else librosa.load(audio_path, sr=sample_rate)[0] \
                                if audio_path !=None and os.path.exists(audio_path)\
                                else librosa.load(self.__random_path , sr=sample_rate)[0]
        self.__sample_rate = sample_rate
        self.__wav_name = "custom array"\
                        if isinstance(wav_array, np.ndarray) and len(wav_array) != 0\
                        else audio_path if audio_path !=None and os.path.exists(audio_path)\
                        else self.__random_path 
                                
        self.__wav_duration_in_seconds = len(self.__wav_file) / self.sample_rate #duration of a file is the number of samples divided by sample_rate



    @property
    def sample_rate(self)->int:
        return self.__sample_rate

    @property
    def wav_file(self)->np.array:
        return self.__wav_file

    @property
    def wav_name(self)->str:
        return self.wav_name

    @property
    def wav_duration_in_seconds(self)->float:
        return self.__wav_duration_in_seconds

    def __divide_into_segments(self, segment_length_in_seconds:float=2.0, zero_pad_factor:int=0)->np.array:

        '''
        This method divides the song into fixed length segments and returns an array of the segments. Segments can have any length but 2.0 second is recommended

        Parameters
        ----------
        segment_length_in_seconds:float (optional)
            the length of the segments in seconds. if not provided, the default value is 2.0 seconds

        zero_pad_factor:int (optional)
            the number of silence segments to add to the end of the segments. if not provided, the default value is 0.
            
        Notes
        ----------
        To make all the segments have same length, silence will be added to the last segment (if needed) to make it the same length as the other segments.
        This behaviout doesn't depend on the value of `zero_pad_factor`. If you want to add more silence to the end of the segments, you can do it by changing the value of `zero_pad_factor`

        Returns
        ----------
        np.array with 2 dimension. This is an array of fixed length segments
            
        '''

        if segment_length_in_seconds <= 0.0 or segment_length_in_seconds > librosa.get_duration(y=self.__wav_file, sr=self.sample_rate):
            
            return np.array([self.__wav_file])
            
            #if segment_length_in_seconds is 0.0 or less, or greater than the duration of audio file, no segmentation is performed
            #first the self.__wav_file will be converted to a list that contains it and then a numpy array of that list is returned
            #the reason is that we want to treat self.__wav_file as one segment so that in other methods, we still have a segments list that we can loop through
            
        number_of_segments_in_file = self.get_number_of_possible_segments(segment_length_in_seconds) 

        #the number of seconds in a wave file is 1/sample_rate. if we divide that by the length of chunks we want, 
        #we get the number of chunks with that length in our audio file

        segments = []
        for i in range(0, number_of_segments_in_file):
                
                segments.append(self.__wav_file[int(i*self.sample_rate*segment_length_in_seconds): int((i+1)*self.sample_rate*segment_length_in_seconds)])
            
        #the block of code below checks if the last segment that is added to the array
        #has the name number of samples as a segment with segment_length_in_seconds should have
        #if it has less, we create a silence tone and add it to that segment
        #this will ensure all segments have the same number of samples and therefore same length
        #so that our spectrograms all have the same length and the inputs to our model are therefore 
        #have equal size and shapes

        def zero_pad_segments():
            '''
            this function will add zeros to the end of the segments
            if `zero_pad_factor` = 0, number of zeros added to the end is `int(self.sample_rate*segment_length_in_seconds) - len(segments[-1])`
            which makes all segments to have an equal length. otherwise, the number of zeros to add is `int(self.sample_rate*zero_pad_factor*segment_length_in_seconds) - len(segments[-1])
            have a look at https://www.bitweenie.com/listings/fft-zero-padding/ for zero padding reasons
            '''
            number_of_needed_samples_for_the_last_segment = int(self.sample_rate*segment_length_in_seconds)-len(segments[-1])
            silence_to_last_segment = librosa.tone(0.0, sr=self.sample_rate, length=number_of_needed_samples_for_the_last_segment) #create a silence tone with the length of the needed number of samples to make the last segment have same length as the other segments
            segments[-1] = np.concatenate((segments[-1], silence_to_last_segment)) #add the silence to the last segment
            #adds the paddings to the segment arrays based on zero_pad_factor value
            for i in range(zero_pad_factor): 
                segments.append(librosa.tone(0.0, sr=self.sample_rate, length=self.sample_rate*segment_length_in_seconds))
            return segments

        segments = zero_pad_segments()
        return np.array(segments)

    def __stft(self, data:np.array, frame_size:int=2048, window_size:int=2048, hop_length:int=512) -> np.array:
        '''
        This method calculates stft of each segment and returns an array of the stfts of each segment

        Parameters
        ----------
        data:np.array
            the array of segments. must be 2 dimensional
        
        frame_size:int (optional)
            the number of samples in each frame. if not provided, the default value is 2048
            
        window_size:int (optional)
            the length of stft window. if not provided, the default value is 2048
        
        hop_length:int (optional)
            the number of samples to move the window forward. if not provided, the default value is 512  
        
        Returns
        ----------
        np.array with 3 dimensions => (# of segments, #frequency bins, #frames). #frequency bins = `frame_size//2 + 1` and #frames = `number_of_samples_in_segment//hop_length + 1`.
                                    with default values, the shape will be (# of segments, 1025, 173) with sample_rate of 44100 because #frequency bins = 2048//2 + 1 = 1025 and #frames = (2*44100)//512 + 1 = 173
        '''

        stfts = []
        for segment in data:
            stfts.append(librosa.stft(segment, n_fft=frame_size, win_length=window_size, hop_length=hop_length))
        return np.array(stfts)

    def __spectrogram_from_stft(self, stft_data:np.array)->np.array:
        '''
        This method calculates the power spectrum of each segment stft and returns an array of the power spectrograms of each segment

        Parameters
        ----------
        stft_data:np.array
                the array of segments stfts.

        Returns
        ----------
        np.array

        '''
        stft_magnitude = np.abs(stft_data)**2
        return librosa.power_to_db(stft_magnitude)


    def get_number_of_possible_segments(self, segment_length_in_seconds:float=2.0):
        '''
        This method returns the number of possible segments based on the segment length in seconds
        
        Parameters
        ----------
        segment_length_in_seconds:float (optional)
            the length of the segments in seconds. if not provided, the default value of 2.0 will be used
        
        Returns
        ----------
        int: number of possible `segment_length_in_seconds` segments in the wav array
        '''
        return int(np.ceil(len(self.__wav_file) / (self.sample_rate*segment_length_in_seconds)))

    def generate_segments_and_spectrograms(self, segment_length_in_seconds:float=2.0, frame_size:int=2048, window_size:int=2048, hop_length:int=512, zero_pad_factor:int=0)->tuple:
        '''
        This method is used to generate segments and spectrograms of the wav file with specified parameters.
        
        Parameters
        ----------
        segment_length_in_seconds:float (optional)
            the length of the segments in seconds. if not provided, the default value of 2.0 will be used
        
        frame_size:int (optional)
            the frame_size of stft. if not provided, the default value is 2048
        
        window_size:int (optional)
            the window_size of stft. if not provided, the default value is 2048
        
        hop_length:int (optional)
            the hop_length of stft. if not provided, the default value is 512
        
        zero_pad_factor:int (optional)
            the number of zero padding segments. if not provided, the default value is 0
        
        Returns
        ----------
        tuple: (np.array of segments, np.array of segment spectrograms)
        
        Raises
        ----------
        AssertionError
        '''
        assert isinstance(segment_length_in_seconds, float) and segment_length_in_seconds > 0 and segment_length_in_seconds <= self.wav_duration_in_seconds, "segment_length_in_seconds must be a float greater than 0 and at most the duration of the wav file"
        assert isinstance(frame_size, int) and frame_size > 0 and frame_size <= 5096, "frame_size must be an integer greater than 0 and at most 5096"
        assert isinstance(window_size, int) and window_size > 0 and window_size <= 5096, "window_size must be an integer greater than 0 and at most 5096"
        assert isinstance(hop_length, int) and hop_length > 0 and hop_length <= 5096 and hop_length <= window_size, "hop_length must be an integer greater than 0 and at most 5096 and less than or equal to window_size"
        assert isinstance(zero_pad_factor, int) and zero_pad_factor >= 0 and zero_pad_factor <= 10, "zero_pad_factor must be an integer greater than or equal to 0 and at most 10"

        segments = self.__divide_into_segments(segment_length_in_seconds=segment_length_in_seconds, zero_pad_factor=zero_pad_factor)
        stft = self.__stft(segments, frame_size, window_size, hop_length)
        spectrogram_array_of_segments = self.__spectrogram_from_stft(stft)
        
        return (segments, spectrogram_array_of_segments)

    def reconstruct_audio_from_spectrogram_using_inverse_stft(self, spectrogram:np.array, segment_length_in_seconds:float=2.0, hop_length:int=512, frame_size:int=2048, win_length:int=2048, denoise=False)->Wav_File_Handler:
        '''
        This method is used to reconstruct the audio file from the spectrogram using inverse stft and phases of the audio file segments.
        The audio data will be segmeted and after calculating the stft of each segment, the phase of each stft will be extracted and multiplied by the spectrogram of the segments.
        For the best performance, make sure spectrogram is generated using the same parameters as the one passed to this method.

        Parameters
        ----------
        spectrogram:np.array
                the spectrogram to reconstruct the audio from

        segment_length_in_seconds:float (optional)
                the length of the segments in seconds. if not provided, the default value of 2.0 will be used

        frame_size:int (optional)
                the frame_size of stft. if not provided, the default value is 2048
            
        window_size:int (optional)
                the window_size of stft. if not provided, the default value is 2048
            
        hop_length:int (optional)
                the hop_length of stft. if not provided, the default value is 512

        denoise:bool (optional)
                whether to denoise the audio. if not provided, the default value is False

        Returns
        ----------
        Wave_File_Handler object of the reconstructed audio file

        Raises
        ----------
        AssertionError

        '''
        assert isinstance(spectrogram, np.ndarray) and len(spectrogram) > 0 and len(spectrogram.shape) >= 2 and spectrogram.dtype in (float, np.float16, np.float32, np.float64) , "spectrogram must be a non-empty float numpy array with at least wav file length with atleast 2 dimensions"
        assert isinstance(segment_length_in_seconds, float) and segment_length_in_seconds > 0 and segment_length_in_seconds <= self.wav_duration_in_seconds, "segment_length_in_seconds must be a float greater than 0 and at most the duration of the wav file"
        assert isinstance(frame_size, int) and frame_size > 0 and frame_size <= 5096, "frame_size must be an integer greater than 0 and at most 5096"
        assert isinstance(win_length, int) and win_length > 0 and win_length <= 5096, "win_length must be an integer greater than 0 and at most 5096"
        assert isinstance(hop_length, int) and hop_length > 0 and hop_length <= 5096 and hop_length <= win_length, "hop_length must be an integer greater than 0 and at most 5096 and less than or equal to win_length"
        assert isinstance(denoise, bool), "denoise must be a boolean"
        
        while len(spectrogram.shape) > 2:
            spectrogram = np.hstack(spectrogram) #if spectrogram is passed as arrays of segments spectrogram, we concatenate the dimensions to get a 2d array (frequency_bins, spectrogram_magnitude) (i think)
        #the following code is trying to get the original phase value from the song
        segments = self.__divide_into_segments(segment_length_in_seconds=segment_length_in_seconds)
        segments_stft = self.__stft(segments, frame_size, win_length, hop_length)
        stft_for_whole_song = np.hstack(segments_stft)
        #the following code gets the phase from the complex stft, multiplies it element-wisely with the predicted spectrogram and uses istft to get the reconstructed audio
        complex_segment_phases = librosa.magphase(stft_for_whole_song)[1]
        spectrogram = spectrogram[:, :complex_segment_phases.shape[1]] #removes the added silence during original segmentation of data
        complex_valued_spectrogram = np.multiply(complex_segment_phases, spectrogram)
        reconstructed_audio = librosa.istft(complex_valued_spectrogram, hop_length=hop_length,  win_length = win_length) #istft to move the audio from time-frequency domain to time-domain

        if denoise:
            reconstructed_audio = self.denoise(reconstructed_audio)
        return Wav_File_Handler(wav_array=reconstructed_audio)



    def __hilbert_metrics(self)->tuple:
        '''this calculates the amplitude envelope of the audio and returns it'''
        analytic_signal = sp.signal.hilbert(self.__wav_file)
        amplitude_envelope = np.abs(analytic_signal)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = (np.diff(instantaneous_phase) /
                                    (2.0*np.pi) * 44100)
        instantaneous_frequency += np.max(instantaneous_frequency)
        return amplitude_envelope, instantaneous_frequency

    def denoise(self, hop_length_in_seconds:float=0.02, window_length_in_seconds:float=0.1, threshold_softness:float=3.0, stat_mode="median", verbose:bool=False)->np.array:
        '''
        This method is used to denoise the audio file and it returns a copy of the denoised audio array.
        It performs a windowing function to denoise segments by comparing them to a threshold value.
        This threshold value is calculated using a statistical mode on the amplitude envelope generated by hilbert transform of the signal.
        If the calcutated statistic of the amplitude envelope of the previous and next segment of the current segment in the windowing function is less than the threshold,
        this probably means the current segment is noise (as there is an amplitude difference and usually noise has a lower amplitude) and it will be silenced.

        Parameters
        ----------
        hop_length_in_seconds:float (optional)
                the hop length of the denoising windowing function. if not provided, the default value is 0.02. it is advised to keep this number as default or at most the same as `window_length_in_seconds`

        window_length_in_seconds:float (optional)
                the length of the denoising windowing function. if not provided, the default value is 0.1 second.

        threshold_softness:float (optional)
                the softness of the threshold. if this number is high, the threshold will be lower and therefore, the low amplitude areas may persist.

        stat_mode:str (optional)
                the statistical mode to use to calculate from the amplitude envelope of the signal. if not provided, the default value is "median".
                median is recommended as it's not affected by outliers and is faster to calculate.

        verbose:bool (optional)
                whether to print the progress of the denoising process. if not provided, the default value is False

        Returns
        ----------
        np.array: The denoised copy of the audio file

        Raises
        ----------
        AssertionError

        '''
        assert isinstance(window_length_in_seconds, float) and window_length_in_seconds > 0 and window_length_in_seconds*self.sample_rate < len(self.__wav_file), f"window_length_in_seconds must be a positive float and less than {self.wav_duration_in_seconds} seconds"
        assert isinstance(hop_length_in_seconds, float) and hop_length_in_seconds > 0 and hop_length_in_seconds <=  window_length_in_seconds, f"hop_length_in_seconds must be a positive float and at most {window_length_in_seconds} seconds"
        assert isinstance(threshold_softness, float) and threshold_softness > 0, f"threshold_softness must be a positive float"
        assert isinstance(verbose, bool), f"verbose must be a boolean. Found {type(verbose).__name__}"
        stat_mode = str.lower(stat_mode).replace(" ", "")
        assert isinstance(stat_mode, str) and stat_mode in ["median", "mean", "mode"], print(f"expected 'mean', 'median' or 'mode' for `stat_mode` but received: '{stat_mode}'")


        def amps_reducer_function(amps):
            if stat_mode == "median":
                return np.median(amps)
            elif stat_mode == "mean":
                return np.mean(amps)
            elif stat_mode == "mode":
                return sp.stats.mode(amps)

        wav = np.copy(self.__wav_file)
        amp, freq = self.__hilbert_metrics()
        amp_stat = amps_reducer_function(amp)
        threshold = amp_stat/threshold_softness #calculates the threshold
        hop_length = int(hop_length_in_seconds*self.sample_rate) #converts the hop length in seconds to number of samples
        window_length = int(window_length_in_seconds*self.sample_rate) #converts the window length in seconds to number of samples
        muted_segments_count = 0

        for i in range(window_length, len(wav)-window_length, hop_length): #iterates over the audio file and moves the window by hop_length samples
            segment = amp[i: i+window_length] #gets the current segment of the window
            previous_segment_stat = amps_reducer_function(amp[i-window_length: i]) #calculates the amplitude stats of the previous segment
            next_segment_stat = amps_reducer_function(amp[i+window_length: i+window_length*2]) #calculates the amplitude stats of the next segment

            if previous_segment_stat < threshold and next_segment_stat < threshold:       
                segment *= 0.0
                fade_next = np.linspace(0, 1, len(wav[i+window_length: i+window_length*2]))
                if i == window_length: #if it's the first sample of the window, silence the previous and current segments and fade in the next segment linearly
                    wav[i-window_length: i] = segment
                    wav[i: i+window_length] = segment
                    wav[i+window_length: i+window_length*2] *= fade_next
                else: #oterwise, fade out the previous segment linearly, mute the current segment and fade in the next segment linearly 
                    fade_prev = np.linspace(1, 0, len(wav[i-window_length: i]))
                    wav[i-window_length: i] *= fade_prev
                    wav[i: i+window_length] = segment
                    wav[i+window_length: i+window_length*2] *= fade_next
        
                #fading makes the denoising more realistic

            muted_segments_count +=	 1
            
        if verbose: print(f"Denoising completed! muted {muted_segments_count} segments")
        return wav
