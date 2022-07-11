import sys
import numpy as np
sys.path.append('G:\My Drive\Final_year_project\src') #change the path to the specific path you extracted the src dir to

import unittest
from wav_file_handler import Wav_File_Handler

class Wav_File_Handler_Test(unittest.TestCase):
    

    def test_instansiation(self):
        #model should have 10 blocks (4 encoder blocks, 4 decoder blocks, 1 center block and 1 1x1 convolutional block)
        with self.assertRaises(AssertionError) as context:
            self.wav_file_handler = Wav_File_Handler()
        self.assertTrue("either wav_array or audio_path must be provided. if both are not provided, this method will raise an error. if both are provided, wav_array will be used." in str(context.exception))
    
        with self.assertRaises(AssertionError) as context:
            self.wav_file_handler = Wav_File_Handler(wav_array=[])
        self.assertTrue("wav_array must be a valid non-null np.array with type float!" in str(context.exception))
        
        with self.assertRaises(AssertionError) as context:
            self.wav_file_handler = Wav_File_Handler(wav_array=np.array(["1"]))
        self.assertTrue("wav_array must be a valid non-null np.array with type float!" in str(context.exception))
        
        with self.assertRaises(AssertionError) as context:
            self.wav_file_handler = Wav_File_Handler(audio_path="")
        self.assertTrue("audio_path must be a valid path to a song with a valid format (.mp3, .mp4 or .wav)!" in str(context.exception))
        
        with self.assertRaises(AssertionError) as context:
            self.wav_file_handler = Wav_File_Handler(audio_path="notwav.jpeg")
        self.assertTrue("audio_path must be a valid path to a song with a valid format (.mp3, .mp4 or .wav)!" in str(context.exception))
        
        with self.assertRaises(AssertionError) as context:
            self.wav_file_handler = Wav_File_Handler(audio_path="notwav.wav", sample_rate=0.0)
        self.assertTrue("audio_path must be a valid path to a song with a valid format (.mp3, .mp4 or .wav)!" in str(context.exception))
        
        with self.assertRaises(AssertionError) as context:
            self.wav_file_handler = Wav_File_Handler(audio_path=r"G:\My Drive\Final_year_project\Dataset\dataset-wav\test\Al James - Schoolboy Facination\mixture.wav", sample_rate=0.0)
        
        self.assertTrue("sample_rate must be a positive integer number!" in str(context.exception))
        
        with self.assertRaises(AssertionError) as context:
            self.wav_file_handler = Wav_File_Handler(audio_path=r"G:\My Drive\Final_year_project\Dataset\dataset-wav\test\Al James - Schoolboy Facination\mixture.wav", sample_rate=-10)
        
        self.assertTrue("sample_rate must be a positive integer number!" in str(context.exception))
                
        with self.assertRaises(AssertionError) as context:
            self.wav_file_handler = Wav_File_Handler(audio_path=r"G:\My Drive\Final_year_project\Dataset\dataset-wav\test\Al James - Schoolboy Facination\mixture.wav", sample_rate=0)
        
        self.assertTrue("sample_rate must be a positive integer number!" in str(context.exception))
        
        
        
    
        
    def test_generate_segments_and_spectrograms(self):
        
        self.wav_file_handler = Wav_File_Handler(audio_path=r"G:\My Drive\Final_year_project\Dataset\dataset-wav\test\Al James - Schoolboy Facination\mixture.wav") #change the path accordingly
        
        with self.assertRaises(AssertionError) as context:
            self.wav_file_handler.generate_segments_and_spectrograms(-0.0)
        
        self.assertTrue("segment_length_in_seconds must be a float greater than 0 and at most the duration of the wav file" in str(context.exception))
        
        with self.assertRaises(AssertionError) as context:
            self.wav_file_handler.generate_segments_and_spectrograms(1e10000)
            
        self.assertTrue("segment_length_in_seconds must be a float greater than 0 and at most the duration of the wav file" in str(context.exception))
        
        with self.assertRaises(AssertionError) as context:
            self.wav_file_handler.generate_segments_and_spectrograms(hop_length=-1)
        
        self.assertTrue("hop_length must be an integer greater than 0 and at most 5096 and less than or equal to window_size" in str(context.exception))
        
        with self.assertRaises(AssertionError) as context:
            self.wav_file_handler.generate_segments_and_spectrograms(hop_length=10.2)
        
        self.assertTrue("hop_length must be an integer greater than 0 and at most 5096 and less than or equal to window_size" in str(context.exception))
        
        with self.assertRaises(AssertionError) as context:
            self.wav_file_handler.generate_segments_and_spectrograms(hop_length=5097)
        
        self.assertTrue("hop_length must be an integer greater than 0 and at most 5096 and less than or equal to window_size" in str(context.exception))
        
        with self.assertRaises(AssertionError) as context:
            self.wav_file_handler.generate_segments_and_spectrograms(hop_length=2048, window_size=1024)

        self.assertTrue("hop_length must be an integer greater than 0 and at most 5096 and less than or equal to window_size" in str(context.exception))
        
        with self.assertRaises(AssertionError) as context:
            self.wav_file_handler.generate_segments_and_spectrograms(window_size=10000)

        self.assertTrue("window_size must be an integer greater than 0 and at most 5096" in str(context.exception))
        
        with self.assertRaises(AssertionError) as context:
            self.wav_file_handler.generate_segments_and_spectrograms(window_size=None)

        self.assertTrue("window_size must be an integer greater than 0 and at most 5096" in str(context.exception))
        
        with self.assertRaises(AssertionError) as context:
            self.wav_file_handler.generate_segments_and_spectrograms(zero_pad_factor=-1)

        self.assertTrue("zero_pad_factor must be an integer greater than or equal to 0 and at most 10" in str(context.exception))
        
        with self.assertRaises(AssertionError) as context:
            self.wav_file_handler.generate_segments_and_spectrograms(zero_pad_factor=11)

        self.assertTrue("zero_pad_factor must be an integer greater than or equal to 0 and at most 10" in str(context.exception))
        
        with self.assertRaises(AssertionError) as context:
            self.wav_file_handler.generate_segments_and_spectrograms(zero_pad_factor=None)

        self.assertTrue("zero_pad_factor must be an integer greater than or equal to 0 and at most 10" in str(context.exception))
        
        self.assertTrue(len(self.wav_file_handler.generate_segments_and_spectrograms())==2)
        self.assertTrue(self.wav_file_handler.generate_segments_and_spectrograms()[1].shape[1]==1025)
        
    def test_reconstruct_audio(self):
        
        self.wav_file_handler = Wav_File_Handler(audio_path=r"G:\My Drive\Final_year_project\Dataset\dataset-wav\test\Al James - Schoolboy Facination\mixture.wav") #change the path accordingly

        with self.assertRaises(AssertionError) as context:
            self.wav_file_handler.reconstruct_audio_from_spectrogram_using_inverse_stft("")
        self.assertTrue("spectrogram must be a non-empty float numpy array with at least wav file length with atleast 2 dimensions" in str(context.exception))
        
        with self.assertRaises(AssertionError) as context:
            self.wav_file_handler.reconstruct_audio_from_spectrogram_using_inverse_stft(self.wav_file_handler.wav_file[:10])
        
        self.assertTrue("spectrogram must be a non-empty float numpy array with at least wav file length with atleast 2 dimensions" in str(context.exception))
        
        
        with self.assertRaises(AssertionError) as context:
            self.wav_file_handler.reconstruct_audio_from_spectrogram_using_inverse_stft(-0.0)
        
        self.assertTrue("spectrogram must be a non-empty float numpy array with at least wav file length with atleast 2 dimensions" in str(context.exception))
        
        with self.assertRaises(AssertionError) as context:
            self.wav_file_handler.reconstruct_audio_from_spectrogram_using_inverse_stft(np.expand_dims(self.wav_file_handler.wav_file, -1), segment_length_in_seconds=1e10000)
            
        self.assertTrue("segment_length_in_seconds must be a float greater than 0 and at most the duration of the wav file" in str(context.exception))
        
        with self.assertRaises(AssertionError) as context:
            self.wav_file_handler.reconstruct_audio_from_spectrogram_using_inverse_stft(np.expand_dims(self.wav_file_handler.wav_file, -1), hop_length=-1)
        
        self.assertTrue("hop_length must be an integer greater than 0 and at most 5096 and less than or equal to win_length" in str(context.exception))
        
        with self.assertRaises(AssertionError) as context:
            self.wav_file_handler.reconstruct_audio_from_spectrogram_using_inverse_stft(np.expand_dims(self.wav_file_handler.wav_file, -1), hop_length=10.2)
        
        self.assertTrue("hop_length must be an integer greater than 0 and at most 5096 and less than or equal to win_length" in str(context.exception))
        
        with self.assertRaises(AssertionError) as context:
            self.wav_file_handler.reconstruct_audio_from_spectrogram_using_inverse_stft(np.expand_dims(self.wav_file_handler.wav_file, -1), hop_length=5097)
        
        self.assertTrue("hop_length must be an integer greater than 0 and at most 5096 and less than or equal to win_length" in str(context.exception))
        
        with self.assertRaises(AssertionError) as context:
            self.wav_file_handler.reconstruct_audio_from_spectrogram_using_inverse_stft(np.expand_dims(self.wav_file_handler.wav_file, -1), hop_length=2048, win_length=1024)

        self.assertTrue("hop_length must be an integer greater than 0 and at most 5096 and less than or equal to win_length" in str(context.exception))
        
        with self.assertRaises(AssertionError) as context:
            self.wav_file_handler.reconstruct_audio_from_spectrogram_using_inverse_stft(np.expand_dims(self.wav_file_handler.wav_file, -1), win_length=10000)

        self.assertTrue("win_length must be an integer greater than 0 and at most 5096" in str(context.exception))
        
        with self.assertRaises(AssertionError) as context:
            self.wav_file_handler.reconstruct_audio_from_spectrogram_using_inverse_stft(np.expand_dims(self.wav_file_handler.wav_file, -1), win_length=None)

        self.assertTrue("win_length must be an integer greater than 0 and at most 5096" in str(context.exception))
    
    def test_denoise(self):
        self.wav_file_handler = Wav_File_Handler(audio_path=r"G:\My Drive\Final_year_project\Dataset\dataset-wav\test\Al James - Schoolboy Facination\mixture.wav") #change the path accordingly
        
        with self.assertRaises(AssertionError) as context:
            self.wav_file_handler.denoise(0.0)
        self.assertTrue("hop_length_in_seconds must be a positive float and at most 0.1 seconds" in str(context.exception))
        
        with self.assertRaises(AssertionError) as context:
            self.wav_file_handler.denoise(0.0, window_length_in_seconds=0.5)
        self.assertTrue("hop_length_in_seconds must be a positive float and at most 0.5 seconds" in str(context.exception))
        
        with self.assertRaises(AssertionError) as context:
            self.wav_file_handler.denoise(1, window_length_in_seconds=0.5)
        self.assertTrue("hop_length_in_seconds must be a positive float and at most 0.5 seconds" in str(context.exception))
        
        with self.assertRaises(AssertionError) as context:
            self.wav_file_handler.denoise(window_length_in_seconds=1000)
        self.assertTrue(f"window_length_in_seconds must be a positive float and less than {self.wav_file_handler.wav_duration_in_seconds} seconds" in str(context.exception))
        
        with self.assertRaises(AssertionError) as context:
            self.wav_file_handler.denoise(window_length_in_seconds=None)
        self.assertTrue(f"window_length_in_seconds must be a positive float and less than {self.wav_file_handler.wav_duration_in_seconds} seconds" in str(context.exception))
        
        
        with self.assertRaises(AssertionError) as context:
            self.wav_file_handler.denoise(window_length_in_seconds="1000")
        self.assertTrue(f"window_length_in_seconds must be a positive float and less than {self.wav_file_handler.wav_duration_in_seconds} seconds" in str(context.exception))
        
    
if __name__ == '__main__':
    unittest.main()


