import sys
import tensorflow as tf

sys.path.append('G:\My Drive\Final_year_project\src') #change the path to the specific path you extracted the src dir to

import unittest
import numpy as np
from visualiser import Visualiser
from wav_file_handler import Wav_File_Handler



class Visualiser_Test(unittest.TestCase):
    
    def setUp(self) -> None:
        self.visualiser = Visualiser

    def test_visualise_spectrogram(self):

        with self.assertRaises(AssertionError) as context:
            self.visualiser.visualise_spectrogram_for_whole_song([])

        self.assertTrue("spectrogram must be a numpy array. Found list" in str(context.exception))

        with self.assertRaises(AssertionError) as context:
            self.visualiser.visualise_spectrogram_for_whole_song("not a numpy array")
        
        self.assertTrue("spectrogram must be a numpy array. Found str" in str(context.exception))

        with self.assertRaises(AssertionError) as context:
            self.visualiser.visualise_spectrogram_for_whole_song(np.array([["not a float"]]))
        
        self.assertTrue("the spectrogram must be a numpy array of float values." in str(context.exception))

        with self.assertRaises(AssertionError) as context:
            self.visualiser.visualise_spectrogram_for_whole_song(np.array([[[[[1.0, 2.0]]]]]))
        
        self.assertTrue("the spectrogram has 5 dimensions but 2 dimenstions were expected" in str(context.exception))

        with self.assertRaises(AssertionError) as context:
            self.visualiser.visualise_spectrogram_for_whole_song(np.array([[1.0, " "]]), sample_rate=0)

        self.assertTrue("the spectrogram must be a numpy array of float values." in str(context.exception))

        with self.assertRaises(AssertionError) as context:
            self.visualiser.visualise_spectrogram_for_whole_song(np.array([[1.0, 2.0]]), hop_length=0)

        self.assertTrue("hop_length must be a positive integer" in str(context.exception))

        with self.assertRaises(AssertionError) as context:
            self.visualiser.visualise_spectrogram_for_whole_song(np.array([[1.0, 2.0]]), hop_length="not an integer")

        self.assertTrue("hop_length must be a positive integer" in str(context.exception))

        with self.assertRaises(AssertionError) as context:
            self.visualiser.visualise_spectrogram_for_whole_song(np.array([[1.0, 2.0]]), hop_length=10, window_length=1)

        self.assertTrue("window_length must be a positive integer and greater than hop_length" in str(context.exception))

        with self.assertRaises(AssertionError) as context:
            self.visualiser.visualise_spectrogram_for_whole_song(np.array([[1.0, 2.0]]), hop_length=-100, window_length= -1)

        self.assertTrue("hop_length must be a positive integer" in str(context.exception))

        with self.assertRaises(AssertionError) as context:
            self.visualiser.visualise_spectrogram_for_whole_song(np.array([[1.0, 2.0]]), plot_title=2)

        self.assertTrue("plot_title must be str or None. Found int" in str(context.exception))

        with self.assertRaises(TypeError) as context:
            self.visualiser.visualise_spectrogram_for_whole_song(np.array([[1.0, 2.0]])) #values are not acceptable for creating a spectrogram
        
    def test_visualise_wav(self):
        
        with self.assertRaises(AssertionError) as context:
            self.visualiser.visualise_wav(wav=[])

        self.assertTrue("wav must be a Wav_File_Handler object. Found list" in str(context.exception))

        with self.assertRaises(AssertionError) as context:
            self.visualiser.visualise_wav(wav=3)

        self.assertTrue("wav must be a Wav_File_Handler object. Found int" in str(context.exception))

        with self.assertRaises(AssertionError) as context:
            self.visualiser.visualise_wav(wav=Wav_File_Handler(np.array([1, 2, 3])), plot_title=2)

        self.assertTrue("plot_title must be str or None. Found int" in str(context.exception))
        
        
        with self.assertRaises(AssertionError) as context:
            self.visualiser.visualise_wav(np.array([[1.0, 2.0]]), alpha=None)

        self.assertTrue("alpha must be a float between 0.0 and 1.0" in str(context.exception))



if __name__ == '__main__':

    unittest.main()


