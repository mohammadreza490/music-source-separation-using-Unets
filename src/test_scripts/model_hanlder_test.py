import sys
import os
import tensorflow as tf
import numpy as np

sys.path.append('G:\My Drive\Final_year_project\src') #change the path to the specific path you extracted the src dir to

import unittest

from model_handler import Model_Handler
from config_handler import Config_Handler
from wav_file_handler import Wav_File_Handler
class Model_Handler_Test(unittest.TestCase):
    
    def setUp(self) -> None:
        Config_Handler.init(r"G:\My Drive\Final_year_project") #change the path to the specific path you extracted zip file to

    def test_model_creation(self):
        with self.assertRaises(AssertionError) as context:
            self.model = Model_Handler(" ")
        self.assertTrue("model_name must be a non-empty string" in str(context.exception))
        
        with self.assertRaises(AssertionError) as context:
            self.model = Model_Handler(123123)
        self.assertTrue("model_name must be a non-empty string" in str(context.exception))
        
        self.model = Model_Handler("test_model")
        self.assertTrue(self.model is not None)
        self.assertTrue(os.path.exists(self.model.model_dir) and self.model.model_dir == os.path.join(Config_Handler.PATH_TO_MODELS(), "test_model"))
        self.assertTrue(self.model.model_name == "test_model")
        self.assertTrue(os.path.exists(self.model.trained_model_path) and self.model.trained_model_path ==os.path.join(Config_Handler.PATH_TO_MODELS(), "test_model", "test_model-trained-model"))
        self.assertTrue(os.path.exists(self.model.checkpoint_dir) and self.model.checkpoint_dir == os.path.join(Config_Handler.PATH_TO_MODELS(), "test_model", "checkpoint"))
        self.assertFalse(os.path.exists(self.model.checkpoint_path)) #no checkpoint path is there yet because model is not trained
    
    def test_model_training(self):
                
        self.model = Model_Handler("test_model")
        def loss_function_invalid(one_parameter_only):
            return 0.0
        
        with self.assertRaises(AssertionError) as context:
            self.model.train(loss_function_invalid, learning_rate=0.001, epochs=10, batch_size=10)

        self.assertTrue("The loss function should be a Callable object with at least two arguments: y_true and y_pred. Found 1 argument(s) and type function." in str(context.exception))
    
    def test_prediction(self):
        
        self.model = Model_Handler("test_model")
        
        with self.assertRaises(AssertionError) as context:
            self.model.predict(None, None)
        
        print(context.exception)
        self.assertTrue("either song_data or song_path must be provided. if both are not provided, this method will raise an error. if both are provided, song_data will be used." in str(context.exception))
        
        with self.assertRaises(AssertionError) as context:
            self.model.predict(song_data = "not array")
        print(context.exception)
        self.assertTrue("song_data must be a valid non-null np.array with type float!" in str(context.exception))
        
#
        with self.assertRaises(AssertionError) as context:
            self.model.predict(song_path= "does not exist.wav")
        self.assertTrue("song_path must be a valid path to a song with a valid format (.mp3, .mp4 or .wav)!" in str(context.exception))
        
        with self.assertRaises(AssertionError) as context:
            self.model.predict(np.array([0.0, 0.0, 0.0]), visualise_graphs="not bool")
        self.assertTrue("visualise_graphs must be a boolean!" in str(context.exception))
        
        with self.assertRaises(AssertionError) as context:
            self.model.predict(np.array([0.0, 0.0, 0.0]), batch_size=70.0)
        self.assertTrue("batch_size must be a positive integer!" in str(context.exception))
        
        
        
        

if __name__ == '__main__':
    unittest.main()


