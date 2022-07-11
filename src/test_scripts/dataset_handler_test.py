import sys
sys.path.append('G:\My Drive\Final_year_project\src') #change the path to the specific path you extracted the src dir to

import unittest
from dataset_handler import Dataset_Handler as dh

import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))
class Dataset_Handler_Test(unittest.TestCase):
    
    def setUp(self) -> None:
        self.dataset_handler = dh()
    
    
    def test_load_data_with_non_list_arg(self) -> None:

        with self.assertRaises(AssertionError) as context:
            self.dataset_handler.load_data(None)
        self.assertTrue("files_to_keep_names must be a list!" in str(context.exception))


        with self.assertRaises(AssertionError) as context:
            self.dataset_handler.load_data("not a list")
        self.assertTrue("files_to_keep_names must be a list!" in str(context.exception))



    def test_load_data_with_empty_list_arg(self) -> None:

        with self.assertRaises(AssertionError) as context:
            self.dataset_handler.load_data([])
        self.assertTrue(f"files_to_keep_names must be a list with at least one element. received: 0 elements" in str(context.exception))


    def test_load_data_with_empty_string_elements_in_list(self)-> None:

        with self.assertRaises(AssertionError) as context:
            self.dataset_handler.load_data(["string", " "])
        self.assertTrue("files_to_keep_names must be a list of non empty strings" in str(context.exception))

    
    def test_load_data_with_non_string_elements_in_list(self)-> None:
        with self.assertRaises(AssertionError) as context:
            self.dataset_handler.load_data(["string", 1])
        self.assertTrue("files_to_keep_names must be a list of non empty strings" in str(context.exception))

    
    def test_store_wav_with_wrong_sample_rate(self)-> None:

        import numpy as np
        with self.assertRaises(AssertionError) as context:
            self.dataset_handler.store_new_wav(np.array([1.0, 2.0]), path="test/path", sample_rate="sample_rate")
        
        print(context.exception)
        self.assertTrue("sample_rate must be a positive integer number!" in str(context.exception))


        with self.assertRaises(AssertionError) as context:
            self.dataset_handler.store_new_wav(np.array([1.0, 2.0]), path="test/path", sample_rate=10000.0)
        self.assertTrue("sample_rate must be a positive integer number!" in str(context.exception))


        with self.assertRaises(AssertionError) as context:

            self.dataset_handler.store_new_wav(np.array([1.0, 2.0]), path="test/path", sample_rate=-10)
        self.assertTrue("sample_rate must be a positive integer number!" in str(context.exception))


    def test_store_wav_invalid_data(self)-> None:
        import numpy as np
        with self.assertRaises(AssertionError) as context:
            self.dataset_handler.store_new_wav([], path="test/path", sample_rate=44100)
        self.assertTrue("data must be a non-empty numpy array with int or float values"in str(context.exception))


        with self.assertRaises(AssertionError) as context:
            self.dataset_handler.store_new_wav(123, path="test/path", sample_rate=44100)
        self.assertTrue("data must be a non-empty numpy array with int or float values"in str(context.exception))


        with self.assertRaises(AssertionError) as context:
            self.dataset_handler.store_new_wav(["not float or int", 12, 23.2], path="test/path", sample_rate=44100)
        self.assertTrue("data must be a non-empty numpy array with int or float values"in str(context.exception))



    
    def test_store_wav_with_wrong_path(self)-> None:
            import numpy as np
            with self.assertRaises(AssertionError) as context:
                self.dataset_handler.store_new_wav(np.array([1.0, 2.0]), path=[], sample_rate=44100)
            self.assertTrue("path must be a non empty string and in valid format" in str(context.exception))
    

            with self.assertRaises(AssertionError) as context:
                self.dataset_handler.store_new_wav(np.array([1.0, 2.0]), path=" ", sample_rate=44100)
            self.assertTrue("path must be a non empty string and in valid format" in str(context.exception))
    

            with self.assertRaises(AssertionError) as context:
                self.dataset_handler.store_new_wav(np.array([1.0, 2.0]), path="10", sample_rate=44100)
            self.assertTrue("path must be a non empty string and in valid format" in str(context.exception))
    


            with self.assertRaises(Exception) as context:
                self.dataset_handler.store_new_wav(np.array([1.0, 2.0]), path="notavalidfilepath/random.wav", sample_rate=44100)
            self.assertTrue("Error occured while storing new wav file to notavalidfilepath/random.wav. Check the path and try again!" in str(context.exception))

    
    
    def tearDown(self) -> None:
        del self.dataset_handler
        

if __name__ == '__main__':
    unittest.main()