import sys
import tensorflow as tf
sys.path.append('G:\My Drive\Final_year_project\src')

import unittest
from config_handler import Config_Handler

class Config_Handler_Test(unittest.TestCase):


    def test_init(self):
        
        with self.assertRaises(AssertionError) as context:
            Config_Handler.init(None)
        self.assertTrue("Path should be a non-empty string!" in str(context.exception))
        
        with self.assertRaises(AssertionError) as context:
            Config_Handler.init(2)
        self.assertTrue("Path should be a non-empty string!" in str(context.exception))
        
        with self.assertRaises(AssertionError) as context:
            Config_Handler.init(" ")
        self.assertTrue("Path should be a non-empty string!" in str(context.exception))
        
        Config_Handler.init("G:\My Drive\Final_year_project") #change it to your own path
        
        self.assertTrue(Config_Handler.PROJECT_FOLDER_PATH() == r"G:\My Drive\Final_year_project")
        self.assertTrue(Config_Handler.PATH_TO_MODELS() == r"G:\My Drive\Final_year_project\Models")
        self.assertTrue(Config_Handler.PATH_TO_DATASET_WAVS() == r"G:\My Drive\Final_year_project\Dataset\dataset-wav")
        self.assertTrue(Config_Handler.PATH_TO_DATASET_ZIPPED() == r"G:\My Drive\Final_year_project\Dataset\musdb18.zip")
        self.assertTrue(Config_Handler.PATH_TO_DATASET_STEMS() == r"G:\My Drive\Final_year_project\Dataset\musdb18")
        print(Config_Handler.PATH_TO_TRAIN_DATA_DIR())
        self.assertTrue(Config_Handler.PATH_TO_TRAIN_DATA_DIR() == r"G:\My Drive\Final_year_project\Dataset\dataset-wav\train")
        self.assertTrue(Config_Handler.PATH_TO_TEST_DATA_DIR() == r"G:\My Drive\Final_year_project\Dataset\dataset-wav\test")
        self.assertTrue(Config_Handler.PATH_TO_TEMP_DATA_DIR() == r"G:\My Drive\Final_year_project\Dataset\dataset-wav\temp")
        self.assertTrue(Config_Handler.PATH_TO_TEMP_SPECTROGRAMS_FOR_TRAINING_DIR() == r"G:\My Drive\Final_year_project\Dataset\dataset-wav\temp\model_spec")
        
        


if __name__ == '__main__':
    unittest.main()

