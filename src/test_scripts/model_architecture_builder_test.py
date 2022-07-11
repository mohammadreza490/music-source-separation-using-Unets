import sys
import tensorflow as tf
sys.path.append('G:\My Drive\Final_year_project\src') #change the path to the specific path you extracted the src dir to

import unittest
from model_architecture_builder import Model_Architecture_Builder

class Model_Architecture_Builder_Test(unittest.TestCase):
    
    def setUp(self) -> None:
        self.model = Model_Architecture_Builder.create_default_model()

    def test_model_number_of_blocks(self):
        #model should have 10 blocks (4 encoder blocks, 4 decoder blocks, 1 center block and 1 1x1 convolutional block)
        self.assertTrue(len(self.model.layers) == 10)
    

    def tearDown(self) -> None:
        del self.model


if __name__ == '__main__':
    unittest.main()


