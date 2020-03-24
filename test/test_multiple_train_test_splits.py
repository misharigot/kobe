import unittest

import pandas as pd

from src.multiple_train_test_splits import MultipleTrainTestSplits


class MultipleTrainTestSplitsTest(unittest.TestCase):
    def init_test(self):
        data = pd.DataFrame([1,2,3,4])
        mtts = MultipleTrainTestSplits(pd = data)
        test_set = mtts.test_set
        for train_set, validation_set in mtts.train_validation_split():
            # Still need to exclude the label (Y/shot_made_flag)
            self.assertEqual(len(train_set) == 3)

if __name__ == '__main__':
    unittest.main()
