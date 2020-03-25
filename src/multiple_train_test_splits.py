import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

class MultipleTrainTestSplits:
    """Split the csv given in a train/validation/test set, using the approach
    'Multiple Train-Test Splits'.

    Example usage:
        mtts = MultipleTrainTestSplits(csv_path='../data/data.csv')
        test_set = mtts.test_set
        for train_set, validation_set in mtts.train_validation_split():
            # Still need to exclude the label (Y/shot_made_flag)
            pass
    """
    N_SPLITS = 3  # Number of splits

    def __init__(self, csv_path: str):
        self.data = pd.read_csv(csv_path)
        self._remove_unlabeled_rows()

        # Reset index, so we do not have gaps in the index because of the removed unlabeled rows
        self.data = self.data.reset_index(drop=True)

        train_validation_set, test_set = self._extract_test_set()

        self.test_set = test_set
        self._train_validation_set = train_validation_set

    def _remove_unlabeled_rows(self):
        # Only obtain the labeled rows
        self.data = self.data.loc[~self.data['shot_made_flag'].isnull()].copy()

    def _extract_test_set(self, split=0.8):
        """Returns a tuple with the train/validation and test set on a given split.
        """
        n_rows = self.data.shape[0]
        start = int(n_rows * split)
        
        train_validation_set = self.data.loc[:start].copy()  # including start
        test_set = self.data.loc[start + 1:].copy()
        
        assert train_validation_set.shape[0] + test_set.shape[0] == n_rows
        return train_validation_set, test_set

    def _convert_to_dataframe(self, ndarray: np.ndarray) -> pd.DataFrame:
        """Converts the numpy array to a pandas dataframe.
        """
        columns = self.data.columns
            
        df = pd.DataFrame(ndarray)
        df.columns = columns
        return df
        
    def train_validation_split(self, as_dataframe=False):
        """Returns the training and validation sets created via a time series
        split.
        
        Args:
            verbose (bool, optional): whether to print additional information.
                Defaults to False.
            as_dataframe (bool, optional): whether to return the train and
                validation sets as a Pandas DataFrame. Defaults to False.
        
        Yields:
            tuple(np.ndarray, np.ndarray): the training and validation sets.
        """
        splits = TimeSeriesSplit(n_splits=self.N_SPLITS)
   
        train_val_set = self._train_validation_set.to_numpy()

        for train_index, validation_index in splits.split(train_val_set):
            train = train_val_set[train_index]
            validation = train_val_set[validation_index]

            if as_dataframe:
                train_df = self._convert_to_dataframe(train)
                validation_df = self._convert_to_dataframe(validation)
                yield train_df, validation_df
            else:
                yield train, validation
