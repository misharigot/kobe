import pandas as pd
import numpy as np
from typing import Optional
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

    def __init__(self, csv_path: Optional[str] = None,
        df: Optional[pd.DataFrame] = None
    ):
        if csv_path is not None:
            self.data = pd.read_csv(csv_path)
        elif df is not None:
            self.data = df
        else:
            raise ValueError('Either csv_path or df needs to have an argument.')
            
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
        
    def train_validation_split(self, verbose=False):
        """Returns the training and validation sets created via a time series
        split.
        
        Args:
            verbose (bool, optional): whether to print additional information.
                Defaults to False.
        
        Yields:
            tuple(np.ndarray, np.ndarray): the training and validation sets.
        """
        train_val_set = self._train_validation_set
        splits = TimeSeriesSplit(n_splits=self.N_SPLITS)
    
        # Add dataframe index as a column
        train_val_set.loc[:, 'index'] = train_val_set.index
        index_column_number = len(train_val_set.columns) - 1
        train_val_set = train_val_set.to_numpy()
        assert train_val_set[3][index_column_number] == 3  # Sanity check

        for train_index, validation_index in splits.split(train_val_set):
            train = train_val_set[train_index]
            validation = train_val_set[validation_index]
            
            if verbose:
                print(train[:, index_column_number])
                print(validation[:, index_column_number])

                print(f'Observations: {len(train) + len(validation)}')
                print(f'Training Observations: {len(train)}'
                    f' First index: {train[0][index_column_number]} '
                    f' Last index: {train[len(train) - 1][index_column_number]}')
                print(f'Validation Observations: {len(validation)}'
                    f' First index: {validation[0][index_column_number]}'
                    f' Last index: '
                    f'{validation[len(validation) - 1][index_column_number]}\n')
            
            assert len(train) + len(validation) - 1 \
                == validation[len(validation) - 1][index_column_number]

            yield train, validation
