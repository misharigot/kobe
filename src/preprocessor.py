from datetime import datetime

import pandas as pd
from dateutil import relativedelta as rd
from sklearn import preprocessing


class Preprocessor:
    """Preprocesses the data set.

    Example: 
    data = pd.read_csv('data/data.csv')
    pp = Preprocessor('data/data.csv')
    df = pp.preprocess(data)
    """
    # Categorize all columns based on their data type
    categorical_columns = [
        'action_type',
        'combined_shot_type',
        'game_id',
        'season',
        'shot_type',
        'shot_zone_area',
        'shot_zone_basic',
        'shot_zone_range',
        'opponent'
    ]

    temporal_columns = [
        'months_elapsed_from_first_game'
    ]

    remaining_columns = [
        'period',
        'shot_distance',
        'shot_made_flag',  # y label
    ]

    excluded_columns = [
        'lat',
        'lon',
        'game_event_id',  # Game event ids that are registered by the NBA probably, not relevant
        'shot_id',  # Just an auto-increment id, does not mean anything
        'team_id',
        'team_name',
        'loc_x',
        'loc_y',
    ]

    excluded_but_feature_engineered_columns = [
        'minutes_remaining',  # in time remaining
        'seconds_remaining',  # in time remaining
        'game_date',  # months_elapsed_from_start & day_of_week
        'matchup',  # home or away, opponent already included in other col
    ]

    def __init__(self, path_to_raw_data: str):
        """
        Args:
            path_to_raw_data (str): Raw data is needed to create a one-hot
                encoder that is familiar with all categorical values found in
                the data set.
        """
        self.raw_data = pd.read_csv(path_to_raw_data)
        self.encoder = self._create_one_hot_encoder()

    def _create_one_hot_encoder(self):
        categorical_cols = Preprocessor.categorical_columns
        self.raw_data[categorical_cols] = \
            self.raw_data[categorical_cols].astype('category')
        df_with_only_categoricals = self.raw_data[categorical_cols]

        encoder = preprocessing.OneHotEncoder(handle_unknown='ignore')
        encoder.fit(df_with_only_categoricals)
        return encoder

    def _add_time_remaining(self, data: pd.DataFrame) -> pd.DataFrame:
        """Combine the minutes and seconds remaining columns into one column.
        """
        df = data.copy()
        df['minutes_remaining'] = df['minutes_remaining'].astype(int)
        df['seconds_remaining'] = df['seconds_remaining'].astype(int)

        # Combine minutes and seconds remaining into decimal minutes remaining, e.g. 6.5 for 6 mins and 30 secs.
        df['time_remaining'] = round(df['minutes_remaining'] + (df['seconds_remaining'] / 60), 2)
        return df

    def _add_months_elapsed(self, data: pd.DataFrame) -> pd.DataFrame:
        """Adds a column months_elapsed_from_first game that indicates the 
        number of months that have elapsed from the first game that was found
        in the data set from 1996.
        """
        df = data.copy()
        first_recorded_game = min(self.raw_data['game_date'])

        def get_months(d1, d2):
            # Returns the difference in months between d1 and d2.
            date1 = datetime.strptime(str(d1), '%Y-%m-%d')
            date2 = datetime.strptime(str(d2), '%Y-%m-%d')
            r = rd.relativedelta(date2, date1)
            months = r.months +  12 * r.years
            if r.days > 0:
                months += 1
            return  months

        df['months_elapsed_from_first_game'] = \
            df.apply(lambda x: get_months(
                first_recorded_game,
                x['game_date']
            ), axis=1)
        return df


    def _add_home_away_(self, data: pd.DataFrame) -> pd.DataFrame:
        """Adds two columns that indicate whether the game was a home game
        or an away game.
        """
        df = data.copy()
        df['away'] = 0
        df['home'] = 0
        df.loc[df['matchup'].str.contains('@'), 'away'] = 1
        df.loc[df['matchup'].str.contains('vs'), 'home'] = 1
        return df

    def _one_hot_encode(self, data: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode all categorical columns.
        Optionally provide an encoder. Use the training set encoder to one-hot encode the test set.
        """
        df = data.copy()
        categorical_columns = Preprocessor.categorical_columns
        non_categorical_columns = df.columns.difference(categorical_columns)

        df_with_only_categoricals = df[categorical_columns]
        df_without_categoricals = df[non_categorical_columns]

        one_hot_encoded_df = pd.DataFrame(
            self.encoder.transform(df_with_only_categoricals).toarray()
        )

        # Combine the one hot encoded part of the df with the remaining df
        resulting_df = pd.concat(
            [one_hot_encoded_df, df_without_categoricals],
            axis=1
        )
        return resulting_df

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess (a subset of) the raw data from Kaggle.
        
        Args:
            data (pd.DataFrame): Raw data.
        
        Returns:
            pd.DataFrame: Preprocessed data.
        """
        df = data.copy()
        df = self._add_time_remaining(df)
        df = self._add_months_elapsed(df)
        df = self._add_home_away_(df)
        df = self._one_hot_encode(df)
        df = df.drop(Preprocessor.excluded_columns
            + Preprocessor.excluded_but_feature_engineered_columns
            , axis=1)
        return df
