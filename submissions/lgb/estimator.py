import os
import pandas as pd
from geopy import distance
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import lightgbm as lgb

def _merge_external_data(X):
    
    X = X.copy()  # Working on a copy of X to avoid modifying the original one
    X.loc[:, "DateOfDeparture"] = pd.to_datetime(X['DateOfDeparture']) # Making sure DateOfDeparture is of dtype datetime
    
    # Importing our external dataset
    filepath = os.path.join(os.path.dirname(__file__), 'external_data.csv')
    external_data = pd.read_csv(filepath, parse_dates=['DateOfDeparture'])
    
    # Merging our external data to retrieve all the information about the departure and arrival of each flight
    columns = external_data.columns
    external_data.columns = ['DateOfDeparture', 'Departure'] + ['dep_' + col for col in columns[2:]]
    X_merged = pd.merge(X, external_data, how='left', on=['DateOfDeparture', 'Departure'], sort=False)
    external_data.columns = ['DateOfDeparture', 'Arrival'] + ['arr_' + col for col in columns[2:]]
    X_merged = pd.merge(X_merged, external_data, how='left', on=['DateOfDeparture', 'Arrival'], sort=False)

    # Dropping the duplicate columns since they're the same for departure and arrival
    X_merged = X_merged.rename(columns={'arr_is_WeekEnd':'is_WeekEnd', 'arr_is_important':'is_important',
                                        'arr_is_Summer':'is_Summer'})
    X_merged.drop(columns=['dep_is_WeekEnd', 'dep_is_important', 'dep_is_Summer'], inplace=True)
    
    # Computing the distance between the two airports of each flight
    X_merged['Distance'] = X_merged.apply(lambda x : distance.distance((x['dep_Latitude'], x['dep_Longitude']),
                                                                       (x['arr_Latitude'], x['arr_Longitude'])).km, axis=1)
    
    # Adding product columns to represent the interaction between some departure/arrival pairs of variables
    X_merged["inter_Traffic"] = X_merged.loc[:, "dep_Traffic"] * X_merged.loc[:, "arr_Traffic"]
    X_merged["inter_GDP"] = X_merged.loc[:, "dep_GDP"] * X_merged.loc[:, "arr_GDP"]
    X_merged["inter_StateHoliday"] = X_merged.loc[:, "dep_is_State_Holiday"] * X_merged.loc[:, "arr_is_State_Holiday"]
    X_merged["inter_DistanceToClosestHoliday"] = X_merged.loc[:, "dep_Distance_To_Closest_Holiday"] * X_merged.loc[:, "arr_Distance_To_Closest_Holiday"]
    
    return X_merged


def _encode_dates(X):
    
    # Encoding the date information from the DateOfDeparture column
    X.loc[:, 'year'] = X['DateOfDeparture'].dt.year
    X.loc[:, 'month'] = X['DateOfDeparture'].dt.month
    X.loc[:, 'day'] = X['DateOfDeparture'].dt.day
    X.loc[:, 'weekday'] = X['DateOfDeparture'].dt.weekday
    X.loc[:, 'week'] = X['DateOfDeparture'].dt.isocalendar().week
    X.loc[:, 'n_days'] = X['DateOfDeparture'].apply(lambda date: (date - pd.to_datetime("1970-01-01")).days)
    
    # Creating binary features based on the date information
    date_info = ['year', 'month', 'day', 'weekday', 'week']
    for feat in date_info:
        X = X.join(pd.get_dummies(X[feat], prefix=feat))
    
    # Finally, we can drop the original DateOfDeparture column from the dataframe
    return X.drop(columns=["DateOfDeparture"])


def get_estimator():
    
    data_merger = FunctionTransformer(_merge_external_data)
    date_encoder = FunctionTransformer(_encode_dates)
    
    categorical_encoder = make_pipeline(SimpleImputer(strategy='constant', fill_value='missing'), OneHotEncoder())
    
    categorical_cols = ['Arrival', 'Departure']

    preprocessor = make_column_transformer((categorical_encoder, categorical_cols), remainder='passthrough')
    
    regressor = lgb.LGBMRegressor(boosting_type='dart', n_estimators=10000, learning_rate=0.1, max_depth=-1, num_leaves=16, subsample=0.9, colsample_bytree=0.9, subsample_freq=1, uniform_drop=True)
    
    return make_pipeline(data_merger, date_encoder, preprocessor, regressor)
