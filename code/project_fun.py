import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

def plotter_distribution(df):
    '''
    Computes distribution at specified timeframe.

    INPUT: Dataframe
    OUTPUT: matplotlib object
    '''

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    for user in df.ID.unique().tolist():
        ax.plot(df.ix[df.ID == user].ts, df.ix[df.ID == user].consumption)

def aggregator(df, frequency = '1D'):
    '''
    Computes stats by frequency interval and user defined aggregating functions.

    INPUT: DataFrame, frequency
    OUTPUT: DataFrame

    '''
    # aggregators = [np.sum, np.mean, np.std]
    aggregators = [np.mean]
    return df.groupby([pd.Grouper(freq=frequency)]).agg(aggregators)

# def seasonalfeature(df):
#     seasons = {1: 'winter', 2:'spring', 3:'summer', 4:'autumn'}
#     pass
#

if __name__ == '__main__':

    df = pickle.load(open('../data/working_data.pickle'))

    # Filter for trial period Jan 2010 - Dec 2010
    start_date = datetime(2010,1,1)
    end_date = datetime(2010,12,31,23,59)

    # Set analyzed timeperiod
    df = df[start_date:end_date]

    #Remove IDs if nulls.
    df.dropna(axis = 1, how = 'any', inplace = True)

    #Aggregate df
    df_aggr = aggregator(df, frequency = '6H')

    #CREATE FEATURE MATRIX
    
    #First cluster is based on the mean of 2 hour aggregation.
    df_sum = df_aggr.loc[:,(slice(None),'mean')]

    #Compute the mean for same interval range during the year.
    df_means = df_sum.groupby([df_sum.index.hour]).mean()

    #Transform feature matrix, users be rows, features be columns.
    X = df_means.T.values

    #STANDARDIZE FEATUERES
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)

    #Initiates kMeans instance
    kmeans = KMeans(n_clusters = 3)
    y_pred = kmeans.fit_predict(X_standardized)

    #Plot clusters
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    #Plot where demand is pressumably higher during the day, first 6am-12pm, 6pm-12pm
    ax.scatter(X_standardized[:, 0], X_standardized[:, 1], c=y_pred)
    plt.show()
