import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta


def load_data(picklepath='../data/working_data.pickle', start_date = datetime(2010,1,1), end_date = datetime(2010,12,31,23,59)):
    '''
    Loads data from pickle at given timespan.
    '''

    df = pickle.load(open(picklepath))
    # Set analyzed timeperiod
    df = df[start_date:end_date]
    #Remove IDs if nulls.
    df.dropna(axis = 1, how = 'any', inplace = True)

    return df

def sequential_aggregator(df, frequency = '1h'):
    '''
    Computes stats by frequency interval and user defined aggregating functions.

    INPUT: DataFrame, frequency
    OUTPUT: DataFrame

    '''
    # aggregators = [np.sum, np.mean, np.std]
    aggregators = [np.sum]
    return df.groupby([pd.Grouper(freq=frequency)]).agg(aggregators)

def timespan(df):
    '''
    Apply function to group across during a given timespan, returns a DataFrame.
    Ex. 7-9am for all year for specific group.
    '''

    return df.groupby([df.index.hour]).mean()


def seasonal_feature(df):
    seasons = {1: 'winter', 2:'spring', 3:'summer', 4:'autumn'}

    autumn = datatime(month=8)
    
    pass


def elbows(X_standardized):
    '''
    Calculate distances to centroids as score, returns plot score vs. # clusters.
    INPUT: X standardized (numpy array)
    OUTPUT: matplotlib object
    '''

    SSE_varying_k = []
    k_list = range(1,10)

    for k in k_list:

        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X_standardized)
        SSE_varying_k.append(abs(kmeans.score(X_standardized)))

    fig = plt.figure(figsize=(10,4))
    ax = fig.add_subplot(1,1,1)
    ax.plot(k_list, SSE_varying_k)
    ax.set_xlabel('Varying k')
    ax.set_ylabel('Score')

    plt.show()

def cluster_plotter(X_standardized, k=5):
    '''
    Plot combination of cluster plots.
    INPUT: Features matrix (np array)
    OUTPUT: matplotlib object
    '''

    SSE_varying_k = []
    k_list = range(1,k)
    fig = plt.figure(figsize=(14,10))

    for k in k_list:

        kmeans = KMeans(n_clusters=k)
        y_pred = kmeans.fit_predict(X_standardized)

        ax = fig.add_subplot(1,4,k)
        ax.scatter(X_standardized[:, 0], X_standardized[:, 1], c=y_pred)

    plt.show()

if __name__ == '__main__':
    pass

    # #First cluster is based on the mean of 2 hour aggregation.
    # df_sum = df_aggr.loc[:,(slice(None),'sum')]
