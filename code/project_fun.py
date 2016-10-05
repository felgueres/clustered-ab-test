import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    aggregators = [np.sum, np.mean, np.std]
    return df.groupby([pd.Grouper(freq=frequency)]).agg(aggregators)


# This is how you slice df.loc[:,(slice(None),'sum')]
# trial.loc[:,(slice(1194,1491),'sum')]

if __name__ == '__main__':
    pass
