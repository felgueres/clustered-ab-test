from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def dateDecoder(row):
    '''
    Decodes dates in the following format.
    Five digit code (ex. 19504):
        - Day Code: digits 1-3 (day 1 = 1st January 2009)
        - Time Code: digits 4-5 (1-48 for each 30 minutes with 1 = 00:00:00 - 00:29:59)
    INPUT: None
    Ouput: None
    '''

    # startdate = datetime(year = 2012, month = 1, day = 1)
    delta = timedelta(days =  int(row[:3]), minutes = 30 * int(row[-2:]))

    return delta

def plotter_distribution(df):
    '''
    Computes distribution at specified timeframe.
    '''

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    for user in df.ID.unique().tolist():
        ax.plot(df.ix[df.ID == user].ts, df.ix[df.ID == user].consumption)

def aggregator(df, frequency = '4W'):
    '''
    Computes stats by interval and user defined aggregating functions.
    '''
    aggregators = [np.sum, np.mean, np.std]

    return df.groupby([pd.Grouper(freq=frequency)]).agg(aggregators)


# This is how you slice df.loc[:,(slice(None),'sum')]
# trial.loc[:,(slice(1194,1491),'sum')]




if __name__ == '__main__':
    pass
