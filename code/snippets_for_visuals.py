import project_fun as pf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def plotting_clusters_behavior():
    '''
    Compute features for users.
    '''

    #SCENARIO 1: HOURLY AVERAGES,  24 FEATURES PER USER.

    df = pf.load_data()
    winter = (df.index.month ==12) | (df.index.month <=2)
    summer = (df.index.month >= 6) & (df.index.month <=9)
    autumn = (df.index.month >=9) & (df.index.month <=11)

    df = df[autumn].groupby([pd.Grouper(freq='1H')]).agg(np.sum)

    df = df.groupby(df.index.hour).mean().T
    # Transposed df to ID as rows and hours as features.

    #Detected Outlier 1035
    no_outlier = df.index != 1035
    df = df[no_outlier]

    df = df.subtract(df.mean())
    #Scale data
    Scaler = StandardScaler(with_mean = False, with_std=False)
    X = Scaler.fit_transform(df)
    X_inv = Scaler.inverse_transform(X)

    #Initialize K-Means
    kmeans = KMeans(n_clusters = 6)
    # Compute clusters for each user.
    y_pred = kmeans.fit_predict(X)

    # Lets plot their averaged to see whether there is an actual difference in their behavior.
    fig, axes = plt.subplots(2,3, sharex=True, sharey=True)
    cluster = 0
    for i in range(2):
        for j in range(3):
            cluster_mask = y_pred == cluster
            axes[i,j].plot(kmeans.cluster_centers_[cluster], '--', markersize = 5)
            axes[i,j].plot(df[cluster_mask].T, '.', markersize = 5)
            axes[i,j].plot(np.percentile(X_inv[cluster_mask].T, q=75, axis=1))
            axes[i,j].plot(np.percentile(X_inv[cluster_mask].T, q=25, axis=1))
            cluster += 1

    plt.show()

def features():
    '''
    Compute features for users.
    '''

    #SCENARIO 1: HOURLY AVERAGES ONLY, MEANING 24 FEATURES PER USER.

    df = pf.load_data()
    df = df.groupby([pd.Grouper(freq='1H')]).agg(np.sum)
    df = df.groupby(df.index.hour).mean()
    X = df.T

    kmeans = KMeans(n_clusters = 6)
    y_pred = kmeans.fit_predict(X)

    #Lets plot their average to see whether there is an actual difference in their behavior.
    fig = plt.figure(figsize=(10,4))

    for cluster in xrange(6):
        ax_ = fig.add_subplot(1,1,1)
        ax_.plot(df.T[y_pred == cluster].mean(),label = cluster)

    ax_.set_xlabel('time of day (h)')
    ax_.set_ylabel('Consumption (kWh)')

    plt.title('Mean cluster behaviour')
    plt.legend()
    plt.show()



def seasonal_trend():

    df = pf.load_data()
    df = df.groupby([pd.Grouper(freq='1H')]).agg(np.sum)

    spring = (df.index.month >= 3) & (df.index.month <=5)
    summer = (df.index.month >= 6) & (df.index.month <=9)
    autumn = (df.index.month >=9) & (df.index.month <=11)
    winter = (df.index.month ==12) | (df.index.month <=2)

    seasons = ['spring', 'summer', 'autumn', 'winter']

    fig = plt.figure(figsize=(10,4))

    for i, season in enumerate(seasons):

        ax_ = fig.add_subplot(1,1,1)
        ax_.plot(df[eval(season)].groupby(df[eval(season)].index.hour).mean().mean(axis=1), label = season )

    ax_.set_xlabel('time of day (h)')
    ax_.set_ylabel('Consumption (kWh)')

    plt.title('Seasonal Demand')
    plt.legend()
    plt.show()

def box_plotter():

    '''
    Boxplots of demand distributions (centered) per hour.
    '''

    df = pf.load_data()
    df_aggr = pf.sequential_aggregator(df, frequency = '1H')
    #Drop the extra index from aggregation.
    df_aggr.columns = df_aggr.columns.droplevel(1)
    df_fun = pf.timespan(df_aggr)
    df_fun = df_fun.subtract(df_fun.mean()).T

    fig = plt.figure(figsize=(10,4))
    ax_ = fig.add_subplot(1,1,1)
    ax_.set_xlabel('time of day (h)')
    ax_.set_ylabel('Consumption (kWh)')
    top = 2
    bottom = -1.6
    ax_.set_ylim(bottom, top)
    plt.title('Demand variation of population')

    df_fun.boxplot(ax = ax_ , return_type = 'axes')

    plt.show()


def overall_trend():

    '''
    OVERALL MEAN DAILY AGGREGATION: Hourly and 30-min granularities
    '''

    #Load data
    df = pf.load_data()
    #Sequential aggregator
    df_aggr = pf.sequential_aggregator(df, frequency = '1H')
    #Drop the extra index from aggregation.
    df_aggr.columns = df_aggr.columns.droplevel(1)
    #Timespan aggregator
    df_fun = pf.timespan(df_aggr)
    #Drop multilevel resulting from previous operation.

    fig = plt.figure(figsize=(10,4))
    ax = fig.add_subplot(1,1,1)
    ax.plot(df_fun.mean(axis=1), label = '1-hr agg.')
    ax.set_xlabel('time of day (h)')
    ax.set_ylabel('Consumption (kWh)')
    plt.title('Mean Daily Consumption')

    df_fun2 = pf.timespan(df)
    ax.plot(df_fun2.mean(axis=1), label = '30-min agg.')

    plt.legend()
    plt.show()

def daily_agg():

    '''
    OVERALL MEAN DAILY AGGREGATION: Hourly and 30-min granularities
    '''

    #Load data
    df = pf.load_data()
    #Sequential aggregator
    df_aggr = pf.sequential_aggregator(df, frequency = '1h')
    #Drop the extra index from aggregation.
    df_aggr.columns = df_aggr.columns.droplevel(1)
    #Timespan aggregator
    df_fun = pf.timespan(df_aggr)
    #Drop multilevel resulting from previous operation.

    fig = plt.figure(figsize=(10,4))
    ax = fig.add_subplot(1,1,1)
    ax.plot(df_fun.mean(axis=1), label = '1-hr agg.')
    ax.set_xlabel('month')
    ax.set_ylabel('kWh')
    plt.title('Mean Daily Consumption')

    plt.legend()
    plt.show()


def stats_overview():
    '''
    2. General Statistics
    MAX, MIN, MEAN, STD Distributions

    MAX - Notes: Even though overall the users have a 5 kW demand during the year,
    that doesn't mean it ocurrs in a frequently basis.
    WE WANT TO CHECK FOR THAT.

    '''
    #Load data
    df = pf.load_data()
    #Sequential aggregator
    df_aggr = pf.sequential_aggregator(df, frequency = '1H')
    #Drop the extra index from aggregation.
    df_aggr.columns = df_aggr.columns.droplevel(1)

    fig = plt.figure(figsize=(12,4))
    ax_ = fig.add_subplot(2,2,1)
    df_aggr.max().plot(ax=ax_, title='Max', kind='hist')
    ax_ = fig.add_subplot(2,2,2)
    df_aggr.min().plot(ax=ax_, title='Min', kind='hist')
    ax_ = fig.add_subplot(2,2,3)
    df_aggr.mean().plot(ax=ax_, title='Mean', kind='hist')
    ax_ = fig.add_subplot(2,2,4)
    df_aggr.sum().plot(ax=ax_, title='Period Consumption', kind='hist')

    plt.show()

if __name__ == '__main__':
    plotting_clusters_behavior()
