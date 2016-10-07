import project_fun as pf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns


if __name__ == '__main__':

    #Load data
    df = pf.load_data()
    df_aggr = pf.sequential_aggregator(df, frequency = '1H')

    '''
    1. OVERALL MEAN DAILY AGGREGATION.
    '''
    #
    #Sequential aggregator
    df_aggr = pf.sequential_aggregator(df, frequency = '1H')
    #Timespan aggregator
    df_fun = pf.timespan(df_aggr)
    #Drop multilevel resulting from previous operation.

    df_fun.columns = df_fun.columns.droplevel(1)


    fig = plt.figure(figsize=(12,4))
    ax_ = fig.add_subplot(1,1,1)

    df_fun.iloc[:,:100].mean(axis=1).plot(ax=ax_)
    df_fun.iloc[:,100:200].mean(axis=1).plot(ax=ax_)
    df_fun.iloc[:,200:300].mean(axis=1).plot(ax=ax_)
    df_fun.iloc[:,400:500].mean(axis=1).plot(ax=ax_)

    # Users aggregator
    # fig = plt.figure(figsize=(10,4))
    # ax = fig.add_subplot(1,1,1)
    # ax.plot(df_fun.mean(axis=1))
    # ax.set_xlabel('time of day (h)')
    # ax.set_ylabel('Consumption (kWh)')
    # plt.title('Mean Daily Consumption')

    plt.show()
    #
    #
    # '''
    # 2. Display distributions for time consumptions.
    # MAX, MIN, MEAN, STD Distributions
    #
    # MAX - Notes: Even though overall the users have a 5 kW demand during the year, that doesn't mean it ocurrs in a frequently basis.
    # WE WANT TO CHECK FOR THAT.
    # '''
    #
    # fig = plt.figure(figsize=(12,4))
    # ax_ = fig.add_subplot(2,2,1)
    # df_aggr.max().plot(ax=ax_, title='Max', kind='hist')
    # ax_ = fig.add_subplot(2,2,2)
    # df_aggr.min().plot(ax=ax_, title='Min', kind='hist')
    # ax_ = fig.add_subplot(2,2,3)
    # df_aggr.mean().plot(ax=ax_, title='Mean', kind='hist')
    # ax_ = fig.add_subplot(2,2,4)
    # df_aggr.sum().plot(ax=ax_, title='Period Consumption', kind='hist')
    #
    # plt.show()
    #
    #
    # '''
    #
    # 3.
    #
    # '''
