import project_fun as pf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns


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

def seasonal():

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

    seasonal()
