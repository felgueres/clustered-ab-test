import pandas as pd
import numpy as np
import os
import glob
import sys
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from scipy import stats as scs
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu

def data_merger(folderpath):
    '''
    Returns dataframe of concatenated raw files.

    Parameters
    ----------
    folderpath : string
        Path to folder containing raw files.

    Return
    ------
    df : pandas DataFrame

    Notes:
    Transforms and concatenates directly through Numpy arrays
    as opposed to pandas DataFrame for time and computational performance.

    '''

    # Get list of file names in raw folder.
    filelist = glob.glob(os.path.join(folderpath,'*.txt'))

    # Create a list of np arrays.
    np_array_list = []

    # read them into pandas
    for file in filelist:
        df = pd.read_csv(file, header=0, delim_whitespace=True)
        np_array_list.append(df.as_matrix())

    #Merge arrays together.
    stacked_ = np.vstack(np_array_list)

    # Create dataframe
    df = pd.DataFrame(stacked_)

    return df

def user_group(filepath, code =1 , residential_stimulus='E', residential_tariff='E'):
    '''
    Return group of users.
    If parameters are not passed, Control Group is set as default.
    Returns a list of users.

    Parameters
    ----------

    filepath : string
        Path to CSV file with usergroups.

    Code : int, default 1
        See Notes for datakey.

    residential_stimulus : string, default 'E'
        See Notes for datakey.

    residential_tariff : string, default 'E'
        See Notes for datakey.

    Returns
    -------

    users_id : pandas Series
        Residential users ids.

    users: pandas DataFrame
        Users data.

    Notes
    -----

    1   Residential
    2   SME
    3   Other

    Residential Stimulus:

    E   Control
    1   Bi-monthly detailed bill
    2   Monthly detailed bill
    3   Bi-monthly detailed bill +IHD
    4   Bi-monthly detailed bill +OLR
    W   Weekend tariff

    Residential Tariff:

    E   Control
    A   Tariff A
    B   Tariff B
    C   Tariff C
    D   Tariff D
    W   Weekend Tariff

    SME :

    1   Monthly detailed bill
    2   Bi-monthly detailed bill +IOD
    3   Bi-monthly detailed bill +web-access
    4   Bi-monthly detailed bill
    C   Control

    '''
    df_users = pd.read_csv(filepath)
    # df = df.ix[(df.Residential_Tariff == residential_tariff) & (df.Residential_stimulus == residential_stimulus) & (df.Code == code)]
    df_users = df_users.ix[(df_users.Code == code)]
    df_users.drop(['SME_allocation','Code'], axis = 1, inplace=True)

    return df_users

def date_decoder(date):
    '''
    Decodes dates from original format, returns time delta objects.

    Parameters
    ----------

    date : string
        Encoded date, see datakey in Notes below.

    Returns
    -------

    delta : datatime.timedelta

    Notes
    -----

    Five digit code (ex. 19504):
        - Day Code: digits 1-3 (day 1 = 1st January 2009)
        - Time Code: digits 4-5 (1-48 for each 30 minutes with 1 = 00:00:00 - 00:29:59)

    '''

    delta = timedelta(days =  int(row[:3]), minutes = 30 * int(row[3:5]))

    return delta

def M_shape(X):
    '''
    Construction of 6 consumption and time-based features.

    Parameters
    ----------

    X : array
        Data to calculate features.

    Returns
    -------

    X_M : array
        X transformed in new featured space.

    Notes
    -----

    The average daily usage in the data set demonstrates on morning peak and one evening peak.
    Capturing consumption values at these two peaks characterizes household's patterns.

    THIS FEATURE WILL BE DEVELOPED IF TIME ALLOWS.

    '''

    pass

def plot_behavior_cluster(centroids, num_clusters):
    '''
    Plots computed clusters.

    Parameters
    ----------

    Centroids : array
        Predicted centroids of clusters.

    num_clusters: int
        Number of clusters.

    Returns
    -------

    Plot : matplotlib.lines.Line2D
        Figure.

    '''

    # Figure has all clusters on same plot.

    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(1,1,1)

    # Set colors.
    colors = cm.rainbow(np.linspace(0, 1, num_clusters))

    # Plot cluster and corresponding color.
    for cluster, color in enumerate(colors, start =1):

        ax.plot(centroids[cluster-1], c = color, label = "Cluster %d" % cluster)

    # Format figure.
    ax.set_title("Centroids of consumption pattern of clusters, where k = %d" % num_clusters, fontsize =14, fontweight='bold')
    ax.set_xlim([0, 24])
    ax.set_xticks(range(0, 25, 6))
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Consumption (kWh)")
    leg = plt.legend(frameon = True, loc = 'upper left', ncol =2, fontsize = 12)
    leg.get_frame().set_edgecolor('b')

    plt.show()

def plot_behavior_user(X_featurized, labels, num_clusters):
    '''
    Plot individual customer loads for every cluster.

    Parameters
    ----------

    X_featurized : array-like
        Featurized Data

    labels: array-like
        Predicted cluster to data.

    num_clusters: int
        Number of clusters.

    Returns
    -------

    Plot : matplotlib.lines.Line2D
        Figure.

    '''
    # The figure will have 3 clusters per column and num_clusters/3 rows.
    plots_per_col = 2
    plots_per_row = num_clusters/plots_per_col

    # Creates axes, one per cluster.
    fig, axes = plt.subplots(plots_per_col, plots_per_row, sharex=True, sharey=True)

    # Define colors.
    colors = cm.rainbow(np.linspace(0, 1, num_clusters))

    #Initialize counter
    cluster_counter = 0
    # ,

    for i in range(plots_per_col):
        for j in range(plots_per_row):
            # Create mask to isolate users corresponding to each cluster.
            cluster_mask = labels == cluster_counter
            axes[i,j].plot(X_featurized.columns, X_featurized[cluster_mask].T, c=colors[cluster_counter], alpha = 0.015)
            axes[i,j].set_title('Cluster %d ' % (cluster_counter+1))
            axes[i,j].set_xlim([0, 24])
            axes[i,j].set_ylim([0, 3.2])
            axes[i,j].set_xticks(range(0, 25, 6))
            cluster_counter += 1

    # Set common labels
    fig.text(0.5, 0.04, 'Time (h)', ha='center', va='center')
    fig.text(0.06, 0.5, 'Consumption (kWh)', ha='center', va='center', rotation='vertical')

    plt.subplots_adjust(wspace=0.1)
    # Set title to figure
    fig.suptitle("Individual customer loads, where k = %d" % num_clusters, fontsize = 14, fontweight = 'bold')
    plt.show()

    #save figs

def plot_cluster_hist(X_featurized, labels, num_clusters):
    '''
    Plot histograms of users and corresponding clusters.

    Parameters
    ----------

    X_featurized : array-like
        Featurized Data

    labels: array-like
        Predicted cluster to data.

    num_clusters: int
        Number of clusters.

    Returns
    -------

    Plot : matplotlib.lines.Line2D
        Figure.

    '''

    fig = plt.figure()
    ax_ = fig.add_subplot(1,1,1)

    # Set colors.
    # colors = cm.rainbow(np.linspace(0, 1, num_clusters))

    # Create DataFrame with features and labels.
    # Note sklearn cluster naming starts at zero, so adding 1 is convenient.

    X_featurized['label'] = labels + 1
    # Parameters for plotting.
    params_ = {'ax': ax_ , 'bins': np.arange(num_clusters +2) - 0.5}
    # Plot cluster and corresponding color.
    X_featurized.label.plot(kind = 'hist', **params_)

    # Format figure.

    ax_.set_title("Number of users in each cluster.", fontsize =14, fontweight='bold')
    ax_.set_xticks(range(1, num_clusters +1))
    ax_.set_xlim([0,num_clusters + 1])
    ax_.set_ylim([0,1200])
    ax_.set_xlabel('Cluster')
    ax_.set_ylabel("Number of users")

    plt.show()

def plot_trial(clustersDict, num_clusters, alltariffs_ = True):
    '''
    Plots consumption durin trial period for DR and Control groups to compare the effect of initiatives.

    Parameters
    ----------

    clustersDict : dict
        Contains trial users segmented by cluster.
        Key = cluster
        Values = DataFrame with all features

    num_clusters: int
        Number of clusters.

    alltariffs_: Boolean, default False
        Use when each tariff needs comparison. When False, compares Control vs. all other tariffs.

    tariffsDict: dict
        Contains pricing for each tariff.
        Each value is a list with pricing corresponding to Day, Night and Peak, Time-of-use tariffs.

    Returns
    -------

    Plot : matplotlib.lines.Line2D
        Figure.

    '''

    # The figure will have 3 clusters per column and num_clusters/3 rows.
    plots_per_col = 2
    plots_per_row = num_clusters/plots_per_col

    # Creates axes, one per cluster.
    fig, axes = plt.subplots(plots_per_col, plots_per_row, sharex=True, sharey=True)

    # #Set size
    fig.set_size_inches([12,12])

    # Define colors.
    colors = cm.rainbow(np.linspace(0, 1, num_clusters))

    #Initialize counter
    cluster_counter = 0

    # X_range
    x_range = [0,24]

    # Time-of-use tariffs
    # The value in each key of the following timeofuse dict is [hour range min, hour range max
    # color to shade, tariff pricing for [A, B, C, D]]

    timeofuse = {'day': [8,17,'blue',[14,13.5,13,12.5]], 'peak':[17,19,'magenta',[20,26,32,38]], \
                'night': [0,8,'green',[12,11,10,9]], 'day2':[19,24,'blue',[14,13.5, 13 ,12.5]]}

    #Create dictionary with usercount per tariff per cluster.

    userDict = {}

    for i in range(plots_per_col):
        for j in range(plots_per_row):
            # Create mask to isolate users corresponding to each cluster.
            df = clustersDict[cluster_counter]

            # Tariffs

            tariffs = ['Control', 'A', 'B', 'C', 'D']

            #__________________PLOT CONTROL USERS__________________________

            if alltariffs_:
                # Tarriff 'E' is equivalent to 'Control', lets modify that.
                df.Residential_Tariff = df.Residential_Tariff.apply(lambda x: 'Control' if x == 'E' else x)

            else:

                user_counts = df.ix[df.Residential_Tariff.isin(tariffs)].Residential_Tariff.value_counts().sort_index()
                df.Residential_Tariff = df.Residential_Tariff.apply(lambda x: 'Control' if x == 'E' else ('Trial' if x in tariffs[1:] else x))
                tariffs = ['Control', 'Trial']

            #___________________PLOT TARIFF USERS___________________________

            for tariff in tariffs:

                sample_size = df.ix[df.Residential_Tariff == tariff].shape[0]
                df_tariffs = df.ix[df.Residential_Tariff == tariff].iloc[:,:-3].T.mean(axis=1)

                # Aside to the tariff label, also include sample size.
                axes[i,j].plot(df_tariffs, label = tariff + ': %d' %sample_size )

            #___________________SHADE TIME-OF-USE TARIFFS____________________
            # Set a timespan space to plot the tariff function.
            x_space = np.linspace(0,23,1000)
            # Set default array of zeros
            y = np.zeros(1000)

            #USE fill_between function to shade between two functions, in this case, time periods.
            for time in timeofuse:

                axes[i,j].fill_between(x = [timeofuse[time][0],timeofuse[time][1]], y1=0,y2=4, alpha=0.1, facecolor = timeofuse[time][2])
                # For the aggregate level only, lets compute the expected overall tariff.
                # Since the pricing for control is constant, we can graph the function to check responsiveness in the trial.

                if alltariffs_ == False :

                    #Weight the tariff for the specific period:
                    weighted_average_tariff = np.sum(np.multiply(timeofuse[time][3], user_counts.values))/np.sum(user_counts.values)

                    #Plot it in the secondary axis.
                    # y[(x_space >= timeofuse[time][0]) & (x_space <= timeofuse[time][1])] = weighted_average_tariff
                    y[(x_space >= timeofuse[time][0]) & (x_space <= timeofuse[time][1])] = weighted_average_tariff
                    #

            #Create temporary axis.
            if alltariffs_ == False:
                secondary_ax = axes[i,j].twinx()
                secondary_ax.plot(x_space, y, linewidth=1, linestyle='--', color='red', alpha=0.3)
                secondary_ax.set_ylim([10, 50])
                plt.setp(secondary_ax.get_yticklabels(), visible=False)

            #_______________ ADD SOME FORMATING TO AXIS AND ADD LABELS._________

            axes[i,j].set_title('Cluster %d ' % (cluster_counter+1))
            axes[i,j].set_xlim(x_range)
            axes[i,j].set_ylim([0, 2])
            axes[i,j].set_xticks(range(0, 25, 3))

            # Insert patch with time of use color code on last plot.
            if cluster_counter == num_clusters -1:
                # Create patches to denote the time-of-use tariffs.
                peak_patch = mpatches.Patch(color='magenta', label='peak', alpha = 0.1)
                day_patch = mpatches.Patch(color='blue', label='day', alpha=0.1)
                night_patch = mpatches.Patch(color='green', label='night', alpha = 0.1)

                if alltariffs_ == False:
                    time_of_use = mlines.Line2D([], [], color='red', linestyle = '--', label='DR tarriff', alpha=0.3)
                    time_legends = plt.legend(handles=[peak_patch, day_patch, night_patch,time_of_use], loc=4, frameon = True, ncol =1)

                else:
                    time_legends = plt.legend(handles=[peak_patch, day_patch, night_patch], loc=4, frameon = True, ncol =1)

                #manually create time-of-use-patch labels
                ax_ = plt.gca().add_artist(time_legends)

                #Create trial vs control group labels.
                axes[i,j].legend(frameon=True, loc = 'upper left', ncol =2)

            else:
                axes[i,j].legend(frameon=True, loc = 'upper left', ncol =2)

            cluster_counter += 1

    # Set common labels
    fig.text(0.5, 0.04, 'Time (h)', ha='center', va='center')
    fig.text(0.06, 0.5, 'Consumption (kWh)', ha='center', va='center', rotation='vertical')
    plt.subplots_adjust(wspace=0.1)

    # Set title to figure
    fig.suptitle("Control vs. Time-of-use Tariffs, where k = %d" % num_clusters, fontsize = 14, fontweight = 'bold')
    plt.show()

def AB(k_model, clustersDict):

    '''
    Computes AB testing on clustered samples.

    Parameters
    ----------

    k_model : sklearn.KMEANS
        Trained Kmeans model.

    data: dict
        Dictionary containing DataFrames for all clusters.

    Returns
    -------

    Plot : matplotlib.lines.Line2D
        Figure.

    '''
    tariffs = ['E', 'A', 'B', 'C', 'D']

    timeofuse = {'day': [8,17], 'peak':[17,19], 'night': [0,8], 'day2':[19,24]}

    for cluster in clustersDict:

        df = clustersDict[cluster]
        df = df.ix[df.Residential_Tariff.isin(tariffs)]
        df.Residential_Tariff = df.Residential_Tariff.apply(lambda x: 'Control' if x == 'E' else 'Trial')
        # df = df.ix[df.Residential_Tariff == tariffs].iloc[:,:-3].T.mean(axis=1)

        _df_Control = df.ix[df.Residential_Tariff == 'Control'].iloc[:,:-3].T
        _df_Trial = df.ix[df.Residential_Tariff == 'Trial'].iloc[:,:-3].T

        print 'Cluster %d peak p-value:' % (cluster +1), ttest_ind(_df_Control.iloc[17:20,:].sum(), _df_Trial.iloc[17:20,:].sum(), equal_var=False)
        print 'Cluster %d peak p-value:' % (cluster +1), mannwhitneyu(_df_Control.iloc[17:20,:].sum(), _df_Trial.iloc[17:20,:].sum(), alternative = 'greater')

if __name__ == '__main__':

    pass