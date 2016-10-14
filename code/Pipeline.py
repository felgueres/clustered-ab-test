import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Utilities import date_decoder, data_merger, user_group, plot_behavior_cluster, plot_behavior_user, plot_cluster_hist, plot_trial
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm

class PipeLine(object):
    '''
    Cluster-based testing of electrical user behavior towards demand-response initiatives.

    Parameters
    ----------
    path: string
        Path to smart meter data, either to, 1) folder containing multiple raw csv
        files or, 2) pickled Dataframe object resulting from the Transform method of this class.

    pickle: boolean, default True
        If True, expects to import the pickled DataFrame, else the raw files from folder.
        Note: This is convenient since transforming the dataset everytime would not be time efficient.

    usersgroupfile: string
        Path to file with allocation of test and control groups data.

    Attributes
    ----------

    users: pandas Series
        Selected users to use for analysis.

    start_date: datetime
        The reference starting date of the experiment is January 1st 2009.
        Note raw data is encoded to this date.

    benchmark_start: datetime
        All users are on under the same tariff and no demand initiatives have been implemented.
        Starting date is July 1st 2009.

    benchmark_end: datetime
        All users are on under the same tariff and no demand initiatives have been implemented.
        End date is December 31st 2009.

    trial_start: datetime
        Users are under the same tariff and demand initiatives have been implemented.
        Starting date is January 1st 2009.

    trial_start: datetime
        Users are under the same tariff and demand initiatives have been implemented.
        Ending date is October 31st 2009.

    df_bm: DataFrame
        Data during benchmark period.

    df_trial: DataFrame
        Data during trial period.

    X_features: array-like
        Features to cluster.

    kmeans: array-like
        KMeans fitted model to benchmark data, related to X_features.

    y_pred: array-like
        Predicted labels for each user.

    '''

    def __init__ (self, path, pickle_ = True):

        # Read folder files or reload pickle dataframe
        if pickle_:
            # Load pickle file
            self.df = pickle.load(open(path))

        else:
            # load multiple raw files from folder
            self.df = data_merger(path)

        # Selected users from experiment.
        self.users = None

        # Data source.
        self.pickle_ = pickle_

        # All dates are referenced to startdate: January 1st 2009 00:00:00
        self.startdate = datetime(2009, 1,1,0,0)

        # Benchmark period: all users are on under the same tariff and no demand initiatives have been deployed.
        self.benchmark_start = datetime(2009,7,1,0,0)
        self.benchmark_end = datetime(2009,12,31,23,59)

        # Test period: users are allocated to demand response program or control group.
        self.trial_start = datetime(2010,1,1,0,0)
        self.trial_end = datetime(2010,10,31,23,59)

        # DataFrames for benchmark and trial periods:
        self.df_bm = None
        self.df_trial = None

        # X Features
        self.X_features = None

        # KMeans fitted model.
        self.kmeans = None

        # Predicted Labels for each user.
        self.y_pred = None

        # Dict with trial users, labeled and categorized with tariff subgroups.
        self.clustersDict = {}

    def _dates(self):
        '''
        Decodes dates from raw data, sets DataFrame index to datetime objects.

        Parameters
        ----------

        None

        Returns
        -------

        self : object
            Returns self
        '''
        #1. Convert timecode and ID from float to integer to remove '.0'
        self.df[['ID', 'ts']] = self.df[['ID', 'ts']].astype(int)
        #2. Convert from integer to string for processing.
        self.df.ts = self.df.ts.astype(str)
        #3. Apply date Decoder function
        self.df.ts = self.df.ts.apply(date_decoder)
        #4. Calculate day based on starting date reference.
        self.df.ts = self.df.ts + self.start_date
        #5. Sort values by ID and ts since there are uncontinous instances.
        self.df.sort_values(by = ['ID','ts'], inplace = True)
        #6. Pivot table where ID are columns and consumption values.
        self.df = self.df.pivot_table(index='ts', columns = 'ID', values = 'consumption')


    def _usergroup(self):
        '''
        Isolate the dataset as function of usergroup.
        See user_group in utilitiesforcleaning

        Parameters
        ----------
        None

        Returns
        -------
        self : object
            Returns self

        '''

        #1. Selects all residential users
        self.users = user_group('../data/allocations.csv')

        #2. Resets Dataframe for selected users only.

        if self.pickle_ == False:

            self.df = self.df.ix[self.df.ID.isin(self.users.ID)]


    def transform(self):
        '''
        From raw data, select users, decode times and format to usable DataFrame.

        Parameters
        ----------

        None

        Returns
        -------

        self : object
            Returns self

        '''

        # If raw data; label, decode date and format DataFrame.
        if self.pickle_ == False :

            #1. Set column names to dataframe 'Household ID, timestamp, consumption in kWh'
            self.df.columns = ['ID','ts','consumption']

            #2. Reduce dataset to selected group only.
            self._usergroup()

            #3. Transform dates to workable format.
            self._dates()


        # Import users dataframe
        self._usergroup()

        #1.1. DataFrame partition to Benchamark and Trial Periods.
        self.df_bm = self.df[self.benchmark_start:self.benchmark_end]
        self.df_trial = self.df[self.trial_start:self.trial_end]

        #1.2 Trial data is has incomplete data, users are dropped on both DataFrames.
        self.df_bm.dropna(axis= 1, how = 'any', inplace = True)
        self.df_trial.dropna(axis=1, how = 'any', inplace = True)

        #1.3 Drop users with not present on both DataFrames.

        self.df_bm = self.df_bm.T.ix[self.df_bm.T.index.isin(self.df_trial.T.index)].T
        self.df_trial = self.df_trial.T.ix[self.df_trial.T.index.isin(self.df_bm.T.index)].T


    def fit(self, featurization = 'load_profile', num_cluster = 8):

        '''
        Compute features and k-means clustering on Benchmark data.

        Parameters
        ----------

        features_: string, default load_profile
            Computes features on two sets of criteria,
            1) 'load_profile': profile generated by averaging hourly power usage (Baseline case)
            2) 'M-shape': dimensionality reduction by creating consumption and time based features. See notes for detail.

        Returns
        -------

        self : object
            Returns self

        Notes
        -----

        Using the load_profile, the load profile is not normalized because the load
        pattern and variation are equally important.
        This is to ensure that only users with both similar demand shape and magnitude
        are clustered together.

        '''

        if featurization == 'load_profile':
            # Compute hourly means for entire period per user.
            self.X_features = self.df_bm.groupby(self.df_bm.index.hour).mean().T
            # Same for the trial period
            self.df_trial = self.df_trial.groupby(self.df_trial.index.hour).mean().T

        elif featurization == 'M-shape':
            pass
            # X_features = somethingelse...
            # Initiate scaler :
            # scaler = StandardScaler().fit(self.df_bm.T)

        # Initialize K-Means
        model = KMeans(n_clusters = num_cluster, random_state = 10)

        # Fit clusters.
        self.kmeans = model.fit(self.X_features)

        # Predict labels.
        self.y_pred = self.kmeans.predict(self.X_features)

        # Merge labels and users to trial dataset. Merge by ID.
        _ = pd.DataFrame(data=self.y_pred, index = self.df_bm.T.index, columns = ['label'])
        self.df_trial = pd.merge(self.df_trial, _ , left_index=True, right_index=True, how ='inner')
        self.df_trial = pd.merge(self.df_trial, self.users.set_index('ID',drop=True), left_index=True, right_index=True, how = 'inner')

        # Segment trial users by cluster and tariff into Dict.

        for cluster in np.arange(self.kmeans.n_clusters):
            cluster_mask = self.df_trial.label == cluster
            self.clustersDict[cluster] = self.df_trial[cluster_mask]

    def transform_fit(self):
        '''
        Equivalent to running transform and fit.
        '''

        self.transform()
        self.fit()

    def plotter(self, plot_type = 'behavior_cluster'):
        '''
        Plotter of insights and visuals.

        Parameters
        ----------

        plot_type : string, default 'behavior_cluster'

            Specify type of plot.

            1) 'behavior_cluster': Plots centroids of clusters on a single graph; X-axis: Time(hours), Y-axis : Consumption (kWh)
            2) 'behavior_user': Plots individual loads in every cluster; Time(hours), Y-axis: Consumption (kWh)
            3) 'hist_clusters': Plots the number of users in each cluster.

        '''
        if plot_type == 'behavior_cluster':
            plot_behavior_cluster(self.kmeans.cluster_centers_, self.kmeans.n_clusters)

        elif plot_type == 'behavior_user':
            plot_behavior_user(X_featurized = self.X_features, labels = self.y_pred, num_clusters = self.kmeans.n_clusters)

        elif plot_type == 'hist_clusters':
            plot_cluster_hist(self.X_features, self.y_pred, self.kmeans.n_clusters)

        elif plot_type == 'trial1':
            plot_trial(self.clustersDict, self.kmeans.n_clusters, True)

        elif plot_type == 'trial2':
            plot_trial(self.clustersDict, self.kmeans.n_clusters, alltariffs_ = False)


if __name__ == '__main__':
    pass
