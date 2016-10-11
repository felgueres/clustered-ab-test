import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utilitiesforcleaning import date_decoder
from utilitiesforcleaning import data_merger
from utilitiesforcleaning import user_group
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
        Benchmark during the benchmark period.

    df_trial: DataFrame
        Data during trial period.

    '''

    def __init__ (self, path, pickle_ = True , usersgroupfile='../data/allocations.csv'):

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

        #1. Selects users from user criteria.
        self.users = user_group(usersgroupfile)

        #2. Resets Dataframe for selected users only.
        self.df = self.df.ix[self.df.ID.isin(self.users)]

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

        #1.1. DataFrame partition to Benchamark and Trial Periods.
        self.df_bm = self.df[self.benchmark_start:self.benchmark_end]
        self.df_trial = self.df[self.trial_start:self.trial_end]

        #1.2. Drop users with missing data.
        self.df_bm.dropna(axis = 1, how = 'any', inplace = True)
        self.df_trial.dropna(axis=1, how = 'any', inplace = True)

        #1.3 Trial data is has incomplete data, users are dropped on both DataFrames.
        self.df_bm = self.df_bm.loc[:,a.df_trial.columns].columns

    def fit(self, features_ = 'load_profile'):

        '''
        Compute features and k-means clustering on Benchmark data.

        Parameters
        ----------

        features_: string, default load_profile
            Computes features on two sets of criteria,
            1) 'load_profile': profile generated by averaging hourly power usage and,
            2) 'M-shape': dimension reduction, creating consumption and time based features. See notes for detail.

        Returns
        -------

        self : object
            Returns self

        '''

if __name__ == '__main__':
    pass
