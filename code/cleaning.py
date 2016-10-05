import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utilitiesforcleaning import date_decoder
from utilitiesforcleaning import data_merger
from select_usergroup import user_group
from datetime import datetime, timedelta

class SmartMeter(object):
    '''
    Cleaner function to create working dataset.

    '''

    def __init__ (self, folderpath, usersgroupfile='../data/allocations.csv'):

        # Read fileinto dataframe
        self.df = data_merger(folderpath)

        # self.df = pd.read_csv(folderpath, delim_whitespace=True)

        #sergroup
        self.users = usersgroupfile

        # All dates are referenced to startdate: January 1st 2009 00:00:00
        self.startdate = datetime(2009, 1,1,0,0)


    def _dates(self):
        '''
        Returns timedelta decoded.

        INPUT: None
        Ouput: None
        '''
        #Convert timecode and ID from float to integer to remove '.0'
        self.df[['ID', 'ts']] = self.df[['ID', 'ts']].astype(int)
        # Convert from integer to string for processing.
        self.df.ts = self.df.ts.astype(str)
        # Apply date Decoder function
        self.df.ts = self.df.ts.apply(date_decoder)
        # Calculate day based on starting date reference.
        self.df.ts = self.df.ts + self.startdate
        # Transform to DatetimeIndex object for manipulation capabilities.
        # self.df.ts = pd.DatetimeIndex(self.df.ts)
        # Sort values by ID and ts since there an uncontinous instances.
        self.df.sort_values(by = ['ID','ts'], inplace = True)
        #Pivot table where ID are columns and consumption is values.
        self.df = self.df.pivot_table(index='ts', columns = 'ID', values = 'consumption')

    def _usergroup(self):
        '''
        Isolate the dataset related to usergroup.

        Input: None
        Output: None
        '''

        # Segments users by selected group.
        self.users = user_group(self.users)

        # Redefine df for sellected users only.
        self.df = self.df.ix[self.df.ID.isin(self.users)]

    def transform(self):
        '''
        Helper function to transform data to usable format.

        Input: None
        Output: X
        '''
        #1 Set column names to dataframe 'Household ID, timestamp, consumption in kWh'
        self.df.columns = ['ID','ts','consumption']

        #2. Reduce dataset to selected group only.
        self._usergroup()

        #3. Transform dates to workable format.
        self._dates()

if __name__ == '__main__':
    pass
