import pandas as pd
import numpy as np
import os
import glob
import sys
from datetime import datetime, timedelta

def data_merger(folderpath):
    '''
    Concatenates raw files in specified foldername to dataframe.
    Uses numpy for performance.

    INPUT: String
    OUTPUT: DataFrame
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
    Segment users by being in the control or trial population.
    If parameters are not passed, Control Group is set as default.
    Returns a list of users.

    Input: CSV file (allocation)
    Output: list

    Data Key:

    Code:

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

    df = pd.read_csv(filepath)
    df = df.ix[(df.Residential_Tariff == residential_tariff) & (df.Residential_stimulus == residential_stimulus) & (df.Code == code)]

    users_list = df.ID
    return users_list

def date_decoder(row):
    '''
    Decodes dates in the following format.
    Five digit code (ex. 19504):
        - Day Code: digits 1-3 (day 1 = 1st January 2009)
        - Time Code: digits 4-5 (1-48 for each 30 minutes with 1 = 00:00:00 - 00:29:59)
    INPUT: None
    Ouput: None
    '''

    # startdate = datetime(year = 2009, month = 1, day = 1)
    # if len(row) > 5:
    #     row = row[:5]

    delta = timedelta(days =  int(row[:3]), minutes = 30 * int(row[3:5]))

    return delta


if __name__ == '__main__':

    pass
