import pandas as pd
import numpy as np
import sys

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

if __name__ == '__main__':

    filepath = sys.argv[1]
    trial = usergroups(filepath)
