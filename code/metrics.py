from scipy import stats as sc
from scipy.stats import ttest_ind
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu

def AB1(k_model, clustersDict):

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

    #Create dict with p-value findings and power findings.

    for cluster in clustersDict:

        df = clustersDict[cluster]
        df = df.ix[df.Residential_Tariff.isin(tariffs)]
        df.Residential_Tariff = df.Residential_Tariff.apply(lambda x: 'Control' if x == 'E' else 'Trial')

        _df_Control = df.ix[df.Residential_Tariff == 'Control'].iloc[:,:-3].T
        _df_Trial = df.ix[df.Residential_Tariff == 'Trial'].iloc[:,:-3].T

        for time in timeofuse:

            control = _df_Control.iloc[timeofuse[time][0]:timeofuse[time][1]+1,:].sum()
            trial = _df_Trial.iloc[timeofuse[time][0]:timeofuse[time][1]+1,:].sum()


            fig = plt.figure()
            ax_ = fig.add_subplot(1,1,1)

            # control_ = np.log(control)
            # trial_ = np.log(trial)

            control.plot(kind = 'kde', ax= ax_, alpha = 0.5 )
            trial.plot(kind = 'kde', ax=ax_, alpha = 0.5)

            ax_.set_title('Cluster %d: %s' % (cluster+1, time))
            ax_.set_xlim((1,5))
            ax_.set_ylim([0, 0.6])
            ax_.set_xlabel('Consumption (kWh)')
            # ax_.set_ylabel("Number of users")

            plt.show()

            print 'Cluster %d, %s p-value:' % ((cluster +1), time), ttest_ind(control, trial, equal_var=False)[1], 'power: ', stat_power(control, trial), 'magnitude: ', np.mean(trial)/np.mean(control) -1

def stat_power(control, trial, ci=0.975):
    '''
    Calculates statistical power.

    Parameters
    ----------
    control : array-type
        Control population sample.

    trial: array-type
        Trial population sample.

    Returns
    -------
    Float

    '''

    # Calculate the mean, se and me-(4 std)
    control = np.log(control)
    trial = np.log(trial)

    control_mean = np.mean(control)
    trial_mean = np.mean(trial)

    control_se = np.std(control, ddof=1) / np.sqrt(control.shape[0])
    trial_se = np.std(trial, ddof=1) / np.sqrt(trial.shape[0])

    # Create a normal distribution based on mean and se
    null_norm = sc.norm(control_mean, control_se)
    alt_norm = sc.norm(trial_mean, trial_se)

    # Calculate the rejection values (X*)
    reject_low = null_norm.ppf((1 - ci) / 2)
    reject_high = null_norm.ppf(ci + (1 - ci) / 2)

    # Calculate power
    power_lower = alt_norm.cdf(reject_low)
    power_higher = 1 - alt_norm.cdf(reject_high)
    power = (power_lower + power_higher) * 100
    return power
