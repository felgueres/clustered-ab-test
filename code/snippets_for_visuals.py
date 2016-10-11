import project_fun as pf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm


def plotting_clusters_behavior():
    '''
    Compute features for users.
    '''

    #SCENARIO 1: HOURLY AVERAGES,  24 FEATURES PER USER.

    df = pf.load_data()
    winter = (df.index.month ==12) | (df.index.month <=2)
    summer = (df.index.month >= 6) & (df.index.month <=9)
    autumn = (df.index.month >=9) & (df.index.month <=11)
    df = df[winter]
    # Time aggregator
    # df = df[autumn].groupby([pd.Grouper(freq='1H')]).agg(np.sum)

    df = df.groupby(df.index.time).mean().T
    # Transposed df to ID as rows and hours as features.

    #Detected Outlier 1035
    no_outlier = df.index != 1035
    df = df[no_outlier]

    #Scale data
    Scaler = StandardScaler()
    X = Scaler.fit_transform(df)

    X_inv = Scaler.inverse_transform(X)
    #Initialize K-Means
    kmeans = KMeans(n_clusters = 8)
    # Compute clusters for each user.
    y_pred = kmeans.fit_predict(X)

    # Lets plot their averaged to see whether there is an actual difference in their behavior.
    fig, axes = plt.subplots(2,4, sharex=True, sharey=True)

    cluster = 0
    for i in range(2):
        for j in range(4):
            cluster_mask = y_pred == cluster
            axes[i,j].plot(kmeans.cluster_centers_[cluster], '--', markersize = 10)
            axes[i,j].plot(df[cluster_mask].T, '.', markersize = 3)
            axes[i,j].plot(np.percentile(df[cluster_mask].T, q=75, axis=1))
            axes[i,j].plot(np.percentile(df[cluster_mask].T, q=25, axis=1))
            cluster += 1

    plt.show()

def silloute_test():

    df = pf.load_data()
    winter = (df.index.month ==12) | (df.index.month <=2)
    df = df[winter]
    df = df.groupby(df.index.time).mean().T
    no_outlier = df.index != 1035
    df = df[no_outlier]
    Scaler = StandardScaler(with_mean = False, with_std=False)
    X = Scaler.fit_transform(df)

    range_n_clusters = [3, 5, 7]

    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-1,-0.8,-0.6,-0.4,-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors)

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1],
                    marker='o', c="white", alpha=1, s=200)

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50)

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')

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
    df_aggr = df.groupby([pd.Grouper(freq='1H')]).agg(np.sum)
    #Drop the extra index from aggregation.
    df_aggr.columns = df_aggr.columns.droplevel(1)
    df_fun = df_aggr.groupby([df.index.hour]).mean()
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
    df_aggr = df.groupby([pd.Grouper(freq='1H')]).agg(np.sum)
    #Drop the extra index from aggregation.
    df_aggr.columns = df_aggr.columns.droplevel(1)
    #Timespan aggregator
    df_fun = df_aggr.groupby([df.index.hour]).mean()
    #Drop multilevel resulting from previous operation.

    fig = plt.figure(figsize=(10,4))
    ax = fig.add_subplot(1,1,1)
    ax.plot(df_fun.mean(axis=1), label = '1-hr agg.')
    ax.set_xlabel('time of day (h)')
    ax.set_ylabel('Consumption (kWh)')
    plt.title('Mean Daily Consumption')

    df_fun2 = df.groupby([df.index.hour]).mean()
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
    df_aggr = df.groupby([pd.Grouper(freq='1H')]).agg(np.sum)
    #Drop the extra index from aggregation.
    df_aggr.columns = df_aggr.columns.droplevel(1)
    #Timespan aggregator
    df_fun = df_aggr.groupby([df.index.hour]).mean()
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
    df_aggr = df.groupby([pd.Grouper(freq='1H')]).agg(np.sum)
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
