# kWintessence

This 2-week data science project aims to study household-level responsiveness to time-of-use electricity tariffs.

### Goal and application

The main goal is to show a baseline framework to identify suitable users for demand reduction (driven by their price elasticity) and estimate their capacity to do so.

The motivation of this project and primary application is to identify exploitable strategies for shifting electricity demand from peak hours.

### Overview 

The integration of renewable energy generation, foreseeable significant changes in demand (ie. electric cars, storage, CHP) and the motivation to improve power system's efficiency, are driving unprecedented changes in electricity markets.

These forms of interaction underpin an increased volatility for both, demand and supply, making it increasingly complex for stakeholders to plan, manage and optimize capacity utilization of the existing electrical infrastructure.

At a household-level, smart-metering technology is an enabler of a bi-directional communication between suppliers and consumers as well as a means to collect high resolution data. This in turn enables the analysis of intra-day user behavior which, when combined to different stimulus, has the potential to minimize the demand volatility (at the low voltage level) and help reduce the overall network stress at key periods of time.

Smart meters are expected to be fully deployed by 2020 in developed countries. Nevertheless, there are very few public datasets that carry quality and sufficient historical data of smart-metering. The dataset used in this project derives from the CER Smart Metering Project in Ireland (http://www.ucd.ie/issda/data/commissionforenergyregulationcer/), where 4,000 users were monitored and tested with time-of-use tariffs.

Note the CER project aimed to address the household response towards time-of-use tariffs but to the point of this project, did not attempt to quantify and identify the subgroups of users that drive it.

### Data Source

 * Irish Social Science Data Archive: Smart Meter
   * Includes 4,000 anonymized household data
   * household id, timestamp, consumption (kWh)
   * 15-min time-resolution  
   * 6 csv files: 3+ GB total

*  Household allocation
   * csv file relating households to Time-of-use Tariff and Demand Response stimulus

### General approach and challenges

This project can be conceived as a 4-step process.

1) Feature construction

Essentially, the consumption of each user at any given time period can be thought of as an independent feature.
On a 15-min granularity and for roughly 4,000 households, it implies a very high dimensionality matrix.
Hence, the first challenge is reducing dimensionality while capturing the households' usage profile in 1) magnitude, 2) variability.

2) k-Means clustering

The second step is to implement a clustering technique that focuses on capturing subgroups of users that share the same behavior both in magnitude and variability. Note that feature construction of step one is crucial for this step's success.
This dataset includes a 6-month period where all users where exposed to same conditions and therefore is an unbiased timespan to perform the clustering of all users (benchmark period).

3) Defining a baseline for comparison

In order to assess how responsive a subgroup is to a given stimulus, a baseline is required.
The challenge lies in that the actual baseline load of a household is unknown and one can only estimate it.
As common sense and data suggests, a combination of historical consumption and weather data would be very strong proxies to estimate this value through a regression-based model.  

However, due privacy concerns, this dataset doesn't include any location information to model with.
Thus, the features used to cluster (and therefore also features constructed at 1) are constrained to be a representation of the overall magnitude and variability of a household. This means, no centering nor normalization of the data can be performed, decreasing the capability of capturing relative as opposed to absolute changes.

4) Quantify the response



### Pipeline

Developing a more robust form of this


### 1. Data preprocessing, cleaning and reformatting

The data was easy to load and read through pandas.

Preprocessing included:

* Decoding text into workable datetime objects.
* Sorting timestamps, filling missing/erroneous values and pivoting to a user (rows) x features (power consumption) matrix.
* Merging raw data files, merging the _household allocation_ data to main dataframe by user id, dropping invalid users and creating a pickle of the outcome to avoid re-running computations.

#### 2. To be continued...
