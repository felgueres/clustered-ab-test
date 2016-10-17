# kWintessence

This 2-week data science project aims to study household-level responsiveness to time-of-use electricity tariffs.

### Goal and application

The main goal is to show a baseline framework to identify suitable users for demand reduction (driven by their price elasticity) and estimate their capacity to do so.

The motivation of this project and primary application is to identify exploitable strategies for shifting electricity demand from peak hours.

### Overview

#### In a nutshell: demand driven generation -> generation driven demand

The integration of renewable energy generation, foreseeable significant changes in demand (ie. electric cars, storage, CHP) and the motivation to improve power system's efficiency, are driving unprecedented changes in electricity markets.

These forms of interaction underpin an increased volatility for both, demand and supply, making it increasingly complex for stakeholders to plan, manage and optimize capacity utilization of the existing electrical infrastructure.

At a household-level, smart-metering technology is an enabler of a bi-directional communication between suppliers and consumers as well as a means to collect high temporal resolution data. This in turn enables the analysis of intra-day user behavior which, when combined to different stimulus, has the potential to minimize the demand volatility (at the low voltage level) and help reduce the overall network stress at key periods of time.

Smart meters are expected to be fully deployed by 2020 in developed countries. Nevertheless, there are very few public datasets that carry quality and sufficient historical data of smart-metering. The dataset used in this project derives from the CER Smart Metering Project in Ireland (http://www.ucd.ie/issda/data/commissionforenergyregulationcer/), where 4,000 users were monitored and tested with time-of-use tariffs.

Note the CER project aimed to address the household response towards time-of-use tariffs but to the point of this project, did not attempt to quantify and identify the subgroups of users that drive it.

### Data Source

 * Irish Social Science Data Archive: Smart Meter
   * Includes 4,000 anonymized household data
   * household id, timestamp, consumption (kWh)
   * 15-min time-resolution  
   * 6 csv files: 3+ GB

*  Household allocation
   * csv file relating households to Time-of-use Tariff and stimulus

### General approach and challenges

This project can be conceived as a 4-step process.

1) _Feature construction_

Essentially, the consumption of each user at any given time period can be thought of as an independent feature.
On a 15-min granularity and for roughly 4,000 households, it implies a very high dimensionality matrix.
Hence, the first challenge is reducing dimensionality while capturing the households' usage profile in 1) magnitude, and, 2) variability.

2) _k-Means clustering_

The second step is to implement a clustering technique that focuses on capturing subgroups of users load profile.
The value of this step lies in reducing dimensionality and defining a working hypothesis of the consumption of the users:

> _Households within clusters behave similarly under same circumstances, therefore, the baseline for time-of-use tariffs can be estimated by the actual loads of the corresponding control group_.

Helpful enough, this dataset includes a 6-month period where all users where exposed to same conditions and therefore is an unbiased timespan to perform the clustering of all users (benchmark period).
On this line and thinking about the actual application of demand response (DR) applications, the benchmark doesn't need to be an extended period of time, it could be done within days non-event DR days. In fact, clustering within on-off periods of DR events may be more accurate since seasonal usage patterns could be captured through the clustering.


3) _Defining a baseline for comparison_

In order to assess how responsive a subgroup is to a given stimulus, a baseline is required.
The challenge lies in that the actual baseline load of a household is unknown and one can only estimate it.

In this project, the baseline estimation is calculated as a function of the control (clustered) mean, but note that other models such as a regression-based model may increase the accuracy of the estimation (using temperature along with the base load may be a good predictor). Such variations were not explored since this dataset is very limited in demographic information due to privacy concerns.

4) _Quantify the response_

At this point, through visual inspection its possible to see whether a cluster is responsive or not.
Nevertheless, a metric to evaluate how significant the response comes very handy for objectivity.

Assuming that the underlying distributions are Gaussian, a hypothesis test is implemented with a typical type I error of 5% .

 > _H0_: (Time-of-use tariffs cluster)mean >= Baseload  

There is no significant decrease in consumption as a response to increased pricing.

> _H1_: (Time-of-use tariffs cluster)mean < Baseload

Given the density within clusters vary, it is also helpful to compute the statistical power of the test.
And finally, where a subgroup proves to be significant, its possible to quantify the relative magnitude of change, therefore tackling the initial goal:

> __Identifying responsive users and quantifying their ability to contribute in the demand reduction__.

Considerations:
Since there are no negative values for consumption, it is expected for the underlying distributions to be left-skewed.
To help overcome this, the distributions were scaled through a log function.

5) _Insights_

Results are presented through a visual representations and table summarizing cluster-based time-of-use responsiveness.

### Pipeline

Given the 2-week time constraint, this project was conceived as a baseline workflow where additional features were to be implemented as time allowed.
For this reason, the code architecture is designed in a object-oriented way that makes it easier to build-in future complexity and scalability.

There are two main code files associated:

1) 'code/Pipeline' : Contains the PipeLine class from which all the project runs through. Note that similar attributes and methods to the sklearn library were implemented; see the _init_, _transform_, _fit_ methods for documentation.

2) 'code/Utilities': Contains utility functions for plotting, formatting and computing results.

### 1. Data preprocessing, cleaning and reformatting

The data was easy to load and read through pandas.

Preprocessing included:

* Decoding text into workable datetime objects.
* Sorting timestamps, filling missing/erroneous values and pivoting to a user (rows) x features (power consumption) matrix.
* Merging raw data files, merging the _household allocation_ data to main dataframe by user id, dropping invalid users and creating a pickle of the outcome to avoid re-running computations.

#### 2. To be continued...
