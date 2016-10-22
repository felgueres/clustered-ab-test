# kWintessence

This 2-week data science project aims to study household-level responsiveness to time-of-use electricity tariffs.

The goal is to show a baseline framework to identify responsive users towards demand reduction strategies (in this case driven by their consumption elasticity).

### Context

The integration of renewable energy generation, foreseeable significant changes in demand (ie. electric cars, storage, CHP) and the motivation to improve power system's efficiency are driving unprecedented changes in electricity markets.

These forms of interaction increase generation-consumption volatility, making it increasingly complex for stakeholders to optimize capacity utilization of electrical infrastructure.

At a household-level, smart-metering technology is a means to collect high resolution temporal data.
The analysis of intra-day user behavior opens the possibility to tackle demand volatility and help balance the grid.

Smart meters are expected to be fully deployed by 2020 in developed countries.
Nevertheless, there are very few public datasets with quality historical data of smart-metering.
The dataset used in this project derives from the CER Smart Metering Project in Ireland (http://www.ucd.ie/issda/data/commissionforenergyregulationcer/), where users were monitored and tested with time-of-use tariffs.

Note the CER project aimed to address the household response towards time-of-use tariffs but to the point of this project, did not attempt to identify the subgroups of users that drive it.

### Data Source

 * Irish Social Science Data Archive: Smart Meter
   * Includes 4,000 anonymized household data
   * household id, timestamp, consumption (kWh)
   * 15-min time-resolution  
   * 6 csv files: 3+ GB

*  Household allocation
   * csv file relating households to Time-of-use Tariff and stimulus

### General approach and challenges

![alt tag](https://github.com/felgueres/kWintessence/blob/master/figures_and_presentation/01_overview.png)

1) _Feature construction and clustering_

Essentially, the consumption of each user at any given time period can be thought of as an independent feature.
On a 15-min granularity, spanning 1.5 years and for roughly 4,000 households, implies a high dimensionality matrix.
Hence, the first challenge is reducing dimensionality and apply an unsupervised machine learning technique to cluster users by similar pattern consumption.

The value of this step also lies in defining a working hypothesis about the clusters:

> _Households within clusters behave similarly under same circumstances, therefore, the baseline for time-of-use tariffs can be estimated by the actual loads of the corresponding control group_.

The dataset used includes a 6-month period where all users where exposed to same conditions and therefore is an unbiased timespan to perform the clustering.
On this line and thinking about the actual application of demand response (DR) applications, the benchmark doesn't need to be an extended period of time, it could be done within non-event DR days.

The following image shows plots for every cluster where each curve represents a user.
It also shows how the clusters capture users' variability and magnitude of consumption.

Note the number of clusters was determined heuristically; stakeholder's input would be ideal.

![alt tag] (https://github.com/felgueres/kWintessence/blob/master/figures_and_presentation/02_clusters.png)

3) _Comparison baseline_

The baseline estimate is calculated as a function of the control (clustered) mean, but note that other models (ex. regression-based using temperature) may increase the accuracy of the estimation.
Such variations were not explored since this dataset is very limited in demographic information due to privacy concerns.

The following figure summarizes the mean daily user profile along with the relative price change between both groups.

![alt tag] (https://github.com/felgueres/kWintessence/blob/master/figures_and_presentation/03_experiment.png)

4) _Quantify response_

At this point, through visual inspection its possible to see whether a cluster is responsive or not. Furthermore, assuming the underlying distributions are Gaussian, a hypothesis is formulated and tested with a typical type I error of 5% .

 > _H0_: (Time-of-use tariffs cluster)mean >= Baseload  
 > or: Increasing price does not induce a significant decrease in consumption.   

> _H1_: (Time-of-use tariffs cluster)mean < Baseload

Given the density within clusters vary, it is also helpful to compute the statistical power of the test.

![alt tag] (https://github.com/felgueres/kWintessence/blob/master/figures_and_presentation/04_test_control.png)

5) _Insights_

In the following figure, clusters with a dashed square are presumably responsive.

![alt tag] (https://github.com/felgueres/kWintessence/blob/master/figures_and_presentation/05_evaluation.png)

Consideration:
Note the sample size is reduced as the number of cluster increases.
For clusters 3 and 5, although the hypothesis test proves significant, ideally we would want to increase the sample size to reduce the probability of a Type II error (increasing the statistical power).

6) _Final thoughts_

![alt tag] (https://github.com/felgueres/kWintessence/blob/master/figures_and_presentation/06_futurework.png)

### Code related

Given the 2-week time constraint, this project was conceived as a baseline workflow where additional features were to be implemented as time allowed.
For this reason, the code architecture was designed in a object-oriented way that would make it easier to implement future complexity and scalability.

There are four main code-related files:

1) 'src/Pipeline' : PipeLine class from which all the project runs through. Note that similar attributes and methods to the sklearn library were implemented; see _init_, _transform_ and _fit_ methods for documentation.

2) 'src/import_and_transform': Functions to import data and transform to usable format.

3) 'src/plots': Plotting functions.

4) 'src/metrics': Quantify response functions.

#### _This project was presented at the Galvanize Immersive Data Science Showcase event in San Francisco on October 20th, 2016._
