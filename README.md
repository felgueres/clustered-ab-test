# kWintessence

This 2-week data science project aims to identify price-responsive electrical loads.

### Overview

The integration of renewable energy generation and significant changes in demand (ie. electric cars, storage, CHP) challenge the grid's overall balance.

At a household-level, smart-metering technology is a means to collect high resolution temporal data; enabling service providers to evaluate demand management strategies.

A commonly discussed strategy is controlling demand through users consumption elasticity.
This project explores an unsupervised ML technique along with a clinical-test approach to objectively identify responsive users.

### Data Source

The dataset derives from the CER Smart Metering experiment in Ireland, where users were allocated either to a flat (control) or variable (treatment) electricity tariff.

The CER's goal was to address the overall household response to variable tariffs while this project attempts to identify the subgroups of users that drive such response.

 * Irish Social Science Data Archive: Smart Meter
   * Includes 4,000 anonymized household data
   * household id, timestamp, consumption (kWh)
   * 15-min time-resolution  
   * 6 csv files: 3+ GB
   * http://www.ucd.ie/issda/data/commissionforenergyregulationcer/

*  Household allocation
   * csv file relating households to Time-of-use Tariff and stimulus

### General approach and challenges

![alt tag](https://github.com/felgueres/kWintessence/blob/master/figures_and_presentation/01_overview.png)

1) _Feature construction and clustering_

The consumption of each user at any given time period can be thought of as an independent feature.
At 15-min granularity, spanning 1.5 years and roughly 4,000 households implicates a high dimensionality matrix.

The first challenge is to reduce dimensionality and to cluster users by similar profile consumption.

The value of this step also lies in defining a working assumption about the clusters:

> _Households within clusters under same circumstances behave similarly, thus, the baseline comparison for variable tariffs can be estimated by the actual loads of the corresponding control group_.

The dataset includes a 6-month period where all users where exposed to same conditions and therefore is an unbiased timespan to perform the clustering. Furthermore, thinking about the actual application of demand response (DR) applications, the benchmark doesn't need to be an extended period of time, it could be done within non-event DR days.

The following image shows plots for every cluster where each curve represents a user.
It also shows how the clusters capture users' variability and magnitude of consumption.

Note the number of clusters was determined heuristically; service provider's input would be ideal here.

![alt tag] (https://github.com/felgueres/kWintessence/blob/master/figures_and_presentation/02_clusters.png)

3) _Comparative baseline_

The comparative baseline is calculated as a function of the control (clustered) mean, but note that other models (ex. regression-based load-temperature model) may increase the estimation accuracy.
Such variations were not explored since this dataset is limited in demographic information due to privacy concerns.

The following figure summarizes the mean daily user profile along with the relative price change between both groups.

![alt tag] (https://github.com/felgueres/kWintessence/blob/master/figures_and_presentation/03_experiment.png)

4) _Quantify response_

At this point, through visual inspection its possible to see whether a cluster is responsive or not.
For objectiveness, assuming the underlying distributions are Gaussian, a hypothesis can be formulated and tested with a typical type I error of 5% .

 > _H0_: (Time-of-use tariffs cluster)mean >= Baseload  
 > or: Increasing price does not induce a significant decrease in consumption.   

> _H1_: (Time-of-use tariffs cluster)mean < Baseload

Given the density within clusters vary, it is also helpful to compute the statistical power of the test.

![alt tag] (https://github.com/felgueres/kWintessence/blob/master/figures_and_presentation/04_test_control.png)

5) _Insights_

In the following figure, clusters with a dashed-square are identified as responsive.

![alt tag] (https://github.com/felgueres/kWintessence/blob/master/figures_and_presentation/05_evaluation.png)

Note the sample size is reduced as the number of cluster increases.
For clusters 3 and 5, although the hypothesis test proves significant, ideally we would want to increase the sample size to reduce the probability of a Type II error (increasing statistical power).

6) _Final thoughts_

![alt tag] (https://github.com/felgueres/kWintessence/blob/master/figures_and_presentation/06_futurework.png)

### Code related

Given the 2-week time constraint, this project was conceived as a baseline workflow where additional features were to be implemented as time allowed. The code is object-oriented to make it easier to implement future complexity and scalability.

There are four main code-related files:

1) 'src/Pipeline' : PipeLine class from which all the project runs through. Note that similar attributes and methods to the sklearn library were implemented; see _init_, _transform_ and _fit_ methods for documentation.

2) 'src/import_and_transform': Functions to import data and transform to usable format.

3) 'src/plots': Plotting functions.

4) 'src/metrics': Quantify response functions.

#### _This project was presented at the Galvanize Immersive Data Science Showcase event in San Francisco on October 20th, 2016._
