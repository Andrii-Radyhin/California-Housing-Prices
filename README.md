# California-Housing-Prices

Abstract: This repo includes a pipeline using sklearn Regression Models to predict House Prices. We will use California Housing dataset, available in the sklearn.datasets module. The pipeline involves a systematic approach to data preprocessing, feature engineering, model selection, hyperparameter tuning, and performance evaluation.

## EDA

To obtain a comprehensive understanding of the predictive attributes and the target variable present in the fetch_california_housing dataset, we can utilize the .DESCR method.

Firstly, we can see that we have next Attribute Information:

        - MedInc        median income in block group
        - HouseAge      median house age in block group
        - AveRooms      average number of rooms per househol
        - AveBedrms     average number of bedrooms per household
        - Population    block group population
        - AveOccup      average number of household members
        - Latitude      block group latitude
        - Longitude     block group longitude

The target variable is the median house value for California districts,
expressed in hundreds of thousands of dollars ($100,000).

Let’s have an overview of the entire dataset.

"IMAGE with MedHouseVal"

As written in the description, the dataset contains aggregated data regarding each district in California. Let’s have a close look at the features that can be used by a predictive model. (SYNONYMS)

"IMAGE without MedHouseVal"

In this dataset, we have information regarding the demography (income, population, house occupancy) in the districts, the location of the districts (latitude, longitude), and general information regarding the house in the districts (number of rooms, number of bedrooms, age of the house). Since these statistics are at the granularity of the district, they corresponds to averages or medians. (SYNONYMS)

We can now check more into details the data types and if the dataset contains any missing value.

"IMAGE CHECK IF THERE IS ANY MISSING VALUE"

We can see that:

    - the dataset contains 20,640 samples and 8 features;

    - all features are numerical features encoded as floating number;

    - there is no missing values.

Let’s have a quick look at the distribution of these features by plotting their histograms.

"IMAGE HISTOGRAMS"

(SYNONYMS)We can first focus on features for which their distributions would be more or less expected.

The median income is a distribution with a long tail. It means that the salary of people is more or less normally distributed but there is some people getting a high salary.

Regarding the average house age, the distribution is more or less uniform.

The target distribution has a long tail as well. In addition, we have a threshold-effect for high-valued houses: all houses with a price above 5 are given the value 5.

Focusing on the average rooms, average bedrooms, average occupation, and population, the range of the data is large with unnoticeable bin for the largest values. It means that there are very high and few values (maybe they could be considered as outliers?). We can see this specificity looking at the statistics for these features: (SYNONYMS)

"IMAGE FRAME WITH STATISTICS"

For each of these features, comparing the max and 75% values, we can see a huge difference. It confirms the intuitions that there are a couple of extreme values.

Now we will analyse correleation between features and target variable

"IMAGE CORR"

We can see that:

   - House values are significantly correlated with median income
   - Longitude and Latitude should be analyzed separately (just a correlation with target variable is not very useful)

Longitude and Latitude:

The combination of this feature could help us to decide if there are locations associated with high-valued houses. Indeed, we could make a scatter plot where the x- and y-axis would be the latitude and longitude and the circle size and color would be linked with the house value in the district.

"IMAGE Longitude and Latitude"

Thus, creating a predictive model, we could expect the longitude, latitude, and the median income to be useful features to help at predicting the median house values.

## Methodology

We will try several regressors available in sklearn such as: 

    - Linear
    - Ridge
    - LASSO
    - Decision trees
    - Random forests
    - Gradient boosting

Firstly we will preprocess our data for Linear, Ridge and LASSO. We can do it in such ways as Normalization and Scaling. (DIFFERENCE BETWEEN) Other three do not require any preprocess because tree-based algorithms are not sensitive to the scale of the features because they split the data based on individual features, and the algorithm's results are based on the relative values of the features rather than their absolute values. 

As a result, scaling or normalization of the data is not necessary for these algorithms.

R2 Score, MSE and RMSE our metrics. We will evaluate models by them. Also every model will be fine-tuned (hyperparameters GridSearch) to see best results and weed out luck. Closer look to GridSeach, what parameters were fine-tuned, in experiments.ipynb.

Here i provide results before fine-tuning:

TABLE

Desision Tree:

(Before)
R2 :  0.5992174715251078 
MSE:  0.5318101099611434 
RMSE:  0.7292531178960727

(After)
R2 :  0.6183213705485122 
MSE:  0.506460585172765 
RMSE:  0.7116604423267918

Random Forests:

(Before)
R2 :  0.81276815811356 
MSE:  0.24844343090692636 
RMSE:  0.4984410004272585

(After)
R2 :  0.8177681114209377 
MSE:  0.2418088459904739 
RMSE:  0.49174062877748254

Linear Regression (original): 

(Before)
R2 :  0.5539755298960758 
MSE:  0.5918429712840851 
RMSE:  0.7693133115214407

(After)
R2 :  0.6074054352929974 
MSE:  0.5209452603172996 
RMSE:  0.7217653776105498

LASSO (original):

(Before)
R2 :  0.4805376816931025 
MSE:  0.6892898092905969 
RMSE:  0.8302347916647416

(After)
R2 :  0.510383190381585 
MSE:  0.6496869271044228 
RMSE:  0.8060315918774045

Ridge:

(Before)
R2 :  0.5539714944671493 
MSE:  0.5918483260132679 
RMSE:  0.7693167917140947

(After)
R2 :  0.6074043035287395 
MSE:  0.5209467620885389 
RMSE:  0.7217664179556561

GradientBoostingRegressor:

(Before)
R2:  0.7907863627375509 
MSE:  0.2776117209033578 
RMSE:  0.5268887177605512 

(After)
R2:  0.8436412311966435 
MSE:  0.2074770433410031 
RMSE:  0.4554964800533623 
 
 
Standartization:

Linear:
R2 :  0.6074054352929976 
MSE:  0.5209452603172995 
RMSE:  0.7217653776105497

LASSO:
R2:  0.4392929149966498 
MSE:  0.5587612437422472 
RMSE:  0.7475033402883543

Ridge:
R2 :  0.6048561892912513 
MSE:  0.5243279297207429 
RMSE:  0.7241049162384847


"IMAGE RESULTS"

Gradient boosting regressor best one after fine-tuning. Values do not lie us. We will choose then this model to show predictions and results.

We can visualize the feature importances to get more insight into our model:

Gradient Boosting:

"IMAGE"

To compare our top-2 model Random Forests:

"IMAGE"

We can see that the feature importances of the gradient boosted trees are similar to the feature importances of the random forests, it gives weight to all of the features in this case.


Let's also visualize predictions:
"IMAGE"


We practiced a wide array of machine learning models for regression, what their advantages and disadvantages are, and how to control model complexity for each of them. We saw that for many of the algorithms, setting the right parameters is important for good performance.
