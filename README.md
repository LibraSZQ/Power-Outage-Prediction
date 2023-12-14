<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<script> 
MathJax = {
  tex: {
    inlineMath: [['$', '$']],
    processEscapes: true
  }
};
</script>

# A Prediction on the Severity of Power Outages

## Problem Identification

Understanding and predicting the severity of power outages is vital for ensuring public safety, minimizing economic losses, protecting infrastructure, preparing communities, and allocating emergency resources efficiently. Accurate predictions enable effective response and recovery efforts, helping to mitigate the overall impact of such incidents.

This project mainly centered around a question "how to predict the severity of a major power outage". The project is based on our previous project, our exploratory data analysis on this dataset can be found [here](https://lr580.github.io/power_outages_stats/), where we've made comprehensive investigation of the power outage data. The data we use can be downloaded [here](https://engineering.purdue.edu/LASCI/research-data/outages).

This dataset includes 56 column variables and 1534 row variables in total. In researching the question stated above, not all 56 columns variables will be included in the analysis. Thus, relevant information will be included for the analysis such as: 

| Column               | Description                                                 |
| -------------------- | ----------------------------------------------------------- |
| `OUTAGE.DURATION`    | The minutes an outage lasts                                 |
| `CLIMATE.CATEGORY`   | The climate type(warm, normal, cold) where an outage occurs |
| `CAUSE.CATEGORY`     | The reason for an outage.                                   |
| `CUSTOMERS.AFFECTED` | The number of customers affected by the outage.             |
| `DEMAND.LOSS.MW`     | The power loss(megawatt) in the outage.                     |
| `NERC.REGION`        | The NERC region code where an outage occurs.                |

In our predealing process, we removing all the rows if the `CUSTOMERS.AFFECTED` or `OUTAGE.DURATION` is missing, since we consider with such missing, it's hard to tell the severity. And then, we fill in missing values of another row `DEMAND.LOSS.MW`. 

The prediction problem we'd focus on is predicting the severity of a major power outage.

Our project focused on predicting the severity of major power outages, with a specific focus on one key metric: the number of affected customers. This single factor serves as our primary measure for predicting outage severity.

We chose this indication due to the following reasons:

1. It is a direct and tangible indicator of the outage's impact, making it a more reliable and accessible data point compared to others.
2. Other potential factors, such as outage duration or demand loss, may not always accurately represent the severity of an outage and could introduce unnecessary complexity into our model.
3. The `DEMAND.LOSS.MW` data often has too many missing values that makes it impractical for consistent analysis.
4. Incorporating multiple factors could lead to increased complexity and the risk of collinearity, which might compromise the accuracy of our predictive model.

To quantify outage severity, we apply the formula $severity = \log_2 (number\_of\_customers+1)$. This logarithmic transformation is chosen based on our analysis of the data, which reveals significant variance in the scale of customer impact across different outages. Using the raw `CUSTOMERS.AFFECTED` data would pose challenges in both measurement and model training due to the wide range of values. Large figures can lead to disproportionate effects on the model. The logarithmic scale helps in normalizing these values, ensuring a more balanced and accurate representation of severity.

Our project utilizes a regression model to predict the severity of power outages. The model's response variable is the logarithmic transformation of `CUSTOMERS.AFFECTED`, which serves as a proxy for outage severity.

We chose $R^2$ metric to evaluate our model for the following reasons:

1. This is a regression problem, classification metrics like precision or recall are not applicable.
2. RMSE and $R^2$ are commonly used among classic regression metrics. However, for our purposes, we find it sufficient to use just one of these metrics. Our choice is based on the following considerations:

1) RMSE is preferred when the absolute magnitude of errors is crucial. It directly reflects the average discrepancy between predicted and actual values, and sensitive to larger errors.
2) $R^2$ is more suitable when assessing the explanatory power of a model. It measures how well the model accounts for the variability of the target variable, making it useful for standardized comparisons across different datasets.
3) Given our focus on the model's explanatory power and the need for a standardized performance measure, we chose the $R^2$ metric."

## Baseline Model

In our analysis, we evaluated numerous features, which are `U.S._STATE`, `POSTAL.CODE`, `CLIMATE.REGION`, `ANOMALY.LEVEL`, `OUTAGE.START.DATE`, `OUTAGE.START.TIME`, `OUTAGE.RESTORATION.DATE`, `OUTAGE.RESTORATION.TIME`, `TOTAL.PRICE`, `TOTAL.SALES`, `TOTAL.CUSTOMERS`, `POPULATION`, `POPDEN_URBAN`, both individually and in combination. However, these features showed minimal impact on our model's performance and were confirmed irrelevant. An exception was the feature `CAUSE.CATEGORY`, which was identified as a significant factor in our baseline model.

It is worth notice that we tried adding other caused-related features such as `CAUSE.CATEGORY.DETAIL` and `HURRICANE.NAMES`. However, they barely made no contribution to improve our model. Therefore, our baseline model primarily incorporates the CAUSE.CATEGORY feature. Further exploration of other features and additional refinements will be considered for the final model.

For data splitting, we follow the standard practice of using 75% for training and 25% for validation.

Our exploration of various classic models revealed that `LinearRegression`, `KNeighborsRegressor` and `SVR` were not effective for our dataset. In contrast, `DecisionTreeRegressor` and `RandomForestRegressor` showed promising results. 

We adopt the `DecisionTreeRegressor` as our baseline model and will compare it with the `RandomForestRegressor` to determine the most effective approach for our final model.

To make our reported result stable and reproducible, we set the random seed manually.

First, we created a helper class to transform cause category strings into ordinal values. This class will be utilized in the `ColumnTransformer` in subsequent steps.

We observed all the different values of cause category, and find that there's only 7 different values. So we use the mapping below to convert the feature `CAUSE.CATEGORY` into ordinal encoding.

```python
{'severe weather': 0, 'intentional attack': 1, 'public appeal': 2, 'system operability disruption': 3, 'islanding': 4, 'equipment failure': 5, 'fuel supply emergency': 6}
```

We use the baseline model to predict values in both train set and validation set, and calculate the metric selected above.

Our model demonstrates robust performance on both the training and validation datasets, with $R^2$ ranging approximately between $0.76\sim0.81$. This consistency is further validated by examining specific prediction examples, where the model's estimates closely align with the actual values. These results suggest that $7$ distinct types of 'cause category' correlate well with different scales in the number of affected customers, providing a rough but effective indicator of outage severity.

```
train evaluate: 0.8131982175490576
test evaluate: 0.7675446684209979
```

Samples of train prediction:

| Prediction Severity | Real Severity |
| ------------------- | ------------- |
| 11.557              | 12.41         |
| 0.736               | 0.0           |
| 10.084              | 11.444        |
| 11.557              | 11.744        |
| 10.553              | 11.142        |

Samples of test prediction:

| Prediction Severity | Real Severity |
| ------------------- | ------------- |
| 11.557              | 13.108        |
| 0.736               | 0.0           |
| 0.736               | 0.0           |
| 11.557              | 12.553        |
| 11.557              | 12.832        |

## Final Model

### Hyperparameter Searching

One potential improvement is in hyperparameter selection. 

We first presented a hyperparameter searching helper function using `GridSearchCV`. The searching range was `[1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,22,24,26,28,30,50]`。

Our analysis indicated that the `max_depth` parameter significantly impacts the performance of the `DecisionTreeRegression`. Therefore, we focused our hyperparameter search on optimizing the tree depth.

Our findings show that the optimal tree depth for the model is 6. As expected, no significant improvement was observed. This is likely due to the simplicity of the `CAUSE.CATEGORY` feature, which only encompasses 7 distinct values. Hence, variations in tree depth have limited impact on enhancing the model's performance."

```
Before: 
train evaluate: 0.8131982175490576
test evaluate: 0.7675446684209979
best param: {'model__max_depth': 6}
After: 
train evaluate: 0.8121172035410144
test evaluate: 0.7748218273592309
```

### Different Encodings

Given that `CAUSE.CATEGORY` is a categorical feature, a common approach is to experiment with different encoding methods. To explore this, we then applied `OneHotEncoder` to convert it into nominal encoding, replacing the previously used ordinal encoding."

```
Before: 
train evaluate: 0.8131982175490576
test evaluate: 0.7675446684209979
best param: {'model__max_depth': 6}
After: 
train evaluate: 0.8121172035410144
test evaluate: 0.774821827359231
```

However, no improvement was observed with this change.

### Adding New Feature

Our focus now shifts to identifying new features. We choose to incorporate the `NERC.REGION` attribute, hypothesizing that different NERC regions have varying capacities to handle outages, which in turn could influence the severity of these incidents.

```
Before: 
train evaluate: 0.8422277429766968
test evaluate: 0.7823223717827422
best param: {'model__max_depth': 9}
After: 
train evaluate: 0.8403517560962385
test evaluate: 0.8209825624674565
```

We noticed that incorporating the `NERC.REGION` feature resulted in a slight improvement in our model's performance. Therefore, we included it in our analysis.

After experimenting with numerous additional features, we observed no significant improvements in our model (details omitted due to space constraints).

Notably, the `DEMAND.LOSS.MW` feature showed potential in enhancing the model's accuracy. However, as this data might not be available at the time of prediction, although it demonstrates improvement, we've decided not to include it in our final model.

In conclusion, after evaluating nearly 20 features and their combinations, we identified only three as impactful: `CAUSE.CATEGORY`, `NERC.REGION`, and `DEMAND.LOSS.MW`. However, since `DEMAND.LOSS.MW` is not typically known at the time of prediction, we will only use `CAUSE.CATEGORY` and `NERC.REGION` in our final model.

### Different Models

In our final step, we switch to using the `RandomForestRegressor` model and conduct another round of hyperparameter selection. We search max depth in `[1,2,3,4,5,6,7,8,20,50,100]` and number of estimators in `[1,10,25,50,100]`.

```
Before: 
train evaluate: 0.8412494230284535
test evaluate: 0.7786905744712731
best param: {'model__max_depth': 6, 'model__n_estimators': 50}
After: 
train evaluate: 0.8370717045460225
test evaluate: 0.8161925455663533
```

Our comparison reveals that the performances of the `DecisionTreeRegressor` and `RandomForestRegressor` models are nearly identical. Additionally, we tested `LinearRegression`, `KNeighborsRegressor` and `SVR` models, all of which yielded unsatisfactory results (details omitted due to space constraints). Consequently, we have decided to proceed with the `DecisionTreeRegressor` as our model of choice.

The visualization illustrating the performance of our chosen model is presented below.

```
best param: {'model__max_depth': 9}
train evaluate: 0.8403517560962385
test evaluate: 0.8209825624674565
```

<iframe src="assets/perform_pipeline.html" width=800 height=600 frameBorder=0></iframe>

We'd perform it in the whole data, compared with baseline model. The $R^2$ is shown below:

```
baseline model's R2: 0.801458743651225
final model's R2: 0.8353712022533668
```

There's improvement on $R^2$ in the final model, which means that our improvement methods are useful.



## Fairness Analysis

To assess the fairness of our model, specifically whether it performs differently for individuals in various groups, we will conduct a fairness analysis shown below.

For our fairness analysis, we used the $R^2$ metric as the quantitative attribute. The analysis involved comparing the $R^2$ scores across different groups, focusing on the absolute difference between $R^2$ values, calculated as: 

$$
|R^2_{groupX} - R^2_{groupY}|
$$

We simply define: 

1. group X as the outage where `CLIMATE.CATEGORY` is `cold`
2. group Y as the outage where `CLIMATE.CATEGORY` is not `cold`. 

Obviously, it's a binary groups.

To evaluate model fairness, we will conduct a permutation test with the following hypotheses:

Null Hypothesis: The model is fair, meaning its precision for outages in cold and non-cold climates is approximately the same, with any observed differences attributed to random chance.

Alternative Hypothesis: The model is unfair, exhibiting a significant difference in precision between outages in cold climates and those in non-cold climates.

Significance level: 0.05

The p-value in our analysis measures the probability of observing extreme results under the assumption that the null hypothesis is true. If the results are significantly different (indicating an extreme case), this would be reflected in a higher evaluation metric. Therefore, we accumulate the p-value when the simulated value exceeds the observed value, as demonstrated in the code below.

To ensure the reproducibility of our results, we employ a static seed list (`SEED+_`) for the random shuffling in our permutation test.

<iframe src="assets/permutation_test.html" width=800 height=600 frameBorder=0></iframe>

The analysis yields a p-value of 0.278. Given our significance level of 0.05, this p-value, being greater than 0.05, leads us to not reject the null hypothesis. This suggests that our model is likely fair, with similar precision across different groups.

Furthermore, this outcome implies that `CLIMATE.CATEGORY` does not significantly impact the prediction of outage severity, which proves our earlier conclusion that this feature is not effective for predicting severity.
