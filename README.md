# Boston House Price Prediction

| Name | Date |
|:-------|:---------------|
| B M Shahrier Majumder | 16-March-2019 |

-----

## Resources

- Python script for analysis
- Results figure/saved file
- Dockerfile for your experiment
- runtime-instructions in a file named RUNME.md

-----

## Research Question

**What is the best Model for predicting Boston house prices?**

## Abstract

4 sentence longer explanation about your research question. Include:

- opportunity (what data do we have)
- challenge (what is the "problem" we could solve with this dataset)
- action (how will we try to solve this problem/answer this question)
- resolution (what did we end up producing)

## Introduction

Brief (no more than 1-2 paragraph) description about the dataset. Can copy from elsewhere, but cite the source (i.e. at least link, and explicitly say if it's copied from elsewhere).

## Methods

Following steps have been performed to get the answer of my research question.

**`Step 1:`** Choose the tool and technology for doing the research.

**`Step 2:`** Get the data

**`Step 3:`** Process data for analysis

**`Step 4:`** Perform exploratory data analysis

**`Step 5:`** Prepare Training & Test dataset

**`Step 6:`** Create Models for predicting price and perform testing

**`Step 7:`** Measure performance of the Models and choose best Model

Brief (no more than 1-2 paragraph) description about how you decided to approach solving it. Include:

- pseudocode for this method (either created by you or cited from somewhere else)
- why you chose this method

## Results

### Exploratory Data Analysis (EDA)
I have used some visualizations to understand the relationship of the target variable `MEDV` with other features.
First plot is the distribution of the target variable `MEDV`. I have used the `distplot` function from the `seaborn` library.
![alt text](https://github.com/bmshahrier/boston-house-price-prediction/blob/master/plots/histMEDV.png "Histogram")

We see that the values of `MEDV` are distributed normally with few outliers.

Next, I have created a correlation matrix that measures the linear relationships between the variables. The correlation matrix is formed by using the `corr` function from the `pandas` dataframe library. I have also used the `heatmap` function from the `seaborn` library to plot the correlation matrix.
![alt text](https://github.com/bmshahrier/boston-house-price-prediction/blob/master/plots/PearsonHeatMap.png "Pearson Heat Map")
The correlation coefficient ranges from -1 to 1. If the value is close to 1, it means that there is a strong positive correlation between the two variables. When it is close to -1, the variables have a strong negative correlation.

By looking at the correlation matrix we can see that `RM` has a strong positive correlation with `MEDV` (0.7) and `LSTAT` has a high negative correlation with `MEDV` (-0.74).

Finally I have created scatter plots to see correlate among `MEDV`, `RM` and `LSTAT`. I have used the `scatterplot` function from the `seaborn` library.

![alt text](https://github.com/bmshahrier/boston-house-price-prediction/blob/master/plots/scatter-RM-MEDV-LSTAT-MEDV.png "Scatter Plot")

From the scatter plots we can see that the `MEDV` or prices increase as the value of `RM` increases linearly. There are few outliers and the data seems to be capped at 50. And the prices tend to decrease with an increase in `LSTAT`.

**Based on the above observations I have choosed `RM` and `LSTAT` as independent features to predict dependent feature `MEDV` or Price.** 

### Model Creation and Testing

![alt text](https://github.com/bmshahrier/boston-house-price-prediction/blob/master/plots/Regression.png "Regression Plots")

### Model Performance


## Discussion
Brief (no more than 1-2 paragraph) description about what you did. Include:

- interpretation of whether your method "solved" the problem
- suggested next step that could make it better.

## References
All of the links

-------
