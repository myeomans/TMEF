
'''
WEEK 1

Ensure that pingouin has been installed using pip install pingouin. You will need this later.
Note: you may be prompted to restart the runtime session after the instalation has finished.
Choose restart

%Use: pip install pingouin in terminal


Download week1_answers.csv, week1_mike.csv and week1_burint.csv from Insendi onto your local machine.
'''

import pandas as pd
import numpy as np
import os

# check location if using local machine
# print(os.getcwd())


# Read CSV into dataframe
week1_mike = pd.read_csv('week1_mike.csv')
week1_burint = pd.read_csv('week1_burint.csv')

# Let's check the data has been loaded properly
# Check column names
print(list(week1_mike))
print(list(week1_burint))

# check for any blank cells
print(np.where(pd.isnull(week1_mike)))  # if any empty, arrays should be blank
print(np.where(pd.isnull(week1_burint)))

# Left join 2 dataframes on a column name.
# Note, there are different methods for joining tables (e.g. pd.concat())
# Generally, 'merge' is the safer option
week1 = pd.merge(week1_mike, week1_burint, how='left', on=['review_id', 'text'])

# Note to avoid duplicate 'text' columns, include the column name in the join statement
print(list(week1))

# Fast way of checking the first 5 rows
print(week1.head())

'''
Let's check the consistency of star ratings from Mike and Burint

We need to calculate Pearson's correlation

Check that you have from scipy.stats import pearsonr

The Pearson’s correlation coefficient is calculated as the covariance of the two variables divided by the product of the standard deviation of each data sample.
It is the normalization of the covariance between the two variables to give an interpretable score.
Pearson's correlation coefficient = covariance(X, Y) / (stdv(X) * stdv(Y))
'''

from scipy.stats import pearsonr

corr, _ = pearsonr(week1['stars_mike'], week1['stars_burint'])
print(round(corr, 3))

# Alternatively you could subset the dataframe and perform a correlation on the whole dataframe
# This will return a correlatio matrix
# The default correlation matrix method is pearson's
print(week1[['stars_mike', 'stars_burint']].corr())

# STOP - now try the same thing for price and for gender
######################################################################

# For gender, use one hot encoding to convert categorical variables into binary
# pd.get_dummies will create binary variables for all categories (i.e. male and female).
# We only need one set of variables (e.g. male = 1 and female = 0).
# Use drop_first = True to get k-1 dummies out of k categorical levels by removing the first level
week1 = pd.get_dummies(week1, columns=['gender_mike', 'gender_burint'], drop_first=True)

print(list(week1))

# Now we can do the correlation. Note that gender_mike is now called gender_mike_male
print(week1[['gender_mike_male', 'gender_burint_male']].corr())

# since we know that 1 = male, we can change the column titles back to gender_mike and gender_burint
# {} denote python dictionary structures which stores key value pairs
week1.rename(columns={
    'gender_mike_male': 'gender_mike',
    'gender_burint_male': 'gender_burint'
}, inplace=True)

# Correlations give us "pairwise" summary statistics. What if we want to summarize more than two columns? We calculate "Cronbach's alpha"
# Ensure that pingouin has been installed using pip install pingouin - you should have done this at the beginning.


import pingouin as pg
print(pg.cronbach_alpha(week1[['stars_mike', 'stars_burint']]))

# the output should look somekthing like this: (-0.2857142857142856, array([-1.634,  0.372])) where
# the alpha is the first value in the list and the array gives the confidence intervals
# This alpha is our "consistency" - how correlated are the annotators with each other?
# Note: this is not a measure of "validity" - how correlated are the annotators with the truth?
# What if all the annotators make the same mistake? high consistency, low validity

############################################################################

# Before we join this to the correct answers, we need one more new concept - pivoting to long format

# Converting to long format
# check dimensions of current dataframe
print(week1.shape)

week1_long = week1.melt(id_vars=['review_id', 'text'], var_name='question', value_name='guess')

print(list(week1_long))

# split questions into 2 seperate columns for metric and annotator

week1_long[['metric', 'annotator']] = week1_long['question'].str.split('_', expand=True)
print(week1_long.head())
print(list(week1_long))


# change the order so 'guess' goes last and 'question' is dropped

keep = ['review_id', 'text', 'metric', 'annotator', 'guess']
week1_long = week1_long[keep]

print(week1_long.head())
print(list(week1_long))

# check dimensions of the dataframe again - we should have only 5 columns
print(week1_long.shape)

############################################################################

# Let's bring in our correct answers
week1_answers = pd.read_csv('week1_answers.csv')

print(list(week1_answers))

# First, we need to convert gender to binary variables again
week1_answers = pd.get_dummies(week1_answers, columns=['genderTRUE'], drop_first=True)

print(list(week1_answers))
# Note that after the get_dummies, the new column created (genderTRUE_) is now at the end

# Rename genderTRUE_male to genderTRUE
week1_answers.rename(columns={
    'genderTRUE_male': 'genderTRUE'
}, inplace=True)

# Let's do the same pivot as before but this time, rename the columns first so the metric names will match our previous data

# remember that the order of stars and gender have swapped after using get_dummies
week1_answers.columns = ['review_id', 'text', 'price', 'stars', 'gender']
week1_answers_long = week1_answers.melt(id_vars=['review_id', 'text'], var_name='metric', value_name='answers')
print(list(week1_answers_long))

# As before, let's perform a left_join
week1_all = pd.merge(week1_long, week1_answers_long, how='left', on=['review_id', 'metric'])
print(list(week1_all))


# How many of the actual answers did we guess correctly?
# Calculating accuracy here is easy: does the guess equal the answer?
# Accuracy for classification means how many values in the 'predicted' column are equal to those in the 'actual' column. In this case we compare matches between 'guess' and 'answers'
# In machine learning, calculating accuracy scores is pretty straightforward using sklearn

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(week1_all['guess'], week1_all['answers'])

print(accuracy)

# But we want to calculate accuracy separately for each metric/annotator

metric_names = pd.unique(week1_all['metric'])
annotator_names = pd.unique(week1_all['annotator'])

# One common aproach in Python would be to subset the dataframe
# based on metric/annotator name and then
# calculate the scores for each subset of the dataframe
for metric in metric_names:

    week1_sub = week1_all[week1_all['metric'] == metric]
    accuracy = accuracy_score(week1_sub['guess'], week1_sub['answers'])
    print(accuracy)

# However, because we want to repeat this for annotators as well, it would be more efficient to write a general function and then apply that function to the data and field (columns) we want.
# Run the below function to read the function in, then call the function afterwards with the desired parameters.


def acc_subdomain(df, field, classes):
    '''
    Takes in a field (metric_names or annotator_names
    outputs accuracy scores for all classes in that field in a list
    '''

    results = []

    for c in classes:
        df_sub = df[df[field] == c]
        accuracy = accuracy_score(df_sub['guess'], df_sub['answers'])
        results.append(round(accuracy, 3))

    results = pd.DataFrame([classes, results], index=['class', 'accuracy']).T

    return results


# Use the function to get accuracy scores for all classes of the metrics field:
acc_metrics = acc_subdomain(week1_all, 'metric', metric_names)
print(acc_metrics)

# Repeating for the annotators field
acc_annotator = acc_subdomain(week1_all, 'annotator', annotator_names)
print(acc_annotator)


# Also, maybe we want standard errors? our formula for binary data is p*(1-p)/sqrt(n)
# we modify our function to...

def acc_subdomain(df, field, classes):
    '''
    Takes in a field (metric_names or annotator_names)
    outputs:
        accuracy scores for all classes in that field in a list
        standardised scores
    '''

    acc_all = []
    se_all = []

    for c in classes:
        df_sub = df[df[field] == c]
        accuracy = accuracy_score(df_sub['guess'], df_sub['answers'])
        se = np.sqrt(accuracy * (1 - accuracy) / df_sub.shape[0])
        acc_all.append(round(accuracy, 3))
        se_all.append(round(se, 3))

    results = pd.DataFrame([classes, acc_all, se_all], index=['class', 'accuracy', 'se']).T

    return results


acc_metrics = acc_subdomain(week1_all, 'metric', metric_names)
print(acc_metrics)

# Repeating for the annotators field
acc_annotator = acc_subdomain(week1_all, 'annotator', annotator_names)
print(acc_annotator)


import matplotlib.pyplot as plt

# Choose your variables
y = acc_annotator['accuracy']
err = acc_annotator['se']
x = acc_annotator['class']

fig = plt.figure()

# you can add elements to the chart one line at a time

# each point is plotted seperately as matplotlib will otherwise treat them as the same series
# elinewidth = thickness of line, capsize = the top and bottom horizontal lines on the error bars

# NOTE using a notebook (Colab or Jupyter) is a little funny and may throw an assertion error
# if you want to overlay multiple series (e.g. with different colours) on top of each other
# On your local machine, this is not an issue at all.

# If you want different colours for each marker, use these commands instead
plt.errorbar(x[0], y[0], yerr=err[0], fmt="o", color="r", elinewidth=.6, markersize=8, capsize=10)
plt.errorbar(x[1], y[1], yerr=err[1], fmt="o", color="b", elinewidth=.6, markersize=8, capsize=10)

# To get the code to run on Colab, this use this line
# plt.errorbar(x, y, yerr=err, fmt="o", color="r", elinewidth=.6, markersize=8, capsize=10)

# add horizontal line
plt.axhline(y=0.5, color='lightgrey', linestyle='-')

# add axis labels
plt.xlabel('Annotator Name', fontsize=18)
plt.ylabel('Accuracy', fontsize=18)

# Makes the margins a bit wider (useful when there's only 2 points)
plt.margins(0.5, tight=True)

# set the height of the yaxis to be proportional to the data
plt.ylim(top=((max(y) + max(err))) * 1.02,
         bottom=((min(y) - min(err))) * 0.98)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.show()
