import pandas as pd
from scipy.stats import *

"""
Summary Statistics:
	             Min  Max   Mean    SD   Class Correlation
   sepal length: 4.3  7.9   5.84  0.83    0.7826   
    sepal width: 2.0  4.4   3.05  0.43   -0.4194
   petal length: 1.0  6.9   3.76  1.76    0.9490  (high!)
    petal width: 0.1  2.5   1.20  0.76    0.9565  (high!)
"""

# Defined from dataset
sl_mean = 5.84
sw_mean = 3.05
pl_mean = 3.76
pw_mean = 1.20

sl_sd = 0.83
sw_sd = 0.43
pl_sd = 1.76
pw_sd = 0.76

# Import iris dataset 
df = pd.read_csv("./iris/iris.data", header=None)

# Set columns
iris_columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
df.columns = iris_columns

# Shuffle all rows
shuffled_df = df.sample(frac=1)

# Define sample size
training_samp_size = 80
validation_samp_size = 38
testing_samp_size = 32

# Split data into 3 groups
training_df = shuffled_df.iloc[0:training_samp_size]
validation_df = shuffled_df.iloc[training_samp_size:training_samp_size+validation_samp_size]
testing_df = shuffled_df.iloc[training_samp_size+validation_samp_size:]

"""
Assignment 1.)
"""
# Find mean and standard deviation for each attribute and for each dataset
def get_sample_mean(df: pd.DataFrame):
    sl_mean = round(df['sepal_length'].mean(), 2)
    sw_mean = round(df['sepal_width'].mean(), 2)
    pl_mean = round(df['petal_length'].mean(), 2)
    pw_mean = round(df['petal_width'].mean(), 2)
    return sl_mean, sw_mean, pl_mean, pw_mean

def get_sample_std(df: pd.DataFrame):
    sl_sd = round(df['sepal_length'].std(), 2)
    sw_sd = round(df['sepal_width'].std(), 2)
    pl_sd = round(df['petal_length'].std(), 2)
    pw_sd = round(df['petal_width'].std(), 2)
    return sl_sd, sw_sd, pl_sd, pw_sd

# Find statistics for training dataset
training_sl_mean, training_sw_mean, training_pl_mean, training_pw_mean = get_sample_mean(training_df)
training_sl_sd, training_sw_sd, training_pl_sd, training_pw_sd = get_sample_std(training_df)

# Find statistics for validation dataset
validation_sl_mean, validation_sw_mean, validation_pl_mean, validation_pw_mean = get_sample_mean(validation_df)
validation_sl_sd, validation_sw_sd, validation_pl_sd, validation_pw_sd = get_sample_std(validation_df)

# Find statistics for testing dataset
testing_sl_mean, testing_sw_mean, testing_pl_mean, testing_pw_mean = get_sample_mean(testing_df)
testing_sl_sd, testing_sw_sd, testing_pl_sd, testing_pw_sd = get_sample_std(testing_df)


print(testing_sl_mean)