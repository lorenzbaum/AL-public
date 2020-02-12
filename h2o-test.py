import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import time
import json

import h2o
from h2o.automl import H2OAutoML, get_leaderboard

h2o.init()



data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header = None)

# specify columns extracted from wbdc.names
data.columns = ["id","diagnosis","radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",
                "compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean",
                "radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se",
                "concave points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst",
                "perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst",
                "concave points_worst","symmetry_worst","fractal_dimension_worst"] 

# save the data
data.to_csv("data.csv", sep=',', index=False)

# print the shape of the data file
print(data.shape)

# show the top few rows
print(data.head())

# describe the data object
print(data.describe())

# we will also summarize the categorical field diganosis 
print(data.diagnosis.value_counts())


data = h2o.import_file('data.csv')

x = data.columns
y = "diagnosis"
x.remove(y)

train, valid = data.split_frame(ratios = [.8], seed = 4156721)

train[y] = train[y].asfactor()



# Run AutoML for 20 base models (limited to 1 hour max runtime by default)
aml = H2OAutoML(max_models=20, seed=1)
aml.train(x=x, y=y, training_frame=train)


# AutoML Leaderboard
lb = aml.leaderboard

# Optionally edd extra model information to the leaderboard
# lb = get_leaderboard(aml, extra_columns='ALL')

# Print all rows (instead of default 10 rows)
print(lb.head(rows=lb.nrows))


# The leader model is stored here
# print(aml.leader)

perf = aml.leader.model_performance(valid)

print('#######################################')

print(perf)  ## DeepLearning_grid__1_AutoML_20200212_005809_model_1    ## accuracy 0.991304

print('#######################################')