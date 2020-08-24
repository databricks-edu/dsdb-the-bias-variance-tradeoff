# Databricks notebook source

# MAGIC %md-sandbox

# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
  <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
</div>

# COMMAND ----------

# MAGIC %md
# MAGIC # The Bias-Variance Tradeoff

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# MAGIC %run ./includes/configuration

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Data and Scipy Libraries

# COMMAND ----------

# MAGIC %run ./includes/preprocessing

# COMMAND ----------

# MAGIC %md
# MAGIC ## The Bias-Variance Tradeoff
# MAGIC 
# MAGIC In this notebook, we are interested in the uncertainty associated with
# MAGIC a particular classification model.
# MAGIC 
# MAGIC When measuring the uncertainty of a model, we are frequently
# MAGIC interested in:
# MAGIC 
# MAGIC - the **bias** of that model
# MAGIC    - how well it performs classification
# MAGIC - the **variance** of that model
# MAGIC    — how much the model will differ if fit using different training data
# MAGIC 
# MAGIC Minimizing bias will frequently increase the variance and vice
# MAGIC versa. An optimal model will simultaneously minimize both bias and variance.
# MAGIC This simultaneous minimization the Bias-Variance Tradeoff.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model Selection
# MAGIC 
# MAGIC In this notebook, we will examine many different models for predicting
# MAGIC our target prepared using our feature data. Each of these models
# MAGIC will be prepared using a different subset of the four features:
# MAGIC 
# MAGIC - `mean_resting_heartrate`
# MAGIC - `mean_active_heartrate`
# MAGIC - `mean_BMI`
# MAGIC - `mean_VO2_max`
# MAGIC 
# MAGIC We will use the estimated bias and variance of each of these models
# MAGIC to assess which model or models is likely to be the optimal model.
# MAGIC 
# MAGIC We will also consider the complexity of each model relative to the
# MAGIC estimated bias and variance.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Estimating Bias and Variance with the Bootstrap
# MAGIC 
# MAGIC The **bootstrap** is a method for estimating uncertainty, in this
# MAGIC case uncertainty associated with bias and variance.
# MAGIC 
# MAGIC The method involves generating a series of subsample sets by
# MAGIC sampling with replacement from our dataset.
# MAGIC 
# MAGIC We will then fit a particular model under examination against
# MAGIC each of the bootstrap subsample sets.
# MAGIC 
# MAGIC The **accuracy mean** across the models fit to each
# MAGIC subsample set will be used to estimate **bias**.
# MAGIC 
# MAGIC The **accuracy standard deviation** across the models fit to each
# MAGIC subsample set will be used to estimate **variance**.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Bootstrap Generator
# MAGIC 
# MAGIC In the next cell, we define a function to generate bootstrap samples.
# MAGIC 
# MAGIC 💪🏼 Note that we make sure to sample evenly across each of the
# MAGIC three lifestyles.

# COMMAND ----------

lifestyles = health_tracker_sample_agg_pd_df.lifestyle.unique()
np.random.seed(10)

def generate_bootstrap_sample():
  df = health_tracker_sample_agg_pd_df
  sample_df_list = []
  for lifestyle in lifestyles:
    sample_df_list.append(
      df[df.lifestyle == lifestyle].sample(5)
    )
  return pd.concat(sample_df_list)

# COMMAND ----------

generate_bootstrap_sample()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Generate Subsample Sets

# COMMAND ----------

subsample_sets = []

for _ in range(10):
  subsample_sets.append(generate_bootstrap_sample())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Generate Subsample Sets with List Comprehension

# COMMAND ----------

# ANSWER
subsample_sets = [generate_bootstrap_sample() for _ in range(10)]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Display the Number of Samples in Each Subsample Set
# MAGIC 
# MAGIC Use the `len` built-in function to display the size of each subsample set.

# COMMAND ----------

# ANSWER
[len(sample_set) for sample_set in subsample_sets]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Display the Second Subsample Set

# COMMAND ----------

# ANSWER
subsample_sets[1]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Numerically Encode the Target
# MAGIC 
# MAGIC 1. Fit the `LabelEncoder` on the original target (`health_tracker_sample_agg_pd_df["lifestyle"]`)
# MAGIC 2. Transform each dataframe subset using the for-loop

# COMMAND ----------

# ANSWER
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(health_tracker_sample_agg_pd_df.lifestyle)

for sample_set in subsample_sets:
  sample_set["lifestyle_encoded"] = le.transform(sample_set.lifestyle)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Design Experiment

# COMMAND ----------

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1: Build Subsets of Features.

# COMMAND ----------

feature_subset = ["mean_active_heartrate"]
experimental_data_subsets = [
  sample_set[feature_subset]
  for sample_set in subsample_sets
]

targets = [
  sample_set["lifestyle_encoded"]
  for sample_set in subsample_sets
]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fit the `GridSearchCV` on each subset using Cross-Validation

# COMMAND ----------

# ANSWER
experimental_scores = []
for features, target in zip(experimental_data_subsets, targets):
  gs = GridSearchCV(DecisionTreeClassifier(), param_grid={}, cv=5)
  gs.fit(features, target)
  score = gs.cv_results_["mean_test_score"][0]
  experimental_scores.append(score)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Display Cross-Validation Results

# COMMAND ----------

pd.DataFrame(gs.cv_results_).T

# COMMAND ----------

# MAGIC %md
# MAGIC ### Display Experiment Results

# COMMAND ----------

print(feature_subset)
print("Mean Score: ", np.mean(experimental_scores))
print("Standard Deviation Score: ", np.std(experimental_scores))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define MLflow Experiment Runner

# COMMAND ----------

import mlflow
from sklearn.linear_model import LogisticRegression

def experiment_runner(feature_subset):
  """Helper function to run MLflow experiment on a feature subset."""
  with mlflow.start_run() as run:
    "Build Subsets of Features."
    experimental_data_subsets = [
        sample_set[feature_subset]
        for sample_set in subsample_sets
    ]

    targets = [
      sample_set["lifestyle_encoded"]
      for sample_set in subsample_sets
    ]

    "Fit on each subset using Cross-Validation."
    experimental_scores = []
    for features, target in zip(experimental_data_subsets, targets):
      gs = GridSearchCV(LogisticRegression(penalty='none', max_iter=10000), param_grid={}, cv=5)
      gs.fit(features, target)
      score = gs.cv_results_["mean_test_score"][0]
      experimental_scores.append(score)

    "Record experiment results."
    mlflow.log_param("subset", feature_subset)
    mlflow.log_metric("mean score", np.mean(experimental_scores))
    mlflow.log_metric("std score", np.std(experimental_scores))

# COMMAND ----------

experiment_runner(["mean_active_heartrate"])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Generate Feature Subsets
# MAGIC 
# MAGIC Generates the superset of all combinations of features (minus the empty set).

# COMMAND ----------

from itertools import combinations

feature_columns = health_tracker_sample_agg_pd_df.select_dtypes(exclude=["object"]).columns
feature_subsets = []
for i in range(1, 5):
  feature_subsets += [list(feat) for feat in combinations(feature_columns, i)]

feature_subsets

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run Experiments Using Logistic Regression On Each Feature Subset

# COMMAND ----------

for feature_subset in feature_subsets:
  experiment_runner(feature_subset)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Access MLflow Runs Associated with This Notebook

# COMMAND ----------

results = mlflow.search_runs()
type(results)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prepare Results Data

# COMMAND ----------

results = results[['metrics.mean score', 'metrics.std score', 'params.subset']]
results = results[~results["params.subset"].isnull()]
results.drop_duplicates(inplace=True)
results

# COMMAND ----------

# MAGIC %md
# MAGIC #### Augment Results with `n_terms` Column

# COMMAND ----------

results["n_terms"] = results["params.subset"].apply(lambda x: x.count(",") + 1)
results["metrics.mean score"] = 1 - results["metrics.mean score"]
results

# COMMAND ----------

# MAGIC %md
# MAGIC ### Plot the Model Results Versus Bias and Variance

# COMMAND ----------

plt.figure(figsize=(20,10))

for _, (bias, variance, description, n_terms) in results.iterrows():
   plt.scatter(bias, variance, s=100*n_terms, label=description)
plt.xlim(0.1, 0.6)
plt.ylim(0, 0.25)
plt.legend()


# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2020 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>