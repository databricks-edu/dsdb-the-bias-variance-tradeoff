# Databricks notebook source

# MAGIC %md
# MAGIC ##### Load `health_tracker_sample_agg_pd_df` Pandas DataFrame

# COMMAND ----------

health_tracker_sample_agg_pd_df = (
  spark.read
  .format("delta")
  .load(goldPath + "health_tracker_sample_agg")
  .toPandas()
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Scipy Libraries

# COMMAND ----------

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define Scipy Utility Functions

# COMMAND ----------

# MAGIC %md
# MAGIC ###### `scatter_plot_with_decision_boundary(ax, features, target, model)`

# COMMAND ----------

def scatter_plot_with_decision_boundary(ax, features, target, model):
    mesh_step_size = 1

    x_min, x_max = features[0].min() - .5, features[0].max() + .5
    y_min, y_max = features[1].min() - .5, features[1].max() + .5
    xx, yy = np.meshgrid(
      np.arange(x_min, x_max, mesh_step_size),
      np.arange(y_min, y_max, mesh_step_size)
    )
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    for color, lifestyle in zip(
      ("blue", "orange", "green"),
      ('weight trainer', 'cardio trainer', 'sedentary')):
        two_dim_per_lifestyle = features[target.lifestyle == lifestyle]
        two_dim_per_lifestyle.plot(x=0, y=1, kind="scatter", c=color, label=lifestyle, ax=ax)
    Z = Z.reshape(xx.shape)
    ax.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.3)

# COMMAND ----------

# MAGIC %md
# MAGIC ###### `generate_feature_subsets(df)`

# COMMAND ----------

from itertools import combinations

def generate_feature_subsets(df):
  feature_columns = df.select_dtypes(exclude=["object"]).columns
  feature_subsets = []
  for i in range(1, 5):
    feature_subsets += [list(feat) for feat in combinations(feature_columns, i)]
  return feature_subsets

# COMMAND ----------

# MAGIC %md
# MAGIC ###### `generate_bootstrap_sample(df, lifestyles, n=5)`

# COMMAND ----------

def generate_bootstrap_sample(df, lifestyles, n=5):
  sample_df_list = []
  for lifestyle in lifestyles:
    sample_df_list.append(
      df[df.lifestyle == lifestyle].sample(n, replace=True)
    )
  return pd.concat(sample_df_list)

# COMMAND ----------

# MAGIC %md
# MAGIC ###### `generate_subsample_sets(df, lifestyles)`

# COMMAND ----------

from sklearn.preprocessing import LabelEncoder

def generate_subsample_sets(df, lifestyles):
  sample_sets = [generate_bootstrap_sample(df, lifestyles) for _ in range(10)]
  le = LabelEncoder()
  le.fit(df.lifestyle)

  for sample_set in sample_sets:
    sample_set["lifestyle_encoded"] = le.transform(sample_set.lifestyle)

  return sample_sets

# COMMAND ----------

# MAGIC %md
# MAGIC ###### `experiment_runner(feature_subset, model, param_grid={})`

# COMMAND ----------

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

def experiment_runner(feature_subset, model, param_grid={}):
  """Helper function to run MLflow experiment on a feature subset."""
  with mlflow.start_run() as run:
    "Build Subsets of Features."
    experimental_data_subsets = [
        sample_set[feature_subset]
        for sample_set in sample_sets
    ]

    targets = [
      sample_set["lifestyle_encoded"]
      for sample_set in sample_sets
    ]

    "Fit on each subset using LOO Cross-Validation."
    experimental_scores = []
    for features, target in zip(experimental_data_subsets, targets):
      gs = GridSearchCV(model, param_grid=param_grid, cv=5)
      gs.fit(features, target)
      score = gs.cv_results_["mean_test_score"][0]
      experimental_scores.append(score)

    "Record experiment results."
    mlflow.log_param("subset", feature_subset)
    model_name = (
      str(model.__class__)
      .split(".")[-1]
      .replace("'>","")
    )
    mlflow.log_param("model", model_name)
    mlflow.log_metric("mean score", np.mean(experimental_scores))
    mlflow.log_metric("std score", np.std(experimental_scores))

# COMMAND ----------

# MAGIC %md
# MAGIC ###### `retrieve_results(metrics, params)`

# COMMAND ----------

def retrieve_results(metrics, params):
  results = mlflow.search_runs()
  keys = []
  for metric in metrics:
    keys.append("metrics." + metric)
  for param in params:
    keys.append("params." + param)
  results = results[keys]
  results.dropna(inplace=True)
  results.drop_duplicates(inplace=True)
  return results

