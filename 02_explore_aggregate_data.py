# Databricks notebook source

# MAGIC %md-sandbox

# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
  <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
</div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Exploring the Aggregate Sample Data

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# MAGIC %run ./includes/configuration

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the Sample Data as a Pandas DataFrame
# MAGIC 
# MAGIC Recall that we wrote the sample data as a Delta table to
# MAGIC the path, `goldPath + "health_tracker_sample_agg"`.
# MAGIC 
# MAGIC 1. Use `spark.read` to read the Delta table as a Spark DataFrame.
# MAGIC 2. Use the `.toPandas()` DataFrame method to load the Spark
# MAGIC    DataFrame as a Pandas DataFrame.

# COMMAND ----------

# TODO
health_tracker_sample_agg_pd_df = (
  spark.read
  .format("delta")
  FILL_THIS_IN
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Scipy Libraries

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# COMMAND ----------

# MAGIC %md
# MAGIC ### Display the Unique Lifestyles
# MAGIC 
# MAGIC 🤔 Remember, Pandas DataFrames use the `.unique()` method to do this.
# MAGIC Spark DataFrames use the `.distinct()` method.
# MAGIC 
# MAGIC Make sure to specify the correct column, `lifestyle`.

# COMMAND ----------

# TODO
lifestyles = health_tracker_sample_agg_pd_df FILL_THIS_IN
lifestyles

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Feature and Target Objects

# COMMAND ----------

features = health_tracker_sample_agg_pd_df.select_dtypes(exclude=["object"])
target = health_tracker_sample_agg_pd_df[["lifestyle"]].copy()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Use `seaborn` to Display a Pair Plot
# MAGIC 
# MAGIC Generate a `pairplot` of all of the features.

# COMMAND ----------

# TODO
sns FILL_THIS_IN

# COMMAND ----------

# MAGIC %md
# MAGIC ### Use `seaborn` to Display a Distribution Plot for Each Feature
# MAGIC 
# MAGIC Generate a `distplot` for each feature.

# COMMAND ----------

# TODO
fig, ax = plt.subplots(1,4, figsize=(20,5))

for i, feature in enumerate(features):
  sns.FILL_THIS_IN(features[feature], ax=ax[i])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Use `seaborn` to Display a Distribution Plot for Each Feature, Colored by Lifestyle
# MAGIC 
# MAGIC Generate a `distplot` for each feature.

# COMMAND ----------

# TODO
fig, ax = plt.subplots(1,4, figsize=(20,5))

for i, feature in enumerate(features):
  for lifestyle in lifestyles:
    subset = features[target.FILL_THIS_IN == FILL_THIS_IN]
    sns.FILL_THIS_IN(subset[feature], ax=ax[i], label=lifestyle)
  ax[i].legend()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Display a Correlation Plot for Each Feature
# MAGIC 
# MAGIC 1. The `mask` should have the same shape as the `corr`
# MAGIC 2. The `sns.heatmap` takes the `corr` as argument as uses the `mask` to mask.

# COMMAND ----------

# TODO
corr = features.corr()
mask = np.zeros_like(FILL_THIS_IN)
mask[np.triu_indices_from(mask, 0)] = True
sns.heatmap(FILL_THIS_IN, mask=FILL_THIS_IN, square=True, annot=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prepare a Two-Dimensional Projection of the Features using T-SNE
# MAGIC 
# MAGIC You can read more on T-SNE
# MAGIC [here](https://colah.github.io/posts/2014-10-Visualizing-MNIST/).

# COMMAND ----------

from sklearn.manifold import TSNE

np.random.seed(10)
tsne = TSNE(n_components=2)

features_in_two_dimensions = tsne.fit_transform(features)
features_in_two_dimensions = pd.DataFrame(features_in_two_dimensions)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Plot the Two-Dimensional Projection Labeled by Lifestyle

# COMMAND ----------

colors = ("blue", "orange", "green")
fig, ax = plt.subplots(1,1,figsize=(12,6))

for color, lifestyle in zip(colors, lifestyles):
  two_dim_per_lifestyle = features_in_two_dimensions[target.lifestyle == lifestyle]
  two_dim_per_lifestyle.plot(x=0, y=1, kind="scatter", c=color, label=lifestyle, ax=ax)


# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2020 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>