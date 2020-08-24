# Databricks notebook source

# MAGIC %md-sandbox

# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
  <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
</div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Basic Classification

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
# MAGIC ### Create Feature and Target Objects

# COMMAND ----------

features = health_tracker_sample_agg_pd_df.select_dtypes(exclude=["object"])
target = health_tracker_sample_agg_pd_df[["lifestyle"]].copy()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Numerically Encode the Target
# MAGIC 
# MAGIC Pass the `lifestyle` column from the `target` DataFrame to the
# MAGIC `LabelEncoder`.

# COMMAND ----------

# ANSWER
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
target["lifestyle_encoded"] = le.fit_transform(target.lifestyle)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prepare a Two-Dimensional Projection of the Features using T-SNE

# COMMAND ----------

from sklearn.manifold import TSNE

np.random.seed(10)
tsne = TSNE(n_components=2)

features_in_two_dimensions = tsne.fit_transform(features)
features_in_two_dimensions = pd.DataFrame(features_in_two_dimensions)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Split the 2D Data into Training and Testing Sets

# COMMAND ----------

from sklearn.model_selection import train_test_split

(features_2d_train,
 features_2d_test,
 target_train,
 target_test) = train_test_split(features_in_two_dimensions,
                                 target.lifestyle_encoded)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fit a Logistic Regression Model to the 2D Data

# COMMAND ----------

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(penalty='none')

lr.fit(features_2d_train, target_train)
lr.score(features_2d_test, target_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Plot the Logistic Regression Fit and Decision Boundary
# MAGIC 
# MAGIC 🤖 We use the helper function, `scatter_plot_with_decision_boundary`.
# MAGIC 
# MAGIC This function is loaded above when we source the `scipy_stack` notebook.

# COMMAND ----------

fig, ax = plt.subplots(figsize=(20,6))

scatter_plot_with_decision_boundary(
  ax, features_in_two_dimensions, target, lr
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Split the Data into Training and Testing Sets

# COMMAND ----------

# ANSWER

(features_train,
 features_test,
 target_train,
 target_test) = train_test_split(features, target.lifestyle_encoded)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fit a Logistic Regression Model to the Data
# MAGIC 
# MAGIC 1. fit the model on the training data
# MAGIC 1. score the model on the testing data

# COMMAND ----------

# ANSWER
lr = LogisticRegression(penalty='none')

lr.fit(features_train, target_train)
lr.score(features_test, target_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fit a Logistic Regression Model to the Data
# MAGIC 
# MAGIC Fit the model again, increasing the maximum number of
# MAGIC iterations to avoid the convergence warning.
# MAGIC 
# MAGIC 1. fit the model on the training data
# MAGIC 1. score the model on the testing data

# COMMAND ----------

# ANSWER
lr = LogisticRegression(penalty='none', max_iter=1000)

lr.fit(features_train, target_train)
lr.score(features_train, target_train), lr.score(features_test, target_test)


# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2020 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>