# Databricks notebook source

# MAGIC %md-sandbox

# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
  <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
</div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Results Analysis

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
# MAGIC ### Generate Subsample Sets

# COMMAND ----------

lifestyles = health_tracker_sample_agg_pd_df.lifestyle.unique()
sample_sets = generate_subsample_sets(
  health_tracker_sample_agg_pd_df,
  lifestyles
)

feature_subsets = generate_feature_subsets(health_tracker_sample_agg_pd_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run Experiments Using Decision Tree Classification On Each Feature Subset

# COMMAND ----------

from sklearn.tree import DecisionTreeClassifier

for feature_subset in feature_subsets:
  experiment_runner(
    feature_subset=feature_subset,
    model=DecisionTreeClassifier()
  )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Retrieve Results

# COMMAND ----------

results = retrieve_results(metrics=["mean score", "std score"],
                           params=["subset"])
results

# COMMAND ----------

# MAGIC %md
# MAGIC ### Display Correlation Plot

# COMMAND ----------

features = health_tracker_sample_agg_pd_df.select_dtypes(exclude=["object"])
corr = features.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask, 0)] = True
sns.heatmap(corr, mask=mask, square=True, annot=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run Experiments Using Logistic Regression On Each Feature Subset
# MAGIC 
# MAGIC Use a Logistic Regression with maximum iterations of 10000 and the
# MAGIC penalty set to `'none'`.

# COMMAND ----------

# TODO
from sklearn.linear_model import LogisticRegression

for feature_subset in feature_subsets:
  experiment_runner(
    feature_subset=feature_subset,
    model=FILL_THIS_IN
  )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Retrieve Results and Display Top Performing Models by Bias

# COMMAND ----------

results = retrieve_results(metrics=["mean score", "std score"],
                           params=["model", "subset"])
results["bias"] = 1 - results["metrics.mean score"]
results["variance"] = results["metrics.std score"]**2
results.drop(["metrics.mean score", "metrics.std score"], axis=1, inplace=True)
results.sort_values("bias").head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Retrieve Results and Display Top Performing Models by Tradeoff

# COMMAND ----------

results["n_terms"] = results["params.subset"].apply(lambda x: x.count(",") + 1)
results["tradeoff"] = results["bias"]**2 + results["variance"]
results.sort_values("tradeoff").head(10)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Plot Models by Tradeoff and Number of Terms

# COMMAND ----------

plt.figure(figsize=(20,10))

for _, (_, description, _, _, n_terms, tradeoff) in results.sort_values("tradeoff").head(10).iterrows():
   plt.scatter(n_terms, tradeoff, s=100*n_terms, label=description)
plt.ylim(0, 0.1)
plt.legend()


# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2020 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>