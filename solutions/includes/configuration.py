# Databricks notebook source

# MAGIC %md
# MAGIC ### Define Data Paths

# COMMAND ----------

# ANSWER
username = "dbacademy"

# COMMAND ----------

projectPath     = f"/dbacademy/{username}/mlmodels/profile/"
landingPath     = projectPath + "landing/"
silverDailyPath = projectPath + "daily/"
dimUserPath     = projectPath + "users/"
goldPath        = projectPath + "gold/"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Configure Database

# COMMAND ----------

spark.sql(f"CREATE DATABASE IF NOT EXISTS dbacademy_{username}")
spark.sql(f"USE dbacademy_{username}");

