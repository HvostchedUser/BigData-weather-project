from pyspark.sql import SparkSession
from pyspark.sql.functions import col, unix_timestamp
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit

# Initialize Spark session
spark = SparkSession.builder\
    .appName("WeatherSummaryPrediction")\
    .config("spark.executor.memory", "40g") \
    .config("spark.driver.memory", "40g") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")  # Reduce log verbosity

# Load data
df = spark.read.csv("../data/weatherHistory.csv", header=True, inferSchema=True)
df = df.withColumn("Formatted Date", unix_timestamp(col("Formatted Date")))  # Convert 'Formatted Date' to Unix time
df = df.drop("Daily Summary")  # Drop 'Daily Summary'

# Handle categorical columns with StringIndexer
precipTypeIndexer = StringIndexer(inputCol="Precip Type", outputCol="Precip Type Indexed", handleInvalid="keep").fit(df)
summaryIndexer = StringIndexer(inputCol="Summary", outputCol="label").fit(df)

# Feature assembler
featureCols = [c for c in df.columns if c not in ["Summary", "Precip Type"]]
assembler = VectorAssembler(inputCols=featureCols + ["Precip Type Indexed"], outputCol="features")

# Data splitting
(trainingData, testData) = df.randomSplit([0.7, 0.3])

# Random Forest Classifier
rf = RandomForestClassifier(labelCol="label", featuresCol="features")

# Logistic Regression
lr = LogisticRegression(labelCol="label", featuresCol="features")

# Hyperparameter tuning setup
paramGrid_rf = ParamGridBuilder() \
    .addGrid(rf.maxDepth, [5, 10, 15]) \
    .addGrid(rf.numTrees, [10, 20, 30]) \
    .build()

paramGrid_lr = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.01, 0.1, 0.5]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    .build()

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

# TrainValidationSplit
tvs_rf = TrainValidationSplit(estimator=rf,
                              estimatorParamMaps=paramGrid_rf,
                              evaluator=evaluator,
                              trainRatio=0.75)

tvs_lr = TrainValidationSplit(estimator=lr,
                              estimatorParamMaps=paramGrid_lr,
                              evaluator=evaluator,
                              trainRatio=0.75)

# Pipelines
pipeline_rf = Pipeline(stages=[precipTypeIndexer, summaryIndexer, assembler, tvs_rf])
pipeline_lr = Pipeline(stages=[precipTypeIndexer, summaryIndexer, assembler, tvs_lr])

# Fit models
model_rf = pipeline_rf.fit(trainingData)
model_lr = pipeline_lr.fit(trainingData)

# Make predictions
predictions_rf = model_rf.transform(testData)
predictions_lr = model_lr.transform(testData)

# Evaluate models
accuracy_rf = evaluator.evaluate(predictions_rf)
accuracy_lr = evaluator.evaluate(predictions_lr)

print(f"Random Forest Model Accuracy: {accuracy_rf}")
print(f"Logistic Regression Model Accuracy: {accuracy_lr}")

# Save the models
model_rf.write().overwrite().save("../models/random_forest_model")
model_lr.write().overwrite().save("../models/logistic_regression_model")

spark.stop()

