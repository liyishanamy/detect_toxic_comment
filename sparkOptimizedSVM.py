import os
import re
import pandas as pd
import pyspark
import numpy as np
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StopWordsRemover 
from pyspark.sql import SQLContext

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.linalg import Vector
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import HashingTF,StopWordsRemover,IDF,Tokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from sklearn.metrics import confusion_matrix
from pyspark.ml.classification import LinearSVC


# import modules for feature transformation
from pyspark.ml.linalg import Vector
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import RegexTokenizer
#Tokenize into words

from nltk.stem.snowball import SnowballStemmer
from pyspark.sql.types import ArrayType, StringType, IntegerType, StructType, StructField
from pyspark.sql.functions import *

# Check the pyspark version
import pyspark
print(pyspark.__version__)

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64/jre"
os.environ["SPARK_HOME"] = "venv/lib/python3.6/site-packages/pyspark"

TRAIN_FILE = "data/preprocessed/preprocessedDataWikiTrain"
TEST_FILE = "data/preprocessed/preprocessedDataWikiTest"
OTHER_DATA_FILE = "data/preprocessed/preprocessDataOther"

customSchema = StructType([
  StructField("id", StringType(), True),
  StructField("comment_text", StringType(), True),
  StructField("toxic", IntegerType(), True),
  StructField("severe_toxic", IntegerType(), True),
  StructField("obscene", IntegerType(), True),
  StructField("threat", IntegerType(), True),
  StructField("insult", IntegerType(), True),
  StructField("identity_hate", IntegerType(), True),
  StructField("clean", IntegerType(), True)]
)

customSchema2 = StructType([
  StructField("id", IntegerType(), True),
  StructField("count", IntegerType(), True),
  StructField("hate_speech", IntegerType(), True),
  StructField("offensive_language", IntegerType(), True),
  StructField("neither", IntegerType(), True),
  StructField("class", IntegerType(), True),
  StructField("label", IntegerType(), True)]
)

sc = SparkContext('local')
spark = SparkSession(sc)

training_spark_df_binary = spark.read.csv(TRAIN_FILE, header='true', schema=customSchema)
training_spark_df_binary.show()

training_spark_df_binary = training_spark_df_binary.withColumnRenamed("clean", "label")
training_spark_df_binary = training_spark_df_binary.drop('toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate')

training_spark_df_binary.show(100)
training_spark_df_binary.printSchema()


# training_spark_df_binary, testing_spark_df_binary = training_spark_df_binary.randomSplit([0.8, 0.2], seed = 2018)

# print("Check null values")
# print(training_spark_df_binary.where(col("comment_text").isNull()))

# testing_spark_df_binary = spark.read.csv(TEST_FILE, header='true', multiLine=True, escape="\"", schema=customSchema)

# testing_spark_df_binary = testing_spark_df_binary.withColumnRenamed("clean", "label")
# testing_spark_df_binary = testing_spark_df_binary.drop('toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate')

# ldt = spark.read.csv(OTHER_DATA_FILE, header='true', multiLine=True, escape="\"", schema=customSchema2)
# ldt = ldt.drop("count", "hate_speech", "offensive_language", "neither", "class")


tokenizer = Tokenizer().setInputCol("comment_text").setOutputCol("words")
remover= StopWordsRemover().setInputCol("words").setOutputCol("filtered").setCaseSensitive(False)
hashingTF = HashingTF().setNumFeatures(1000).setInputCol("filtered").setOutputCol("rawFeatures")
idf = IDF().setInputCol("rawFeatures").setOutputCol("features").setMinDocFreq(0)
lr = LinearSVC(labelCol="label", featuresCol="features", maxIter=20)
pipeline=Pipeline(stages=[tokenizer, remover, hashingTF, idf, lr])


paramGrid = ParamGridBuilder()\
    .addGrid(hashingTF.numFeatures,[1000]) \
    .addGrid(lr.regParam, [0.1]) \
    .build()

crossval = TrainValidationSplit(estimator=pipeline,
                            estimatorParamMaps=paramGrid,
                            evaluator=BinaryClassificationEvaluator().setMetricName('areaUnderPR'), # set area Under precision-recall curve as the evaluation metric
                            # 80% of the data will be used for training, 20% for validation.
                            trainRatio=0.8)

cvModel = crossval.fit(training_spark_df_binary)

cvModel.bestModel.write().overwrite().save("LinearSVMModel")

# read pickled model via pipeline api
persistedModel = PipelineModel.load("LinearSVMModel")

train_prediction = persistedModel.transform(training_spark_df_binary)
# test_prediction = persistedModel.transform(testing_spark_df_binary)
# otherDatasetTest = persistedModel.transform(ldt)

# pd_prediction = test_prediction.select("*").toPandas()
# actual = pd_prediction["label"].tolist()
# pred = pd_prediction["prediction"].tolist()

# pd_prediction_other_dataset = otherDatasetTest.select("*").toPandas()
# actual_otherdataset = pd_prediction_other_dataset["label"].tolist()
# pred_otherdataset = pd_prediction_other_dataset["prediction"].tolist()

# tn, fp, fn, tp = confusion_matrix(actual, pred).ravel()
# print(confusion_matrix(actual, pred))
# print("true positive rate",tp / (tp + fp))
# print("true negative rate",tn / (tn + fp))

# # compute the accuracy score on training data
# correct_train = train_prediction.filter(train_prediction.label == train_prediction.prediction).count()  
# accuracy_train = correct_train/train_prediction.count() # store the training accuracy score
# print('Training set accuracy {:.2%} data items: {}, correct: {}'.format(accuracy_train,train_prediction.count(), correct_train))
    
# # Caculate the accuracy score for the best model 
# correct_test = test_prediction.filter(test_prediction.label == test_prediction.prediction).count()  
# accuracy_test = correct_test/test_prediction.count()
# print('Testing set accuracy {:.2%} data items: {}, correct: {}'.format(accuracy_test, test_prediction.count(), correct_test))
    
# # Caculate the accuracy score for the best model 
# correct_test_otherDataset = otherDatasetTest.filter(otherDatasetTest.label == otherDatasetTest.prediction).count()  
# accuracy_test_otherDataset = correct_test_otherDataset/otherDatasetTest.count()
# print('Testing set accuracy for other dataset is  {:.2%} data items: {}, correct: {}'.format(accuracy_test_otherDataset, otherDatasetTest.count(), correct_test_otherDataset))