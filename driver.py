import os
import re
import gc
import pyspark
import pandas as pd
import numpy as np

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession

from pyspark.sql.types import ArrayType, StringType, IntegerType, StructType, StructField
from pyspark.sql.functions import *

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, CrossValidator
from pyspark.mllib.evaluation import BinaryClassificationMetrics

from pyspark.ml.linalg import Vector
from pyspark.ml import Pipeline, PipelineModel

from pyspark.ml.feature import RegexTokenizer, HashingTF, StopWordsRemover, IDF, Tokenizer

from nltk.stem.snowball import SnowballStemmer

from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from classificationModels import trainLinearSVCModel, trainLogisticRegressionModel, trainNaiveBayesModel, trainGradientBoostModel, trainBinaryTreeModel


os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64/jre"
os.environ["SPARK_HOME"] = "venv/lib/python3.6/site-packages/pyspark"

TRAIN_FILE = "data/preprocessed/preprocessedDataWikiTrain"
TEST_FILE = "data/preprocessed/preprocessedDataWikiTest"
OTHER_DATA_FILE = "data/preprocessed/preprocessedDataLDT"

customSchema = StructType([
  StructField("id", StringType(), True),
  StructField("comment_text", StringType(), True),
  StructField("toxic", IntegerType(), True),
  StructField("severe_toxic", IntegerType(), True),
  StructField("obscene", IntegerType(), True),
  StructField("threat", IntegerType(), True),
  StructField("insult", IntegerType(), True),
  StructField("identity_hate", IntegerType(), True),
  StructField("clean", IntegerType(), True),
  StructField("hate_speech", IntegerType(), True)]
)

customSchema2 = StructType([
  StructField("id", IntegerType(), True),
  StructField("count", IntegerType(), True),
  StructField("hate_speech", IntegerType(), True),
  StructField("offensive_language", IntegerType(), True),
  StructField("neither", IntegerType(), True),
  StructField("class", IntegerType(), True),
  StructField("comment_text", StringType(), True),
  StructField("label", IntegerType(), True)]
)

def nullCheck(df):
  print("Check null values")
  df.where(df.comment_text.isNull()).show()
  df.where(df.comment_text == '').show()


def binaryData(spark):
  training_spark_df_binary = spark.read.csv(TRAIN_FILE, header='true', schema=customSchema)

  training_spark_df_binary = training_spark_df_binary.withColumnRenamed("clean", "label")
  training_spark_df_binary = training_spark_df_binary.drop('toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate', 'hate_speech')

  #  testing_spark_df_binary = spark.read.csv(TEST_FILE, header='true', multiLine=True, escape="\"", schema=customSchema)
  testing_spark_df_binary = spark.read.csv(TEST_FILE, header='true', schema=customSchema)

  testing_spark_df_binary = testing_spark_df_binary.withColumnRenamed("clean", "label")
  testing_spark_df_binary = testing_spark_df_binary.drop('toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate', 'hate_speech')

  ldt = spark.read.csv(OTHER_DATA_FILE, header='true', schema=customSchema2)
  ldt = ldt.drop("count", "hate_speech", "offensive_language", "neither", "class")

  return training_spark_df_binary, testing_spark_df_binary, ldt



def multiData(spark, col):
  training_spark_df_binary = spark.read.csv(TRAIN_FILE, header='true', schema=customSchema)
  training_spark_df_binary = training_spark_df_binary.withColumnRenamed(col, "label")
  training_spark_df_binary = training_spark_df_binary.select([c for c in training_spark_df_binary.columns if c in {'comment_text', 'label'}])
  
  testing_spark_df_binary = spark.read.csv(TEST_FILE, header='true', schema=customSchema)
  testing_spark_df_binary.show()

  testing_spark_df_binary = testing_spark_df_binary.withColumnRenamed(col, "label")
  testing_spark_df_binary = testing_spark_df_binary.select([c for c in testing_spark_df_binary.columns if c in {'comment_text', 'label'}])
  testing_spark_df_binary.show()

  return training_spark_df_binary, testing_spark_df_binary



def evaluateModel(model, data1, data2, data3):
  persistedModel = PipelineModel.load(model)

  train_prediction = persistedModel.transform(data1)
  test_prediction = persistedModel.transform(data2)
  otherDatasetTest = persistedModel.transform(data3)

  pd_prediction = test_prediction.select("*").toPandas()
  actual = pd_prediction["label"].tolist()
  pred = pd_prediction["prediction"].tolist()

  pd_prediction_other_dataset = otherDatasetTest.select("*").toPandas()
  actual_otherdataset = pd_prediction_other_dataset["label"].tolist()
  pred_otherdataset = pd_prediction_other_dataset["prediction"].tolist()

  tn, fp, fn, tp = confusion_matrix(actual, pred).ravel()
  print(confusion_matrix(actual, pred))
  print("true positive rate",tp / (tp + fp))
  print("true negative rate",tn / (tn + fp))

  # compute the accuracy score on training data
  correct_train = train_prediction.filter(train_prediction.label == train_prediction.prediction).count()  
  accuracy_train = correct_train/train_prediction.count() # store the training accuracy score
  print('Training set accuracy {:.2%} data items: {}, correct: {}'.format(accuracy_train,train_prediction.count(), correct_train))
      
  # Caculate the accuracy score for the best model 
  correct_test = test_prediction.filter(test_prediction.label == test_prediction.prediction).count()  
  accuracy_test = correct_test/test_prediction.count()
  print('Testing set accuracy {:.2%} data items: {}, correct: {}'.format(accuracy_test, test_prediction.count(), correct_test))
      
  # Caculate the accuracy score for the best model 
  correct_test_otherDataset = otherDatasetTest.filter(otherDatasetTest.label == otherDatasetTest.prediction).count()  
  accuracy_test_otherDataset = correct_test_otherDataset/otherDatasetTest.count()
  print('Testing set accuracy for other dataset is  {:.2%} data items: {}, correct: {}'.format(accuracy_test_otherDataset, otherDatasetTest.count(), correct_test_otherDataset))



if __name__ == '__main__':
  sc = SparkContext('local')
  spark = SparkSession(sc)

  training_spark_df_binary, testing_spark_df_binary, ldt = binaryData(spark)
  modelName = trainLogisticRegressionModel(training_spark_df_binary)

  evaluateModel(modelName, training_spark_df_binary, testing_spark_df_binary, ldt)


