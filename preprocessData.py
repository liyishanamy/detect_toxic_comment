import os
import re
import pandas as pd
import pyspark
import numpy as np
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StopWordsRemover 
from pyspark.sql import SQLContext

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import HashingTF,StopWordsRemover,IDF,Tokenizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.ml.classification import NaiveBayes

from pyspark.ml.linalg import Vector
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import RegexTokenizer

from nltk.stem.snowball import SnowballStemmer
from pyspark.sql.types import ArrayType, StringType, IntegerType, StructType, StructField
from pyspark.sql.functions import *

from sklearn.metrics import confusion_matrix

import pyspark
print(pyspark.__version__)

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64/jre"
os.environ["SPARK_HOME"] = "venv/lib/python3.6/site-packages/pyspark"

TRAIN_FILE = "data/train.csv"
TEST_DATA_FILE = "data/labeled_data_toxic.csv"

stemmer = SnowballStemmer("english")
def stemming(sentence):
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence

def preprocessDataWiki(spark, split=True):
  customSchema = StructType([
    StructField("id", StringType(), True),
    StructField("comment_text", StringType(), True),
    StructField("toxic", IntegerType(), True),
    StructField("severe_toxic", IntegerType(), True),
    StructField("obscene", IntegerType(), True),
    StructField("threat", IntegerType(), True),
    StructField("insult", IntegerType(), True),
    StructField("identity_hate", IntegerType(), True)]
  )

  train = spark.read.csv(TRAIN_FILE, header='true', multiLine=True, escape="\"", schema=customSchema)

  train.na.drop()

  train = train.withColumn("comment_text", regexp_replace(col("comment_text"), "[\n\r\W]", " "))
  train = train.withColumn("comment_text", regexp_replace(col("comment_text"), r'\d+', ""))

  stemmer_udf = udf(lambda line: stemming(line), StringType())
  train = train.withColumn("comment_text", stemmer_udf("comment_text"))

  def checkClean(toxic, severe_toxic, obscene, threat, insult, identity_hate):
      if (toxic + severe_toxic + obscene + threat + insult + identity_hate) > 0:
          return 0
      else:
          return 1

  mergeCols = udf(lambda toxic, severe_toxic, obscene, threat, insult, identity_hate: checkClean(toxic, severe_toxic, obscene, threat, insult, identity_hate), IntegerType())
  train = train.withColumn("clean", mergeCols(train["toxic"], train["severe_toxic"], train["obscene"], train["threat"], train["insult"], train["identity_hate"]))

  if split == True:
    training_spark_df_binary, testing_spark_df_binary = train.randomSplit([0.8, 0.2], seed = 2018)
    training_spark_df_binary.write.format("csv").save("data/preprocessed/preprocessedDataWikiTrain", header="true")
    testing_spark_df_binary.write.format("csv").save("data/preprocessed/preprocessedDataWikiTest", header="true")
  else:
    train.write.format("csv").save("data/preprocessed/precessedDataWiki", header="true")


def preprocessDataOther(spark):
  ldt = spark.read.csv(TEST_DATA_FILE, header='true', multiLine=True, escape="\"")

  ldt = ldt.withColumn("count", ldt["count"].cast(IntegerType()))
  ldt = ldt.withColumn("hate_speech", ldt["hate_speech"].cast(IntegerType()))
  ldt = ldt.withColumn("offensive_language", ldt["offensive_language"].cast(IntegerType()))
  ldt = ldt.withColumn("neither", ldt["neither"].cast(IntegerType()))
  ldt = ldt.withColumn("class", ldt["class"].cast(IntegerType()))

  ldt = ldt.withColumn("tweet", regexp_replace(col("tweet"), "[\n\r\W]", " "))
  ldt = ldt.withColumn("tweet", regexp_replace(col("tweet"), r'\d+', ""))

  def checkCleanLdt(langClass):
      if langClass == 2:
          return 1
      else:
          return 0

  ldtCreateLabel = udf(lambda langClass: checkCleanLdt(langClass), IntegerType())

  ldt = ldt.withColumn("label", ldtCreateLabel(ldt["class"]))

  ldt = ldt.withColumnRenamed("tweet", "comment_text")
  ldt = ldt.withColumnRenamed("_c0", "id")

  stemmer_udf = udf(lambda line: stemming(line), StringType())
  ldt = ldt.withColumn("comment_text", stemmer_udf("comment_text"))

  ldt.write.format("csv").save("data/preprocessed/preprocessDataOther", header="true")


if __name__ == '__main__':
  sc = SparkContext('local')
  spark = SparkSession(sc)

  preprocessDataWiki(spark)
  preprocessDataOther(spark)


