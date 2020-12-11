import numpy as np
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.mllib.tree import GradientBoostedTrees
from pyspark.ml.classification import GBTClassifier

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
from pyspark.ml.classification import GBTClassifier

# import modules for feature transformation
from pyspark.ml.linalg import Vector
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import RegexTokenizer
from pyspark.ml.classification import DecisionTreeClassifier
#Tokenize into words

from nltk.stem.snowball import SnowballStemmer
from pyspark.sql.types import ArrayType, StringType, IntegerType, StructType, StructField
from pyspark.sql.functions import *

from sklearn.metrics import confusion_matrix

# Check the pyspark version
import pyspark
print(pyspark.__version__)

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64/jre"
os.environ["SPARK_HOME"] = "venv/lib/python3.6/site-packages/pyspark"

TRAIN_FILE = "data/train.csv"
TEST_DATA_FILE = "data/labeled_data_toxic.csv"

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

sc = SparkContext('local')
spark = SparkSession(sc)

# train = spark.read.csv(TRAIN_FILE, header='true', sep=',', multiLine=True, escape="\"")
train = spark.read.csv(TRAIN_FILE, header='true', multiLine=True, escape="\"", schema=customSchema)
ldt = spark.read.csv(TEST_DATA_FILE, header='true', multiLine=True, escape="\"")

ldt = ldt.withColumn("count", ldt["count"].cast(IntegerType()))
ldt = ldt.withColumn("hate_speech", ldt["hate_speech"].cast(IntegerType()))
ldt = ldt.withColumn("offensive_language", ldt["offensive_language"].cast(IntegerType()))
ldt = ldt.withColumn("neither", ldt["neither"].cast(IntegerType()))
ldt = ldt.withColumn("class", ldt["class"].cast(IntegerType()))


ldt = ldt.withColumn("tweet", regexp_replace(col("tweet"), "[\n\r\W]", " "))
ldt = ldt.withColumn("tweet", regexp_replace(col("tweet"), r'\d+', ""))


def check_clean_ldt(langClass):
    if langClass == 2:
        return 1
    else:
        return 0

ldtCreateLabel = udf(lambda langClass: check_clean_ldt(langClass), IntegerType())
ldt = ldt.withColumn("label", ldtCreateLabel(ldt["class"]))

ldt = ldt.withColumnRenamed("tweet", "comment_text")
ldt = ldt.withColumnRenamed("_c0", "id")
ldt = ldt.drop("count", "hate_speech", "offensive_language", "neither", "class")


stemmer = SnowballStemmer("english")
def stemming(sentence):
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence

# Stem text
stemmer_udf = udf(lambda line: stemming(line), StringType())


train.na.drop()
train = train.withColumn("comment_text", regexp_replace(col("comment_text"), "[\n\r\W]", " "))
train = train.withColumn("comment_text", regexp_replace(col("comment_text"), r'\d+', ""))

train = train.withColumn("comment_text", stemmer_udf("comment_text"))

def check_clean(toxic, severe_toxic, obscene, threat, insult, identity_hate):
    if (toxic + severe_toxic + obscene + threat + insult + identity_hate) > 0:
        return 0
    else:
        return 1

mergeCols = udf(lambda toxic, severe_toxic, obscene, threat, insult, identity_hate: check_clean(toxic, severe_toxic, obscene, threat, insult, identity_hate), IntegerType())

train = train.withColumn("clean", mergeCols(train["toxic"], train["severe_toxic"], train["obscene"], train["threat"], train["insult"], train["identity_hate"]))
train = train.drop('toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate')
train = train.withColumnRenamed("clean", "label")


finalPrediction_lr= {}
finalPrediction_dt={}
finalPrediction_boostedTree ={}

# read pickled model via pipeline api
boostModel = PipelineModel.load("BoostTreeModel")
otherDatasetTest = boostModel.transform(ldt)
pd_prediction_other_dataset = otherDatasetTest.select("*").toPandas()
actual_otherdataset = pd_prediction_other_dataset["label"].tolist()
pred_otherdataset_boost = pd_prediction_other_dataset["prediction"].tolist()


# read pickled model via pipeline api
svmModel = PipelineModel.load("LinearSVMModel")
otherDatasetTest = svmModel.transform(ldt)
pd_prediction_other_dataset = otherDatasetTest.select("*").toPandas()
actual_otherdataset = pd_prediction_other_dataset["label"].tolist()
pred_otherdataset_svm = pd_prediction_other_dataset["prediction"].tolist()


# read pickled model via pipeline api
lrModel = PipelineModel.load("LogisticRegressionModel")
otherDatasetTest = lrModel.transform(ldt)
pd_prediction_other_dataset = otherDatasetTest.select("*").toPandas()
actual_otherdataset = pd_prediction_other_dataset["label"].tolist()
pred_otherdataset_lr = pd_prediction_other_dataset["prediction"].tolist()


label = ldt.select("label").rdd.flatMap(lambda x: x).collect()

#Voting:
votingPred =[]
for i in range(len(pred_otherdataset_lr)):
  sumRound = pred_otherdataset_lr[i] + pred_otherdataset_svm[i] + pred_otherdataset_boost[i]
  if(sumRound>=2):
    votingPred.append(1)
  else:
    votingPred.append(0)
  
tn, fp, fn, tp = confusion_matrix(label, votingPred).ravel()
print(confusion_matrix(label, votingPred))
print("voting precision",tp / (tp + fp))
print("voting recall",tp / (tp + fn))
correct_test=0
for i in range(len(votingPred)):
  if(votingPred[i]==label[i]):
    correct_test+=1
# Caculate the accuracy score for the best model 
accuracy_test = correct_test/len(votingPred)
print('Majority voting Testing set accuracy {:.2%} data items: {}, correct: {}'.format(accuracy_test, len(votingPred), correct_test))
      