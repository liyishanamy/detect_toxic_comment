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

from sklearn.metrics import confusion_matrix

# Check the pyspark version
import pyspark
print(pyspark.__version__)

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64/jre"
os.environ["SPARK_HOME"] = "venv/lib/python3.6/site-packages/pyspark"

TRAIN_FILE = "data/train.csv"
TEST_DATA_FILE = "data/labeled_data_toxic.csv"
# TEST_LABEL = "/content/drive/MyDrive/CS651Final/jigsaw-toxic-comment-classification-challenge/test_labels.csv"

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

train.na.drop()

train = train.withColumn("comment_text", regexp_replace(col("comment_text"), "[\n\r\W]", " "))
train = train.withColumn("comment_text", regexp_replace(col("comment_text"), r'\d+', ""))

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

from pyspark.sql.types import ArrayType, StringType

# Stem text
stemmer_udf = udf(lambda line: stemming(line), StringType())
train = train.withColumn("comment_text", stemmer_udf("comment_text"))

def check_clean(toxic, severe_toxic, obscene, threat, insult, identity_hate):
    if (toxic + severe_toxic + obscene + threat + insult + identity_hate) > 0:
        return 0
    else:
        return 1

from pyspark.sql.types import IntegerType

mergeCols = udf(lambda toxic, severe_toxic, obscene, threat, insult, identity_hate: check_clean(toxic, severe_toxic, obscene, threat, insult, identity_hate), IntegerType())

train = train.withColumn("clean", mergeCols(train["toxic"], train["severe_toxic"], train["obscene"], train["threat"], train["insult"], train["identity_hate"]))

tokenizer = Tokenizer().setInputCol("comment_text").setOutputCol("words")
#Remove stopwords
remover= StopWordsRemover().setInputCol("words").setOutputCol("filtered").setCaseSensitive(False)
# ngram = NGram().setN(2).setInputCol("filtered").setOutputCol("ngrams")
#For each sentence (bag of words),use HashingTF to hash the sentence into a feature vector. 
hashingTF = HashingTF().setNumFeatures(1000).setInputCol("filtered").setOutputCol("rawFeatures")
#Create TF_IDF features
idf = IDF().setInputCol("rawFeatures").setOutputCol("features").setMinDocFreq(0)
# Create a Logistic regression model
lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10)
# Streamline all above steps into a pipeline
pipeline=Pipeline(stages=[tokenizer,remover,hashingTF,idf, lr])

train = train.drop('toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate')
train = train.withColumnRenamed("clean", "label")

training_spark_df_binary, testing_spark_df_binary = train.randomSplit([0.8, 0.2], seed = 2018)

paramGrid = ParamGridBuilder()\
    .addGrid(hashingTF.numFeatures,[1000])\
    .addGrid(lr.regParam, [0.1])\
    .addGrid(lr.elasticNetParam, [0.3])\
    .build()

crossval = TrainValidationSplit(estimator=pipeline,
                            estimatorParamMaps=paramGrid,
                            evaluator=BinaryClassificationEvaluator().setMetricName('areaUnderPR'), # set area Under precision-recall curve as the evaluation metric
                            # 80% of the data will be used for training, 20% for validation.
                            trainRatio=0.8)

cvModel = crossval.fit(training_spark_df_binary)

cvModel.bestModel.write().overwrite().save("LogisticRegressionModel")

# read pickled model via pipeline api
from pyspark.ml.pipeline import PipelineModel
persistedModel = PipelineModel.load("LogisticRegressionModel")

train_prediction = persistedModel.transform(training_spark_df_binary)
test_prediction = persistedModel.transform(testing_spark_df_binary)
otherDatasetTest = persistedModel.transform(ldt)

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