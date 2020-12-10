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
test = spark.read.csv(TEST_DATA_FILE, header='true')

train.na.drop()

train_clean = train.withColumn("comment_text", regexp_replace(col("comment_text"), "[\n\r\W]", " "))
train_clean = train_clean.withColumn("comment_text", regexp_replace(col("comment_text"), r'\d+', ""))

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
df_stemmed = train_clean.withColumn("comment_text", stemmer_udf("comment_text"))

def check_clean(toxic, severe_toxic, obscene, threat, insult, identity_hate):
    if (toxic + severe_toxic + obscene + threat + insult + identity_hate) > 0:
        return 0
    else:
        return 1

from pyspark.sql.types import IntegerType

mergeCols = udf(lambda toxic, severe_toxic, obscene, threat, insult, identity_hate: check_clean(toxic, severe_toxic, obscene, threat, insult, identity_hate), IntegerType())

df_clean = df_stemmed.withColumn("clean", mergeCols(df_stemmed["toxic"], df_stemmed["severe_toxic"], df_stemmed["obscene"], df_stemmed["threat"], df_stemmed["insult"], df_stemmed["identity_hate"]))

tokenizer = Tokenizer().setInputCol("comment_text").setOutputCol("words")
#Remove stopwords
remover= StopWordsRemover().setInputCol("words").setOutputCol("filtered").setCaseSensitive(False)
# ngram = NGram().setN(2).setInputCol("filtered").setOutputCol("ngrams")
#For each sentence (bag of words),use HashingTF to hash the sentence into a feature vector. 
hashingTF = HashingTF().setNumFeatures(1000).setInputCol("filtered").setOutputCol("rawFeatures")
#Create TF_IDF features
idf = IDF().setInputCol("rawFeatures").setOutputCol("features").setMinDocFreq(0)
# Create a Logistic regression model
lr = LinearSVC(labelCol="label", featuresCol="features", maxIter=20)
# Streamline all above steps into a pipeline
pipeline=Pipeline(stages=[tokenizer,remover,hashingTF,idf, lr])

df_clean = df_clean.drop('toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate')

df_clean = df_clean.withColumnRenamed("clean", "label")

training_spark_df_binary, testing_spark_df_binary = df_clean.randomSplit([0.8, 0.2], seed = 2018)

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

train_prediction = cvModel.transform(training_spark_df_binary)
test_prediction = cvModel.transform(testing_spark_df_binary)

pd_prediction = test_prediction.select("*").toPandas()
actual = pd_prediction["label"].tolist()
pred = pd_prediction["prediction"].tolist()

# pd_prediction_other_dataset = otherDatasetTest.select("*").toPandas()
# actual_otherdataset = pd_prediction_other_dataset["label"].tolist()
# pred_otherdataset = pd_prediction_other_dataset["prediction"].tolist()

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
# correct_test_otherDataset = otherDatasetTest.filter(otherDatasetTest.label == otherDatasetTest.prediction).count()  
# accuracy_test_otherDataset = correct_test_otherDataset/otherDatasetTest.count()
# print('Testing set accuracy for other dataset is  {:.2%} data items: {}, correct: {}'.format(accuracy_test_otherDataset, otherDatasetTest.count(), correct_test_otherDataset))