import os
import re
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

from driver import binaryData, multiData


os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64/jre"
os.environ["SPARK_HOME"] = "venv/lib/python3.6/site-packages/pyspark"


def evaluate(model, data):
    # read pickled model via pipeline api
    persistedModel = PipelineModel.load("models/binary/"+model)

    transformedData = persistedModel.transform(data)
    pandasTransformedData = transformedData.select("*").toPandas()
    
    actual = pandasTransformedData["label"].tolist()
    pred = pandasTransformedData["prediction"].tolist()

    return actual, pred


def evaluateMulti(model, data, level):
    # read pickled model via pipeline api
    persistedModel = PipelineModel.load("multiclass/"+ level + "/" + model)

    transformedData = persistedModel.transform(data)
    pandasTransformedData = transformedData.select("*").toPandas()
    
    actual = pandasTransformedData["label"].tolist()
    pred = pandasTransformedData["prediction"].tolist()

    return actual, pred


def vote(models, data, numClasses="binary", level=""):
    preds = []
    actual = []
    for model in models:
        p = []
        if numClasses == "binary":
            actual, p = evaluate(model, data)
        else:
            actual, p = evaluateMulti(model, data, level)
        
        preds.append(p)
    
    numTests = len(preds[0])
    numVoters = len(preds)
    votingPred =[]
    for i in range(numTests):
        sumRound = 0
        for j in range(numVoters):
            sumRound += preds[j][i] 

        if(sumRound >= numVoters/2):
            votingPred.append(1)
        else:
            votingPred.append(0)

    tn, fp, fn, tp = confusion_matrix(actual, votingPred).ravel()
    print(confusion_matrix(actual, votingPred))
    print("voting precision", tp / (tp + fp))
    print("voting recall", tp / (tp + fn))
    
    correct_test=0

    for i in range(len(votingPred)):
        if (votingPred[i] == actual[i]):
            correct_test += 1

    # Caculate the accuracy score for the best model 
    accuracy_test = correct_test/len(votingPred)
    print('Majority voting Testing set accuracy {:.2%} data items: {}, correct: {}'.format(accuracy_test, len(votingPred), correct_test))


def evaluateMultiDriver(spark, models, level):
    training_spark_df_binary, testing_spark_df_binary = multiData(spark, level)
    vote(models, testing_spark_df_binary, numClasses="multi", level=level)


def evaluateBinaryDriver(spark, models):
    training_spark_df_binary, testing_spark_df_binary, ldt = binaryData(spark)
    vote(models, testing_spark_df_binary)


if __name__ == '__main__':
    sc = SparkContext('local')
    spark = SparkSession(sc)

    models = ["LinearSVCModel", "LogisticRegressionModel", "NaiveBayesModel", "BoostTreeModel", "BinaryTreeModel"]

    # evaluateMultiDriver(spark, models, "toxic")
    evaluateBinaryDriver(spark, models)
   