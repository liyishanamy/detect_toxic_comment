import os
import findspark
import pandas as pd
import pyspark
import numpy as np
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.ml.feature import StopWordsRemover 
from pyspark.ml import Pipeline
from pyspark.sql import SQLContext
from imblearn.over_sampling import SMOTE
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pyspark.sql.types as T
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
from sklearn.model_selection import train_test_split
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from sklearn.metrics import confusion_matrix
from pyspark.ml.classification import LinearSVC


# import modules for feature transformation
from pyspark.ml.linalg import Vector
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import HashingTF,StopWordsRemover,IDF,Tokenizer
from pyspark.ml.feature import NGram
#Tokenize into words

from nltk.stem.snowball import SnowballStemmer


os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64/jre"
os.environ["SPARK_HOME"] = "venv/lib/python3.6/site-packages/pyspark"

findspark.init()

from pyspark.sql import SparkSession

spark = SparkSession.builder.master("local[*]").getOrCreate()
# Test the spark 
df = spark.createDataFrame([{"hello": "world"} for x in range(1000)])

df.show(3, False)

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
TRAIN_FILE = "data/train.csv"
# TEST_DATA_FILE = "/content/drive/My Drive/mgc_dataset/CS651Final/jigsaw-toxic-comment-classification-challenge/test.csv"
# TEST_LABEL = "/content/drive/My Drive/mgc_dataset/CS651Final/jigsaw-toxic-comment-classification-challenge/test_labels.csv"
TEST_DATA_FILE = "data/labeled_data_toxic.csv"
train = pd.read_csv(TRAIN_FILE)
test_data = pd.read_csv(TEST_DATA_FILE)
# test_label =  pd.read_csv(TEST_LABEL)
print("training data shape",train.shape)
print("testing data shape",test_data.shape)
# print("testing data label shape",test_label.shape)

testData = test_data["tweet"]
testLabel = test_data["class"]
print(testLabel)
print(testData)

# # Stemming and Lemmatizing 
# import nltk

# nltk.download('wordnet')
# from nltk.stem import WordNetLemmatizer, PorterStemmer
# lemmatizer = WordNetLemmatizer()
# stemmer = PorterStemmer()
stemmer = SnowballStemmer("english")
def stemming(sentence):
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence
train['comment_text'] = train['comment_text'].apply(stemming)
test_data["tweet"] = test_data["tweet"].apply(stemming)


#lower case
train['comment_text'] = train['comment_text'].apply(lambda x:x.lower())
test_data["tweet"] =test_data["tweet"].apply(lambda x:x.lower())
# Get rid of the special charactors
train['comment_text'] = train['comment_text'].str.replace('\W', ' ')
test_data['tweet'] = test_data['tweet'].str.replace('\W', ' ')
#get rid of the digit
train['comment_text']  = train['comment_text'].str.replace(r'\d+','')
test_data['tweet']  = test_data['tweet'].str.replace(r'\d+','')


newTrain = []
newTest =[]
for i in range(len(train['comment_text'])):
  temp= re.sub(' +', ' ', train['comment_text'][i]).strip()
  train['comment_text'][i] = temp
for i in range(len(test_data['tweet'])):
  temp= re.sub(' +', ' ', test_data['tweet'][i]).strip()
  test_data['tweet'][i] = temp

rowsums=train.iloc[:,2:].sum(axis=1)
train['clean']=(rowsums==0)
train['clean'].sum()
train['clean']=np.where(train['clean']==True,1,0)

train_binary_classification = pd.concat([train["id"],train["comment_text"],train["clean"]],axis=1)

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

#Clean=1/toxic=0
test = []
for i in test_data["class"]:
  if(i==2):
    test.append(1)
  else:
    test.append(0)

rebalanceDatasetTechnique = ["undersampling"] #,"oversampling","no technique"] 
for technique in rebalanceDatasetTechnique:
    print('**Processing {} on imbalanced data...**'.format(technique))
    df = spark.createDataFrame(train_binary_classification)
    X_train, X_test, y_train, y_test = train_test_split(pd.concat([df['id'],df['comment_text']],axis=1), df['clean'], test_size=0.2)

    training_spark_df_binary = spark.createDataFrame(data=pd.concat([X_train,y_train],axis=1),schema=["id","comment_text","label"])
    testing_spark_df_binary = spark.createDataFrame(data=pd.concat([X_test,y_test],axis=1),schema=["id","comment_text","label"])
    otherDatasetTest_df_binary = spark.createDataFrame(data=pd.concat([test_data["tweet"],pd.DataFrame(test)],axis=1),schema=["comment_text","label"])
    # ,5000,10000])\
    # [0.1, 0.05, 0.01]) \
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
    cvModel.bestModel.save("model")
    # Make predictions
    train_prediction = cvModel.transform(training_spark_df_binary)
    test_prediction = cvModel.transform(testing_spark_df_binary)
    otherDatasetTest = cvModel.transform(otherDatasetTest_df_binary)

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
