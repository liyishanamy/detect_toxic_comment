from pyspark.ml.linalg import Vector
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import HashingTF, StopWordsRemover, IDF, Tokenizer

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, CrossValidator

from pyspark.ml.classification import LinearSVC, LogisticRegression, NaiveBayes, GBTClassifier, DecisionTreeClassifier


def trainLinearSVCModel(data): 
  tokenizer = Tokenizer().setInputCol("comment_text").setOutputCol("words")
  remover= StopWordsRemover().setInputCol("words").setOutputCol("filtered").setCaseSensitive(False)
  hashingTF = HashingTF().setNumFeatures(1000).setInputCol("filtered").setOutputCol("rawFeatures")
  idf = IDF().setInputCol("rawFeatures").setOutputCol("features").setMinDocFreq(0)
  lr = LinearSVC(labelCol="label", featuresCol="features", maxIter=20)
  pipeline=Pipeline(stages=[tokenizer, remover, hashingTF, idf, lr])

  paramGrid = ParamGridBuilder()\
      .addGrid(hashingTF.numFeatures,[200, 500, 1000, 5000]) \
      .addGrid(lr.regParam, [0.01, 0.05, 0.1]) \
      .build()

  crossval = TrainValidationSplit(estimator=pipeline,
                              estimatorParamMaps=paramGrid,
                              evaluator=BinaryClassificationEvaluator().setMetricName('areaUnderPR'), # set area Under precision-recall curve as the evaluation metric
                              # 80% of the data will be used for training, 20% for validation.
                              trainRatio=0.8)

  cvModel = crossval.fit(data)
  modelName = "LinearSVCModel"
  cvModel.bestModel.write().overwrite().save(modelName)

  return modelName


def trainLogisticRegressionModel(data):
  tokenizer = Tokenizer().setInputCol("comment_text").setOutputCol("words")
  remover= StopWordsRemover().setInputCol("words").setOutputCol("filtered").setCaseSensitive(False)
  hashingTF = HashingTF().setNumFeatures(1000).setInputCol("filtered").setOutputCol("rawFeatures")
  idf = IDF().setInputCol("rawFeatures").setOutputCol("features").setMinDocFreq(0)
  lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10)
  pipeline=Pipeline(stages=[tokenizer, remover, hashingTF, idf, lr])

  paramGrid = ParamGridBuilder()\
    .addGrid(hashingTF.numFeatures,[200, 500, 800, 2000])\
    .addGrid(lr.regParam, [0.01, 0.05, 0.1])\
    .addGrid(lr.elasticNetParam, [0.0,0.3,0.6,0.8])\
    .build()

  crossval = TrainValidationSplit(estimator=pipeline,
                            estimatorParamMaps=paramGrid,
                            evaluator=BinaryClassificationEvaluator().setMetricName('areaUnderPR'), # set area Under precision-recall curve as the evaluation metric
                            # 80% of the data will be used for training, 20% for validation.
                            trainRatio=0.8)

  cvModel = crossval.fit(data)
  modelName = "LogisticRegressionModel"
  cvModel.bestModel.write().overwrite().save(modelName)

  return modelName


def trainNaiveBayesModel(data):
  tokenizer = Tokenizer().setInputCol("comment_text").setOutputCol("words")
  remover= StopWordsRemover().setInputCol("words").setOutputCol("filtered").setCaseSensitive(False)
  hashingTF = HashingTF().setNumFeatures(1000).setInputCol("filtered").setOutputCol("rawFeatures")
  idf = IDF().setInputCol("rawFeatures").setOutputCol("features").setMinDocFreq(0)
  nb = NaiveBayes(labelCol="label", featuresCol="features")
  pipeline=Pipeline(stages=[tokenizer, remover, hashingTF, idf, nb])

  paramGrid = ParamGridBuilder()\
    .addGrid(hashingTF.numFeatures,[200, 500, 1000, 5000]) \
    .addGrid(nb.smoothing, [0.5, 1, 1.5, 2]) \
    .build()

  crossval = TrainValidationSplit(estimator=pipeline,
                            estimatorParamMaps=paramGrid,
                            evaluator=BinaryClassificationEvaluator().setMetricName('areaUnderPR'), # set area Under precision-recall curve as the evaluation metric
                            # 80% of the data will be used for training, 20% for validation.
                            trainRatio=0.8)

  cvModel = crossval.fit(data)
  modelName = "NaiveBayesModel"
  cvModel.bestModel.write().overwrite().save(modelName)

  return modelName


def trainGradientBoostModel(data): 
  tokenizer = Tokenizer().setInputCol("comment_text").setOutputCol("words")
  remover= StopWordsRemover().setInputCol("words").setOutputCol("filtered").setCaseSensitive(False)
  hashingTF = HashingTF().setNumFeatures(1000).setInputCol("filtered").setOutputCol("rawFeatures")
  idf = IDF().setInputCol("rawFeatures").setOutputCol("features").setMinDocFreq(0)
  gbt = GBTClassifier(labelCol="label", featuresCol="features", maxIter=10)
  pipeline=Pipeline(stages=[tokenizer, remover, hashingTF, idf, gbt])

  cvModel = pipeline.fit(data)
  modelName = "BoostTreeModel"
  cvModel.write().overwrite().save(modelName)
  return modelName


def trainBinaryTreeModel(data):
  tokenizer = Tokenizer().setInputCol("comment_text").setOutputCol("words")
  remover= StopWordsRemover().setInputCol("words").setOutputCol("filtered").setCaseSensitive(False)
  hashingTF = HashingTF().setNumFeatures(1000).setInputCol("filtered").setOutputCol("rawFeatures")
  idf = IDF().setInputCol("rawFeatures").setOutputCol("features").setMinDocFreq(0)
  dt = DecisionTreeClassifier(labelCol="label", maxDepth=30,featuresCol="features")
  pipeline=Pipeline(stages=[tokenizer, remover, hashingTF, idf, dt])

  paramGrid = ParamGridBuilder()\
            .addGrid(dt.maxDepth, [2, 5, 10, 20, 30]) \
            .addGrid(dt.maxBins, [10, 50, 80]) \
            .build()

  crossval = TrainValidationSplit(estimator=pipeline,
                            estimatorParamMaps=paramGrid,
                            evaluator=BinaryClassificationEvaluator().setMetricName('areaUnderPR'), # set area Under precision-recall curve as the evaluation metric
                            # 80% of the data will be used for training, 20% for validation.
                            trainRatio=0.8)

  cvModel = crossval.fit(data)
  modelName = "BinaryTreeModel"
  cvModel.bestModel.write().overwrite().save(modelName)

  return modelName



