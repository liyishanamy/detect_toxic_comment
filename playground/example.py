  tokenizer = Tokenizer() ...
  remover= StopWordsRemover() ...
  hashingTF = HashingTF().setNumFeatures(...) ...
  idf = IDF() ...
  lr = LinearSVC(labelCol="label", featuresCol="features", ...)
  pipeline=Pipeline(stages=[tokenizer, remover, hashingTF, idf, lr])




stemmer = SnowballStemmer("english")
def stemming(sentence):
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence

stemmer_udf = udf(lambda line: stemming(line), StringType())
train = train.withColumn("comment_text", stemmer_udf("comment_text"))


tokenizer = Tokenizer() ...
remover = StopWordsRemover() ...
ngram = NGram(n=2) ...
countVectorizer = CountVectorizer ...
idf = IDF() ...
lr = LinearSVC(labelCol="label",featuresCol="features",...)
pipeline = Pipeline(stages=[tokenizer, remover, ngram, countVectorizer, idf, lr])