import sys

import pyspark
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.linalg import Vectors
from pyspark.mllib.linalg import Vectors
from pyspark.sql.functions import col
from pyspark.sql.functions import udf
from pyspark.sql.session import SparkSession
from pyspark.sql.types import StringType, DoubleType

if __name__ == "__main__":

    # Starting the spark session
    conf = pyspark.SparkConf().setAppName('winequality')
    sc = pyspark.SparkContext.getOrCreate()
    spark = SparkSession(sc)

    # Loading the dataset
    df = spark.read.csv('TrainingDataset.csv', inferSchema='true', header=True, sep=";")
    df.printSchema()
    # changing the 'quality' column name to 'label'
    for col_name in df.columns[1:-1] + ['quality']:
        df = df.withColumn(col_name, col(col_name).cast('float'))
    df = df.withColumnRenamed('quality', "label")


    # Convert to float format
    def string_to_float(x):
        return float(x)


    # catalog the data
    def catelogy(r):
        if 0 <= r <= 6.5:
            label = "bad"
        elif 6.5 < r <= 10:
            label = "good"
        else:
            label = "n/a"
        return label


    string_to_float_udf = udf(string_to_float, DoubleType())
    quality_udf = udf(lambda x: catelogy(x), StringType())

    df = df.withColumn("label", quality_udf("label"))


    def transData(data):
        return data.rdd.map(lambda r: [Vectors.dense(r[:-1]), r[-1]]).toDF(['features', 'label'])


    transformed = transData(df)

    labelIndexer = StringIndexer(inputCol='label', outputCol='indexedLabel').fit(transformed)

    featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(transformed)

    # TODO
    from pyspark.ml.feature import PCA

    data = transformed
    pca = PCA(k=6, inputCol="features", outputCol="pcaFeatures")
    model = pca.fit(data)

    result = model.transform(data).select("pcaFeatures")
    result.show(truncate=False)
    (trainingData, testData) = transformed.randomSplit([0.8, 0.2])
    print("Training Dataset Count: " + str(trainingData.count()))
    print("Test Dataset Count: " + str(testData.count()))

    # Train a RandomForest model.
    rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=8, maxDepth=20, seed=42)

    # Convert indexed labels back to original labels.
    labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel", labels=labelIndexer.labels)
    # Chain indexers and tree in a Pipeline
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf, labelConverter])
    # Train model.  This also runs the indexers.
    model = pipeline.fit(trainingData)

    # Make predictions.
    predictions = model.transform(testData)
    # Select example rows to display.

    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("accuracy = %g" % accuracy)

    model.save(sys.argv[2])
    print(sys.argv[2])
