from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.functions import col, count, isnan, when
from pyspark.mllib.evaluation import MulticlassMetrics

if __name__ == "__main__":
    sc = SparkContext(master="local[2]")
    spark = SparkSession.builder.appName("src Prediction").getOrCreate()
    data = spark.read.csv("s3://raghuvp/TrainingDataset.csv", header=True, inferSchema=True)
    data.show(10)
    data.printSchema()

    new_data = data.select(*(col(c).cast("float").alias(c) for c in data.columns))
    data.printSchema()

    data.select([count(when(col(c).isNull(), c)).alias(c) for c in new_data.columns]).show()
    cols = new_data.columns
    cols.remove("quality")

    assembler = VectorAssembler(inputCols=cols, outputCol="features")
    d = assembler.transform(new_data)
    d.select("features", 'quality').show(truncate=False)

    ss = StandardScaler().setInputCol("features").setOutputCol("Scaled_features")
    d = ss.fit(d).transform(d)
    d.select("Scaled_features", 'quality').show()
    assembled_data = d.select("Scaled_features", "quality")
    assembled_data.show()

    train, test = d.randomSplit([0.7, 0.3], seed=42)
    train.show()
    test.show()

    lr = LogisticRegression(labelCol="quality", featuresCol="Scaled_features", maxIter=40)
    model = lr.fit(train)
    pred = model.transform(test)
    pred.select("Scaled_features", "quality").show(5)
    metrics = model.summary
    f_measure = metrics.weightedFMeasure()
    print(f_measure)
