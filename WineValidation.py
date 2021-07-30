from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, isnan, when
from pyspark.mllib.evaluation import MulticlassMetrics

if __name__ == "__main__":
    spark = SparkSession.builder.appName("src Validation").getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel('ERROR')
    data = spark.read.csv("s3://raghuvp/TrainingDataset.csv", header=True, inferSchema=True)
    data.show(10)
    new_data = data.select(*(col(c).cast("float").alias(c) for c in data.columns))
    data.printSchema()
    new_data.select([count(when(col(c).isNull(), c)).alias(c) for c in new_data.columns]).show()
    cols = new_data.columns
    cols.remove("quality")
    assembler = VectorAssembler(inputCols=cols, outputCol="features")
    d = assembler.transform(new_data)
    d.select("features", 'quality').show(truncate=False)
    ss = StandardScaler().setInputCol("features").setOutputCol("Scaled_features")
    d = ss.fit(d).transform(d)
    d.select("features", 'quality', 'Scaled_features').show()
    train = d.select("Scaled_features", "quality")
    train.show()

    validData = spark.read.csv("s3://raghuvp/ValidationDataset.csv", header=True, inferSchema=True, sep=';')
    validData.show(5)
    new_data1 = validData.select(*(col(c).cast("float").alias(c) for c in validData.columns))
    validData.printSchema()
    new_data1.select([count(when(col(c).isNull(), c)).alias(c) for c in new_data1.columns]).show()
    colsV = new_data1.columns
    colsV.remove("quality")
    assembler1 = VectorAssembler(inputCols=colsV, outputCol="features")
    v = assembler1.transform(new_data1)
    v.select("features", 'quality').show(truncate=False)
    ssv = StandardScaler().setInputCol("features").setOutputCol("Scaled_features")
    v = ssv.fit(v).transform(v)
    v.select("features", 'quality', 'Scaled_features').show()
    test = v.select("Scaled_features", "quality")
    test.show()
    lr = LogisticRegression(labelCol="quality", featuresCol="Scaled_features", maxIter=40)
    model = lr.fit(lr)
    pred = model.transform(test)
    metrics = model.summary
    fMeasure = metrics.weightedFMeasure()
    print(fMeasure)
