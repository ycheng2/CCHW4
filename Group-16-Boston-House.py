from pyspark import SparkContext
from pyspark.mllib.regression import *
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD
from pyspark.mllib.feature import StandardScaler
from pyspark.mllib.feature import StandardScalerModel
from pyspark.mllib.linalg import Vectors

#read file in and remove header
sc = SparkContext("local", "Simple App")
houses = sc.textFile('/Users/Claudio/Apps/spark-2.0.0-bin-hadoop2.7/work/boston_house.csv')
header = houses.first()
headerless_houses = houses.filter(lambda line: line != header)

map_headerless = headerless_houses.map(lambda line: line.split(','))
features = map_headerless.map(lambda x : x[:-1])
label = map_headerless.map(lambda x : x[-1])

scaler = StandardScaler(withMean=True, withStd=True).fit(features)
scaledFeature = scaler.transform(features)
labeledDataPoints = label.zip(scaledFeature)
labeledDataPoints = labeledDataPoints.map(lambda x: LabeledPoint(x[0], [x[1:]]))

model = LinearRegressionWithSGD.train(labeledDataPoints, intercept=True)

test = sc.textFile('/Users/Claudio/Apps/spark-2.0.0-bin-hadoop2.7/work/verification.csv')
testheader = test.first()
testheaderless= test.filter(lambda line: line != testheader)
mapedtest = testheaderless.map(lambda x: x.split(','))
parsedTest=scaler.transform(mapedtest)

result = model.predict(parsedTest)
result.saveAsTextFile('output.txt')

sc.stop()




