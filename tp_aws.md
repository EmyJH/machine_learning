# exécuter script
```
spark-shell -i <name_script>
```

# Prépa eclipse
https://enahwe.wordpress.com/2015/11/25/how-to-configure-eclipse-for-developing-with-python-and-spark-on-hadoop/

# prepa machine locale
sudo ssh -i "NC_AWS_KEY" -N -f -L localhost:7775:localhost:7777 ubuntu@54.229.180.149

# Préparation AWS

* à changer dans :
``` 
bashrc : /usr/hdp/2.6.3.0-235/spark2
/usr/hdp/current/spark2-client/bin/pyspark: line 24: 
/usr/hdp/2.6.3.0-235/spark2/bin/load-spark-env.sh
line 77 :
/usr/hdp/2.6.3.0-235/spark2/bin/spark-submit
```
/home/ubuntu/.ssh/id_rsa

* patch
```
#!/bin/bash

set -e

sudo apt-get install -y git
git clone https://github.com/xianyi/OpenBlas.git
cd OpenBlas/
make clean
make -j
sudo mkdir /usr/lib64/OpenBLAS
sudo chmod o+w,g+w /usr/lib64/OpenBLAS/
make PREFIX=/usr/lib64/OpenBLAS install
sudo rm /etc/ld.so.conf.d/atlas-x86_64.conf
sudo ldconfig
sudo ln -sf /usr/lib64/OpenBLAS/lib/libopenblas.so /usr/lib64/libblas.so
sudo ln -sf /usr/lib64/OpenBLAS/lib/libopenblas.so /usr/lib64/libblas.so.3
sudo ln -sf /usr/lib64/OpenBLAS/lib/libopenblas.so /usr/lib64/libblas.so.3.5
sudo ln -sf /usr/lib64/OpenBLAS/lib/libopenblas.so /usr/lib64/libblas.so.3.5.0
sudo ln -sf /usr/lib64/OpenBLAS/lib/libopenblas.so /usr/lib64/liblapack.so
sudo ln -sf /usr/lib64/OpenBLAS/lib/libopenblas.so /usr/lib64/liblapack.so.3
sudo ln -sf /usr/lib64/OpenBLAS/lib/libopenblas.so /usr/lib64/liblapack.so.3.5
sudo ln -sf /usr/lib64/OpenBLAS/lib/libopenblas.so /usr/lib64/liblapack.so.3.5.0
```
# TP2 
* créer data :
```python
(en ubuntu)
sudo nano kmeans_data.txt
0,0,0
0.1,0.1,0.1
0.2,0.2,0.2
9,9,9
9.1,9.1,9.1
9.2,9.2,9.2
```

* mettre le fichier dans hdfs :
```python
sudo su hdfs
hdfs dfs -copyFromLocal /home/spark/kmeans_data.txt /tmp/kmeans_data.txt
hdfs dfs -copyFromLocal /home/spark/sample_fpgrowth.txt /tmp/sample_fpgrowth.txt

```

* lancement de pyspark :

```python
pyspark
from pyspark.mllib.clustering import KMeans
from numpy import array
from math import sqrt
data = sc.textFile("/tmp/kmeans_data.txt")

data = MLUtils.loadLibSVMFile(sc, '/tmp/libsvm.txt.txt').cache()

```

* préparer les données
```python
splitedData = data.map(lambda line: array([float(x) for x in line.split(',')]))
splitedData.take(6)
[array([0., 0., 0.]), array([0.1, 0.1, 0.1]), array([0.2, 0.2, 0.2]), array([9., 9., 9.]), array([9.1, 9.1, 9.1]), array([9.2, 9.2, 9.2])]
```
* créer le modèle
```python
clusters = KMeans.train(splitedData , 3, maxIterations=10)
```
* afficher les centres des clusters
```
print(clusters.clusterCenters)
```


# en local
```python
from pyspark import SparkContext
sc = SparkContext("local", "App Name", pyFiles=['MyFile.py', 'lib.zip', 'app.egg'])
```

# TP4
```python
from pyspark.mllib.fpm import FPGrowth
data = sc.textFile("/tmp/sample_fpgrowth.txt")
splitedData= data.map(lambda line: line.strip().split(','))
splitedData.take(6)
```

* Appliquer FP-Growth
```python
model = FPGrowth.train(splitedData, minSupport=0.2, numPartitions=10)
result = model.freqItemsets().collect()
```

* Afficher le résultat
```python
for item in result: print(item)
```

# TP5

* Créer le fichier
```python
sudo nano sample_libsvm_data.txt
hdfs dfs -copyFromLocal /home/spark/sample_libsvm_data.txt /tmp/sample_libsvm_data.txt
```

* démarrer pyspark
```python
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree 
from pyspark.mllib.util import MLUtils
```
* charger les données
```python
data = MLUtils.loadLibSVMFile(sc, '/tmp/sample_libsvm_data.txt').cache()
```
* Appliquer les arbres de décisions
```python
model = DecisionTree.trainClassifier(data, numClasses=2,
categoricalFeaturesInfo={},
impurity='gini', maxDepth=5)
```

* Afficher le modèle
```python
print(model.toDebugString())
```
* Evaluer le résultat
```python
predictions = model.predict(data.map(lambda x: x.features))
predictions.collect()
labelsAndPredictions = data.map(lambda lp: lp.label).zip(predictions)
labelsAndPredictions.collect()
trainErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() /
float(data.count())
print('Training Error = ' + str(trainErr))
print('Learned classification tree model:')
print(model)
```

# TP9
```python
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD,LinearRegressionModel
```
* charger et préparer les données
```python
>>> def parsePoint(line):
...     values = [float(x) for x in line.replace(',', ' ').split(' ')]
...     return LabeledPoint(values[0], values[1:])
data = sc.textFile("/tmp/lpsa.data")
parsedData = data.map(parsePoint)
```
* créer le modèle
```python
model = LinearRegressionWithSGD.train(parsedData, iterations=100)
```

* évaluer le résultat
```python
VP = parsedData.map(lambda p: (p.label, model.predict(p.features)))
MSE = VP.map(lambda (v, p): (v - p)**2) .reduce(lambda x, y: x + y) / data.count()
print("Mean Squared Error = " + str(MSE))
Mean Squared Error = 6.207597210613578
```
* Sauvegarder le modèle
```python
model.save(sc, "/tmp/pythonLinearRegressionWithSGDModel")
OurModel = LinearRegressionModel.load(sc,"/tmp/pythonLinearRegressionWithSGDModel")
```

#  TP 10
```python
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD,LinearRegressionModel
data = sc.textFile("/tmp/fertility.txt")
```

# TP 11

https://spark.apache.org/docs/2.2.0/ml-collaborative-filtering.html
```python
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

lines = spark.read.text("/tmp/movies.txt").rdd
parts = lines.map(lambda row: row.value.split("::"))
ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),
                                     rating=float(p[2]), timestamp=long(p[3])))
ratings = spark.createDataFrame(ratingsRDD)
(training, test) = ratings.randomSplit([0.8, 0.2])

# Build the recommendation model using ALS on the training data
# Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating",
          coldStartStrategy="drop")
		  #als = ALS(maxIter=5, regParam=0.01, implicitPrefs=True,userCol="userId", itemCol="movieId", ratingCol="rating")
model = als.fit(training)

# Evaluate the model by computing the RMSE on the test data
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))
Root-mean-square error = 1.77685441008

# Generate top 10 movie recommendations for each user
userRecs = model.recommendForAllUsers(10)
# Generate top 10 user recommendations for each movie
movieRecs = model.recommendForAllItems(10)
```

# TP 12
https://spark.apache.org/docs/2.1.0/ml-classification-regression.html#binomial-logistic-regression

```python

```
