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
```
nano kmeans_data.txt
0.0 0.0 0.0
0.1 0.1 0.1
0.2 0.2 0.2
9.0 9.0 9.0
9.1 9.1 9.1
9.2 9.2 9.2 
```

* mettre le fichier dans hdfs :
```
hdfs dfs -copyFromLocal /home/spark /tmp/kmeans_data.txt
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

*préparer les données
```
splitedData = data.map(lambda line: array([float(x) for x in line.split(' ')]))
splitedData.take(6)
[array([0., 0., 0.]), array([0.1, 0.1, 0.1]), array([0.2, 0.2, 0.2]), array([9., 9., 9.]), array([9.1, 9.1, 9.1]), array([9.2, 9.2, 9.2])]
```
* créer le modèle
```
clusters = KMeans.train(splitedData , 3, maxIterations=10)
```
* afficher les centres des clusters
```
print(clusters.clusterCenters)
```


# en local
```
from pyspark import SparkContext
sc = SparkContext("local", "App Name", pyFiles=['MyFile.py', 'lib.zip', 'app.egg'])
```