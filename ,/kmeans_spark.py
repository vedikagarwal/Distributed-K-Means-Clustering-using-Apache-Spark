from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
import time

# Initialize Spark session
spark = SparkSession.builder.appName("Distributed KMeans Example").getOrCreate()

# Load the dataset
data_path = "/tmp/data/HIGGS.csv"  # make sure this is the correct path
df = spark.read.csv(data_path, inferSchema=True).cache()

# Prepare features
feature_cols = df.columns[1:]  # skip label
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
assembled_data = assembler.transform(df).select("features")

# KMeans
kmeans = KMeans(k=5, seed=1) 

print("Training started...")
start_time = time.time()
model = kmeans.fit(assembled_data)
end_time = time.time()
print("Training completed.")

# Results
print(f"Time taken: {end_time - start_time:.2f} seconds")
print(f"Number of clusters: {kmeans.getK()}")
print("Cluster Centers:")

centers = model.clusterCenters()
for i, center in enumerate(centers):
    print(f"Cluster {i + 1}: {center}")

# Optional: show predictions on a small sample
predictions = model.transform(assembled_data)
print("\nSample Predictions:")
predictions.select("features", "prediction").show(5, truncate=False)

# Stop Spark
spark.stop()

