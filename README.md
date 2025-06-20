# Distributed-K-Means-Clustering-using-Apache-Spark

A comprehensive implementation and analysis of distributed K-means clustering using Apache Spark in a Docker-based multi-container environment, demonstrating the complexities and trade-offs of distributed computing at scale

 ## Project Overview
This project implements and analyzes distributed K-means clustering on the massive HIGGS dataset (11 million samples, 28 features) using Apache Spark across different cluster configurations. The experiment reveals critical insights about the non-linear relationship between computational resources and performance in distributed systems.

## Key Results

- Optimal Performance: 2-worker configuration achieved 39.6% improvement (98.84s vs 163.18s baseline)
- Diminishing Returns: 5-worker setup performed nearly identical to single-worker (164.22s vs 163.18s)
- Algorithmic Consistency: All configurations produced identical cluster centers
- Real-world Validation: Demonstrated Amdahl's Law and distributed computing trade-offs

## Architecture

### Infrastructure Setup

- Framework: Apache Spark 3.4 (Bitnami Docker Images)
- Orchestration: Docker Compose
- Algorithm: K-Means Clustering via PySpark MLlib
- Dataset: HIGGS Dataset (7.5GB, 11M records, 28 features)

### Cluster Configurations Tested

- Single Worker: 1 master + 1 worker (11 cores, 6.7 GiB)
- Two Workers: 1 master + 2 workers (22 cores, 13.4 GiB)
- Five Workers: 1 master + 5 workers (55 cores, 33.5 GiB)

## Quick Start

### Prerequisites

- Docker installed
- At least 8GB RAM available

### Running the Experiment

1. Clone the repository
```bash
git clone https://github.com/vedikagarwal/Distributed-K-Means-Clustering-using-Apache-Spark.git
cd Distributed-K-Means-Clustering-using-Apache-Spark
```

2. Download the HIGGS dataset
```bash
 wget https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz
gunzip HIGGS.csv.gz
mv HIGGS.csv ./data/
```

3. Start the Spark cluster For single worker
```bash
docker compose up -d  # For single worker

docker compose up -d --scale spark-worker=2  # For multiple workers, modify docker-compose.yml and scale the worker service
```

4. Submit the K-means job

```bash
docker exec spark-master spark-submit \
  --master spark://spark-master:7077 \
  --executor-memory 1g \
  --total-executor-cores 11 \
  /tmp/app/kmeans_spark.py
```

### Monitor via Spark UI

- Open http://localhost:8080 to view cluster status
- Monitor job progress and resource utilization
