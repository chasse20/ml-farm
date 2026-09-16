# Python ML Farm

This is a code sample from a much larger personal machine learning platform I have been developing since 2023 as a hobby.

The larger system treated a group of high core count machines as a shared ML processing farm. PostgreSQL acted as the coordination and state backend, nodes and workers dynamically claimed training jobs, models were converted to ONNX, validated locally, and published to a lightweight internal artifact store.

This repository contains the worker/training side of that system.

- distributed job coordination through tuned PostgreSQL with transactional job claiming
- worker registration and targeted or unassigned job routing
- multiprocessing with explicit CPU affinity to efficiently train in parallel
- per-process BLAS/OpenMP thread budgets to prevent oversubscription
- asynchronous orchestration around parallelized worker processes
- model training across several algorithm families, easily scalable
- ONNX conversion, graph normalization, validation, and inference smoke testing
- retry-aware artifact publishing
- also a special endpoint specifically to aid in Bayesian Process parameter sweeping using random forest statistics
- Dockerized

## Worker

Each instance registers itself as a slave/worker in PostgreSQL and starts a configured number of child processes. Each process receives a fixed CPU set, while this parent process keeps a separate CPU reserve for orchestration and database work.

The worker controllers atomically claim available jobs from PostgreSQL. Multiple deployments can simultaneously compete for work without central scheduling locks or duplicate assignment/race conditions.

CPU training happens outside the asyncio event loop. Work is sent to isolated processes over multiprocessing queues, and otherwise thread-heavy ML libraries are constrained.

```text
                     PostgreSQL
                  jobs / state / data
                         |
             +-----------+-----------+
             |                       |
        Node A                     Node B
        async controller           async controller
             |                       |
       +-----+-----+           +-----+-----+
       |     |     |           |     |     |
     worker worker worker    worker worker worker
       |                       |
   train / sweep           train / sweep
       |                       |
       +------ ONNX export ----+
                  |
           internal model store
```

## Trainers

- LightGBM Binary Classification
- LightGBM Regression
- LightGBM Ranking
- Ridge Regression
- Random Forest Regression
- Linear SVC Classification
- CatBoost Regression
- XGBoost Regression

`Trainer/Service.py` provides the dispatch layer. Trainers share a common data representation so the orchestration layer does not need model-specific logic.

After training, models are converted to ONNX in `FTP/Service.py`. The exported graph is normalized where needed, validated with the ONNX checker, loaded by ONNX Runtime, and executed once with generated input data before the artifact is accepted.

## Sweeper

There is also a separate "sweeper" branch that is useful for training custom Bayesian processes. I chose a Random Forest algorithm and evaluated candidate feature vectors across individual trees. This can easily be expanded to support regular Gaussian Processes.

## ONNX Artifacts

The original implementation used basic SFTP for ONNX artifact movement because the traffic stayed on an isolated private network and the mechanism was intentionally simple (i.e., I have my Unraid server for Postgres/SMB/storage and my god-tier ML server). It can easily be adapted to support SMB or other protocol. For a production environment crossing untrusted networks, I would use an authenticated encrypted artifact service or object store instead.

## What is not included

This was a component of a considerably larger personal project. The public sample intentionally excludes database creation and migration history, datasets, produced models, application-specific feature generation, broader homelab orchestration, infrastructure configuration, and private deployment details. Good luck! You'll have to setup your own SQL database!
