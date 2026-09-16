# Python ML Farm

This is a code sample from a larger personal machine-learning platform I developed 2023-2026 for my homelab as a hobby.

The larger system treated a group of high-core-count machines as a shared ML processing farm. PostgreSQL acted as the coordination and state backend, workers dynamically claimed training jobs, models were converted to ONNX, validated locally, and published to a lightweight internal artifact store.

This repository contains the worker/training side of that system. It is intentionally not the complete platform.

- distributed job coordination through tuned PostgreSQL with transactional job claiming
- worker registration and targeted or unassigned job routing
- multiprocessing with explicit CPU affinity
- per-process BLAS/OpenMP thread budgets to prevent oversubscription
- asynchronous orchestration around CPU-bound worker processes
- model training across several algorithm families, easily scalable
- ONNX conversion, graph normalization, validation, and inference smoke testing
- retry-aware artifact publishing
- also a special endpoint specifically to aid in Bayesian Process sweeping using per-tree random-forest statistics
- Dockerized deployment and environment-driven configuration

## Worker model

Each FarmPy instance registers itself as a slave/worker in PostgreSQL and starts a configured number of child processes. Each process receives a fixed CPU set, while the parent process keeps a separate CPU reserve for orchestration and database work.

The worker controllers atomically claim available jobs from PostgreSQL. Technically multiple machines and multiple controllers can compete for work without central scheduling locks or duplicate assignment.

CPU-heavy training happens outside the asyncio event loop. Work is sent to isolated processes over multiprocessing queues, and thread-heavy ML libraries are constrained.

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

The sample retains the trainer implementations that were active in this version of the farm:

- LightGBM binary classification
- LightGBM regression
- LightGBM ranking
- ridge regression
- random-forest regression
- linear SVC classification
- CatBoost regression
- XGBoost regression

`Trainer/Service.py` provides the dispatch layer. Trainers share a common table/row representation so the orchestration layer does not need model-specific logic.

After training, models are converted to ONNX in `FTP/Service.py`. The exported graph is normalized where needed, checked with the ONNX checker, loaded by ONNX Runtime, and executed once with generated input data before the artifact is accepted.

## Sweeper

The ML Farm/FarmPy also supported a separate sweeper branch that I use in practice for a custom Bayesian process. I chose a Random Forest algorith and evaluated candidate feature vectors across individual trees. This can easily be expanded to support regular Gaussian Processes.

## ONNX Artifacts

The original implementation used basic SFTP for ONNX artifact movement because the traffic stayed on an isolated private network and the mechanism was intentionally simple (i.e., I have my Unraid server for Postgres/SMB/storage and my god-tier ML server). It can easily be adapted to support SMB or other protocol. For a production environment crossing untrusted networks, I would use an authenticated encrypted artifact service or object store instead.

## Configuration

`appsettings.json` contains non-sensitive local defaults. Deployment values can be overridden with environment variables using:

```text
FARMPY__<Section>__<Key>
```

For example:

```bash
export FARMPY__Slave__SQLHost=postgres
export FARMPY__Slave__SQLPassword='...'
export FARMPY__FTP__Host=model-store
```

See `.env.example` for the main deployment settings.

## What is not included

This was a component of a considerably larger personal project. The public sample intentionally excludes database creation and migration history, datasets, produced models, application-specific feature generation, broader homelab orchestration, infrastructure configuration, and private deployment details. Good luck! You'll have to setup your own SQL database!

The repository is therefore intended as an architecture and implementation sample rather than a turnkey ML platform.
