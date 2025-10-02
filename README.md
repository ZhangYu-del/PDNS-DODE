1. Title
PDNS-DODE: Discrete Osprey Optimization and Differential Evolution Algorithm with PageRank Diffusion Neighborhood Search for Influence Maximization in Social Networks

2. Description
This project provides the source code and experimental framework for the PDNS-DODE algorithm—a discrete hybrid optimizer designed to solve the Influence Maximization (IM) problem in social networks (selecting optimal seed nodes to maximize information spread, applicable to viral marketing, public policy diffusion, etc.). It integrates PageRank Diffusion Neighborhood Search (PDNS), discretized Osprey Optimization Algorithm (OOA) for global exploration, and influence-aware Differential Evolution (DE) for local exploitation. 

3. Dataset Information
Experiments use 7 real-world social network datasets: Email-Eu (European research institution email network), Blog (MSN Spaces blog owner network), Wiki-Vote (Wikipedia admin election voting network), NetHEHT (Arxiv academic collaboration network), CA-AstroPh (Arxiv astrophysics collaboration network), CA-HepTh (Arxiv high-energy physics theory collaboration network), and Brightkite (location-based social network).These datasets are sourced from two websites: https://snap.stanford.edu/data/ and https://www.arxiv.org/. 

5. Code Information
The core framework of the algorithm is also provided in the PDNS-DODE.py file. Core code follows the algorithm’s 4-stage workflow, including: data preprocessing (load datasets, build graphs, compute node PageRank), initialization (PageRank-descending initial population), global exploration (discretized OOA + PDNS node replacement), local exploitation (influence-aware DE with dynamic mutation and fitness-based crossover), local search (3-hop neighborhood seed set optimization), main logic (control iterations, evaluate fitness via Expected Diffusion Value (EDV)).

6. Usage Instructions & Requirements
Programming language: Python 3.8+. Dependencies: networkx (graph operations), numpy (numerical computing), scipy (PageRank calculation), matplotlib (result visualization), pandas (data organization).

7. This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
