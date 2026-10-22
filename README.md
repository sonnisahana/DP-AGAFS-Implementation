# DP-AGAFS: Adaptive Feature Selection Using a CUDA-Accelerated Genetic Algorithm for CNN-Based Image Classification

Official code and result files for the paper **"Adaptive Feature Selection Using CUDA-Accelerated Genetic Algorithm for CNN-Based Image Classification"**, presented at the *2026 International Conference on Sustainability, Innovation & Technology (ICSIT 2026)*.

**DP-AGAFS** (Diversity-Preserving Adaptive Genetic Algorithm for Feature Selection) selects a compact, low-redundancy subset of deep CNN features for image classification.

## Highlights

- **Correlation-penalized fitness function** that jointly optimizes accuracy, subset size, and inter-feature redundancy:
  `F(s) = α·Acc(s) − β·(|s|/N) − γ·ρ̄(s)`, with α = 0.75, β = 0.15, γ = 0.10 (α > β + γ keeps accuracy dominant).
- **Entropy-driven adaptive operators**: Shannon entropy of the population controls mutation and crossover rates every generation.
- **GPU-accelerated fitness evaluation**: the k-NN fitness (the dominant cost of the wrapper loop) runs on the GPU with feature matrices resident in device memory, and the correlation matrix is precomputed once on the GPU.
- Evaluated on **CIFAR-10, PathMNIST, STL-10** against Binary PSO (BPSO), over **30 independent runs** per method and dataset.

## Main results (30 runs, mean)

| Dataset   | Method   | Accuracy (%) | # Features (of 1280) |
|-----------|----------|:------------:|:--------------------:|
| CIFAR-10  | BPSO     | 84.26        | 613.2                |
| CIFAR-10  | DP-AGAFS | 83.92        | 458.8                |
| PathMNIST | BPSO     | 83.91        | 609.8                |
| PathMNIST | DP-AGAFS | 83.73        | 444.2                |
| STL-10    | BPSO     | 95.38        | 602.8                |
| STL-10    | DP-AGAFS | 95.00        | 405.1                |

- DP-AGAFS selects **25–33% fewer features** than BPSO; the reduction is statistically significant (Wilcoxon signed-rank, p < 0.001) and holds in 30 of 30 runs on every dataset.
- Accuracy differences versus BPSO are not significant on CIFAR-10 and PathMNIST (by design, since accuracy is the dominant objective). On STL-10, BPSO is slightly more accurate (0.38 points, significant) while DP-AGAFS uses 33% fewer features.
- GPU vs. CPU (NVIDIA RTX A5000): **10.1×** faster correlation precomputation, **11.6×** faster fitness evaluation, with numerically identical predictions. The entropy and adaptive-rate components add about 5.7% per-generation overhead.

Full tables (ablation, classifier generalisation, backbone comparison, sensitivity analysis, overhead, statistics) are in the paper and in the JSON files under `dp_agafs_results/`.

## Repository structure

```
.
├── DP_AGAFS.ipynb          # full pipeline: feature extraction, GA/BPSO, experiments, statistics, figures
├── dp_agafs_results/       # saved results (JSON) and best chromosomes (.npy)
├── dp_agafs_figures/       # generated figures
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

Python 3.9+ and an NVIDIA GPU with CUDA are recommended (the code falls back to CPU, but will be much slower).

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv .venv
# Windows PowerShell:  .venv\Scripts\Activate.ps1
# Linux / macOS:       source .venv/bin/activate
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA version from <https://pytorch.org/get-started/locally/> if the default `pip` wheel does not use your GPU.

CIFAR-10, STL-10 and PathMNIST (via the `medmnist` package) are downloaded automatically on first run.

## Usage

1. Open `DP_AGAFS.ipynb` in Jupyter / VS Code.
2. Use **Kernel → Restart & Run All** for a clean, consistent run (cells depend on execution order).
3. Results are written to `./dp_agafs_results/` and figures to `./dp_agafs_figures/`.

The full 30-run experiments across three datasets are the long-running part; the GPU is used for the correlation matrix and the k-NN fitness evaluation.

## Experimental setup

| Item | Value |
|---|---|
| Feature extractor | MobileNetV2 (ImageNet pre-trained, 1280-d global-average-pool features) |
| Data | Stratified subset: 5,000 train / 1,000 test per dataset |
| Population / generations | 30 / 40 |
| Selection / elitism | Tournament (k = 3) / 3 elites |
| Mutation / crossover rate | 0.01–0.15 / 0.60–0.95 (entropy-adaptive) |
| Fitness classifier | k-NN (k = 5), GPU |
| Reported classifiers | SVM (RBF), k-NN, and others in the generalisation study |
| Hardware used | NVIDIA RTX A5000, Intel Core i7, 64 GB RAM |

## Notes and limitations

- Experiments use 5,000-sample stratified subsets because the wrapper search needs about 36,000 fitness evaluations per configuration. Evaluation on the full datasets is future work.
- Exact numbers can vary slightly across hardware, library versions, and random seeds. A reproducibility run with an independent seed set gave 439.5 ± 31.6 features at 83.80% accuracy on CIFAR-10.
- GPU speedups depend on the GPU; ratios on other hardware (for example a Colab T4) will differ.

## Citation

If you use this code, please cite the paper:

```bibtex
@inproceedings{sonni2026dpagafs,
  title     = {Adaptive Feature Selection Using CUDA-Accelerated Genetic Algorithm for CNN-Based Image Classification},
  author    = {Sonni, Sahana K and H, Vinutha and Kulkarni, Saachi and K, Sanjana and N, Rajashree S},
  booktitle = {2026 International Conference on Sustainability, Innovation \& Technology (ICSIT)},
  year      = {2026},
  note      = {Add DOI / IEEE Xplore link when available}
}
```

## Authors

Sahana K Sonni, Vinutha H, Saachi Kulkarni, Sanjana K, Rajashree S N
Department of Machine Learning, B.M.S. College of Engineering, Bangalore, India

## License

Released under the MIT License (see `LICENSE`).