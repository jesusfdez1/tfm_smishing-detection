# MetaSMS-HSS: A Unified Framework for Smishing Detection

> **Master's Thesis Project** — Smishing Detection using Machine Learning, Deep Learning, and Large Language Models (LLMs).

This repository contains the complete experimental framework and data engineering pipeline for **MetaSMS-HSS** (*Meta-dataset for Ham, Spam, and Smishing*). It facilitates the ingestion of highly heterogeneous sources into a unified, traceable, and anonymized meta-dataset, alongside a robust machine learning pipeline for experimental validation.

## 1. Dataset Composition

> **Data Dictionary:** For a detailed explanation of each CSV column, including the taxonomy of intents (`theme`), urgency levels, and ISO language codes, please refer to the **[Data Dictionary](docs/data_dictionary.md)**.

The meta-dataset is dynamically aggregated from **13 independent, heterogeneous sources** via `build_metadataset.py`, specifically curated to balance classes and maximize lexical diversity while strictly excluding standard emails to ensure domain fidelity:

| # | Dataset | Samples | Classes | Format | Original Source / Citation |
|---|---------|----------|--------|---------|----------------------------|
| 1 | ExAIS SMS Spam | 5,240 | 2 (Spam, Ham) | CSV | [Onashoga et al. (2015)](#6-academic-references) |
| 2 | Mishra & Soni 2022 | 5,971 | 3 (Ham, Spam, Smishing) | CSV | [Mendeley](https://doi.org/10.17632/f45bkkt8pr.1) |
| 3 | Mishra Extended | 10,191 | 3 (Ham, Spam, Smishing) | CSV | [Mendeley](https://doi.org/10.17632/f45bkkt8pr.1) |
| 4 | Hosseinpour 2025 | — | 3 (Ham, Spam, Smishing) | CSV | [ACM DL](https://doi.org/10.1145/3734477.3736147) |
| 5 | Agarwal IMC 2025 | — | 1 (Smishing) | CSV | [ACM DL](https://doi.org/10.1145/3730567.3764431) |
| 6 | UCI SMS Spam | 5,574 | 2 (Ham, Spam) | TSV | [UCI](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) |
| 7 | NUS SMS Corpus | ~67,000 | 1 (Ham) | JSON | [DOI](https://doi.org/10.1007/s10579-012-9197-9) |
| 8 | Kaggle Spam/Ham | ~10,800 | 2 (Spam, Ham) | CSV | Kaggle |
| 9 | Kaggle Phishing | 1,001 | 1 (Smishing) | CSV | Kaggle |
| 10 | Malicious-Benign SMS/MMS | ~383,000 (Non-AI) | 2 (Benign, Spam) | CSV | [HuggingFace](https://huggingface.co/datasets/notd5a/malicious-benign-sms-mms-dataset) |
| 11 | Spanish Spam/Ham | 1,209 | 2 (Spam, Ham) | CSV | HuggingFace |
| 12 | Smishing-4C | 120 | 4 Thematic Cats. | CSV | [Martínez-Mendoza et al. (2024)](#6-academic-references) |
| 13 | MIMICS-3500 | 3,500 | 7 / 13 Classes | CSV | [Martínez-Mendoza et al. (2026)](#6-academic-references) |

> **Note on SmishTank:** The SmishTank dataset is processed via a separate, standalone pipeline (`src/dataset/build_smishtank_standalone.py`) to preserve its specific community verification scores and metadata.
>
> **Note on Dataset #10:** The *Malicious-Benign SMS/MMS* dataset is sourced from HuggingFace at [https://huggingface.co/datasets/notd5a/malicious-benign-sms-mms-dataset](https://huggingface.co/datasets/notd5a/malicious-benign-sms-mms-dataset). Only `dataset_v3_undersampled_stratified_full.csv` is required for building the MetaSMS dataset. All LLM-generated synthetic data has been strictly excluded from this pipeline to maintain data purity and avoid synthetic bias.

## 2. Architectural Structure

The project is strictly modularized, segregating the Extract-Transform-Load (ETL) data pipeline from the Machine Learning experimental framework:

```text
tfm_smishing-detection/
├── data/
│   ├── raw/                  # Immutable raw data sources
│   └── processed/            # Master CSV outputs (MetaSMS-HSS)
├── experiments/
│   └── results/              # ML artifacts (weights, metrics, confusion matrices)
├── src/
│   ├── dataset/              # Data Engineering & ETL Pipeline
│   │   ├── build_metadataset.py # Master dataset aggregation script
│   │   ├── enrich_metadataset.py # LLM enrichment and feature extraction
│   │   ├── create_subset.py  # CLI utility for stratified subset creation
│   │   └── utils/            # Core ETL modules (anonymization, LLM, deduplication)
│   └── experiments/          # ML Experimental Framework
│       ├── train_model.py    # Training script (Baseline & Transformer support)
│       └── evaluate.py       # Evaluation and visualization utilities
├── docs/                     # Extended academic documentation
│   ├── data_dictionary.md    # Master Data Dictionary and Categories
│   └── experimentos.md       # Experiments tracking and commands
├── requirements.txt
└── README.md
```

## 3. Usage Guide

### 3.1. Data Engineering (Dataset Construction)

The data pipeline handles structural normalization, language detection (Meta's FastText network), privacy preservation (anonymization), and heuristic deduplication (SHA-256 hash matching).

```bash
# Standard complete build
python src/dataset/build_metadataset.py --output data/processed/metasms_hss_master.csv

# Build featuring advanced PII redaction via privacy-filter
python src/dataset/build_metadataset.py --privacy-filter --output data/processed/metasms_hss_master_privacy.csv

# Enrich dataset with LLM metadata and WHOIS features
python src/dataset/enrich_metadataset.py --input data/processed/metasms_hss_master.csv --output data/processed/metasms_hss_master_enriched.csv
```

### 3.2. Machine Learning Framework (Experimentation)

The ML framework is designed to execute a comprehensive evaluation grid combining multiple text encoders (BoW, TF-IDF, Word2Vec, FastText, MiniLM) and traditional classifiers (Naive Bayes, Logistic Regression, Random Forest, XGBoost, SVM) on the consolidated MetaSMS dataset using a stratified 80/10/10 split.

```bash
# Execute the full machine learning evaluation grid
python -m src.experiments.ml.main --data_root data/processed --out_dir output/ml

# Execute the grid on specific datasets with specific models
python -m src.experiments.ml.main --datasets metasms --encoders tfidf fasttext --classifiers logreg rf
```

## 4. Installation & Requirements

Ensure you are operating within a virtual environment before installing the project dependencies:

```bash
pip install -r requirements.txt
```
*Note: Language detection relies on Meta's FastText (via `fasttext-wheel`), which automatically retrieves `lid.176.bin` upon first execution.*

## 5. License & Academic Integrity

For academic and research use only. The source code provided in this repository is part of a Master's Thesis. 
Consult the individual licenses of each constituent dataset located in `docs/raw_data.md` prior to commercial utilization.

## 6. Academic References

This project utilizes and incorporates the following novel multi-class datasets for smishing detection. If you use the **Smishing-4C** or **MIMICS-3500** datasets in your research, please cite the corresponding publications:

### Smishing-4C Dataset
> Martínez-Mendoza, A., Jáñez-Martino, F., Carofilis, A., Fernández-Robles, L., Alegre, E., & Fidalgo, E. (2024). **Towards Multi-Class Smishing Detection: A Novel Feature Vector Approach and the Smishing-4C Dataset**. *SEPLN-2024: 40th Conference of the Spanish Society for Natural Language Processing*. CEUR Workshop Proceedings.

### MIMICS-3500 Dataset
> Martínez-Mendoza, A., Fidalgo, E., Alegre, E., & Fernández-Robles, L. (2026). **Building a multi-class Short Message Service dataset for smishing detection using agglomerative clustering and dataset fusion**. *Engineering Applications of Artificial Intelligence*, 163(1), 112864. [DOI: 10.1016/j.engappai.2025.112864](https://doi.org/10.1016/j.engappai.2025.112864)

### ExAIS SMS Spam Dataset
> Onashoga, A. S., Abayomi-Alli, O. O., Sodiya, A. S., & Ojo, D. A. (2015). **An Adaptive and Collaborative Server-Side SMS Spam Filtering Scheme Using Artificial Immune System**. *Information Security Journal: A Global Perspective*, 24(4-6), 133-145. [GitHub Repository](https://github.com/AbayomiAlli/SMS-Spam-Dataset)