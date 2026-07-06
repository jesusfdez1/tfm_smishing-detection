# MetaSMS-HSS: A Unified Framework for Smishing Detection

This repository hosts a comprehensive ecosystem for advanced SMS threat intelligence. It is structured around two primary pillars:

1. **The MetaSMS-HSS Data Engineering Pipeline**: A robust Extract-Transform-Load (ETL) system that ingests, anonymizes, deduplicates, and standardizes 13 heterogeneous data sources to create **MetaSMS-HSS** (*Meta-dataset for Ham, Spam, and Smishing*), an unprecedented multi-class dataset.
2. **The Multi-Paradigm Experimental Framework**: A modular evaluation environment to comprehensively assess and compare three distinct algorithmic families on the generated data:
   - **Traditional Machine Learning**: Extensive evaluation grids combining classic algorithms (Random Forest, SVM, Logistic Regression, Naive Bayes) with diverse text encoders (BoW, TF-IDF, Word2Vec, FastText).
   - **Deep Learning**: Fine-tuning and evaluation of robust transformer-based architectures (e.g., RoBERTa, Multilingual BERT).
   - **Large Language Models**: Assessment of zero-shot and few-shot detection capabilities using local and API-based generative models.

## 1. Dataset Composition

> **Data Dictionary:** For a detailed explanation of each CSV column, including the taxonomy of intents (`theme`), urgency levels, and ISO language codes, please refer to the **[Data Dictionary](docs/data_dictionary.md)**.
> 
> **Sample Dataset:** A small, anonymized sample of the dataset (30 rows, 10 per class) is available at `data/sample/metasms_sample.csv` for reference without exposing the full dataset.

The meta-dataset is dynamically aggregated from **13 independent, heterogeneous sources** via `build_metadataset.py`, specifically curated to balance classes and maximize lexical diversity while strictly excluding standard emails to ensure domain fidelity:

| # | Dataset | Samples | Classes | Format | Original Source / Citation |
|---|---------|----------|--------|---------|----------------------------|
| 1 | ExAIS SMS Spam | 5,240 | 2 (Spam, Ham) | CSV | [Onashoga et al. (2015)](#6-academic-references) |
| 2 | Mishra & Soni 2022 | 5,971 | 3 (Ham, Spam, Smishing) | CSV | [Mishra & Soni (2022)](#6-academic-references) |
| 3 | Mishra Extended | 10,191 | 3 (Ham, Spam, Smishing) | CSV | [Mishra & Soni (2022)](#6-academic-references) |
| 4 | Hosseinpour 2025 | 84,863 | 3 (Ham, Spam, Smishing) | CSV | [Hosseinpour & Das (2025)](#6-academic-references) |
| 5 | Agarwal IMC 2025 | 33,869 | 1 (Smishing) | CSV | [Agarwal et al. (2025)](#6-academic-references) |
| 6 | UCI SMS Spam | 5,574 | 2 (Ham, Spam) | TSV | [UCI](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) |
| 7 | NUS SMS Corpus | 55,835 | 1 (Ham) | JSON | [Chen & Kan (2013)](#6-academic-references) |
| 8 | Kaggle Spam/Ham | 10,961 | 2 (Spam, Ham) | CSV | [Kumar (Kaggle)](#6-academic-references) |
| 9 | Kaggle Phishing | 1,000 | 1 (Smishing) | CSV | [Tijjani (Kaggle)](#6-academic-references) |
| 10 | Malicious-Benign SMS/MMS | ~383,000 (Non-AI) | 2 (Benign, Spam) | CSV | [HuggingFace](https://huggingface.co/datasets/notd5a/malicious-benign-sms-mms-dataset) |
| 11 | Spanish Spam/Ham | 1,207 | 2 (Spam, Ham) | CSV | [HuggingFace](https://huggingface.co/datasets/softecapps/spam_ham_spanish) |
| 12 | Smishing-4C | 120 | 4 Thematic Cats. | CSV | [Martínez-Mendoza et al. (2024)](#6-academic-references) |
| 13 | MIMICS-3500 | 3,500 | 7 / 13 Classes | CSV | [Martínez-Mendoza et al. (2026)](#6-academic-references) |

> **Note on SmishTank:** The SmishTank dataset is processed via a separate, standalone pipeline (`src/dataset/build_smishtank_standalone.py`) to preserve its specific community verification scores and metadata.


## 2. Architectural Structure

The project is strictly modularized, segregating the Extract-Transform-Load (ETL) data pipeline from the Machine Learning experimental framework:

```text
smishing-detection/
├── data/       # Raw data sources and processed master datasets
├── docs/       # Extended academic documentation and data dictionary
├── results/    # Results, metrics, confusion matrices and generated artifacts
├── scripts/    # Automated shell scripts for task execution
└── src/        # Main project source code
    ├── dataset/       # Data engineering and ETL pipeline
    └── experiments/   # Experimental frameworks
        ├── dl/        # Deep Learning architectures (Transformers)
        ├── llm/       # Large Language Models evaluations
        └── ml/        # Traditional Machine Learning baselines
```

## 3. Usage Guide

### 3.1. Data Engineering (Dataset Construction)

The data pipeline handles structural normalization, language detection (Meta's FastText network), privacy preservation (anonymization), and heuristic deduplication (SHA-256 hash matching). By default, the pipeline also automatically filters out any synthetic or AI-generated messages to prevent synthetic bias in training models.

```bash
# Standard complete build (automatically excludes AI-generated messages)
python src/dataset/build_metadataset.py --output data/processed/metasms_hss_master.csv

# Standard complete build KEEPING AI-generated messages
python src/dataset/build_metadataset.py --keep-ai-generated --output data/processed/metasms_hss_master.csv

# Build featuring advanced PII redaction via privacy-filter
python src/dataset/build_metadataset.py --privacy-filter --output data/processed/metasms_hss_master_privacy.csv

# Enrich dataset with LLM metadata and WHOIS features
python src/dataset/enrich_metadataset.py --input data/processed/metasms_hss_master.csv --output data/processed/metasms_hss_master_enriched.csv
```

### 3.2. Experimental Frameworks

The repository encompasses three core experimental branches evaluated on a stratified 80/10/10 split:

**Traditional Machine Learning (`src/experiments/ml`)**
Executes evaluation grids combining text encoders (BoW, TF-IDF, Word2Vec, FastText) and classifiers (NB, LR, RF, SVM).
```bash
python -m src.experiments.ml.main --data_root data/processed --out_dir results/ml
```

**Deep Learning (`src/experiments/dl`)**
Fine-tunes and evaluates robust transformer architectures (e.g., RoBERTa, BERT).
```bash
python -m src.experiments.dl.main --data_root data/processed --out_dir results/dl
```

**Large Language Models (`src/experiments/llm`)**
Assesses prompt-based detection via local models (e.g., LLaMA, Mistral).
```bash
python -m src.experiments.llm.main --data_root data/processed --out_dir results/llm
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

### Mishra & Soni (2022)
> Mishra, Ritu; Soni, Pinal (2022), **Smishing Dataset**, *Mendeley Data*, V1. [DOI: 10.17632/f45bkkt8pr.1](https://doi.org/10.17632/f45bkkt8pr.1)

### Hosseinpour & Das (2025)
> Hosseinpour, Shaghayegh, and Sanchari Das. (2025). **POSTER: A Multi-Signal Model for Detecting Evasive Smishing**. *18th ACM Conference on Security and Privacy in Wireless and Mobile Networks (WiSec 2025)*, 292-293. [DOI: 10.1145/3734477.3736147](https://doi.org/10.1145/3734477.3736147)

### Agarwal et al. (2025)
> Agarwal, A. et al. (2025). **Fishing for Smishing: Understanding SMS Phishing Infrastructure and Strategies by Mining Public User Reports**. *IMC '25: ACM Internet Measurement Conference*. [DOI: 10.1145/3730567.3764431](https://doi.org/10.1145/3730567.3764431)

### NUS SMS Corpus
> Chen, Tao, and Min-Yen Kan. (2013). **Creating a Live, Public Short Message Service Corpus: The NUS SMS Corpus**. *Language Resources and Evaluation*, 47(2), 299-335. [DOI: 10.1007/s10579-012-9197-9](https://doi.org/10.1007/s10579-012-9197-9)

### Kaggle Spam/Ham Dataset
> Kumar, T. (n.d.). **SMS Spam Dataset**. *Kaggle*. Available at: [https://www.kaggle.com/datasets/tinu10kumar/sms-spam-dataset](https://www.kaggle.com/datasets/tinu10kumar/sms-spam-dataset)

### Kaggle Phishing Dataset
> Tijjani, A. (n.d.). **Phishing Email & SMS Dataset with NLP Categories**. *Kaggle*. Available at: [https://www.kaggle.com/datasets/ahmadtijjani/phishing-email-sms-dataset-with-nlp-categories](https://www.kaggle.com/datasets/ahmadtijjani/phishing-email-sms-dataset-with-nlp-categories)
