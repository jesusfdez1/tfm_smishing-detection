# tfm_smishing-detection

Trabajo de Fin de Máster — Detección de Smishing mediante Machine Learning, Deep Learning y LLMs.

Este repositorio contiene el pipeline completo de construcción del meta-dataset **MetaSMS-HSS**, desde la ingesta de fuentes heterogéneas hasta la generación del dataset unificado con anonimización, detección de idioma, deduplicación y enriquecimiento opcional mediante LLMs.

## Datasets

El meta-dataset se construye a partir de **15 fuentes heterogéneas**:

| # | Dataset | Muestras | Clases | Formato | Fuente |
|---|---------|----------|--------|---------|--------|
| 1 | ExAIS SMS Spam | 5,240 | 2 (Spam, Ham) | CSV (20 archivos) | Onashoga et al. (2015) |
| 2 | SmishTank | ~10,000+ | 1 (Smishing) | JSONL | [smishtank.com](https://smishtank.com/) |
| 3 | Mishra & Soni 2022 | 5,971 | 3 (Ham, Spam, Smishing) | CSV | [Mendeley](https://doi.org/10.17632/f45bkkt8pr.1) |
| 4 | Mishra Extended | 10,191 | 3 (Ham, Spam, Smishing) | CSV | [Mendeley](https://doi.org/10.17632/f45bkkt8pr.1) |
| 5 | Hosseinpour 2025 | — | 3 (Ham, Spam, Smishing) | CSV | [ACM DL](https://doi.org/10.1145/3734477.3736147) |
| 6 | Agarwal IMC 2025 | — | 1 (Smishing) | CSV | [ACM DL](https://doi.org/10.1145/3730567.3764431) |
| 7 | UCI SMS Spam | 5,574 | 2 (Ham, Spam) | TSV | [UCI](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) |
| 8 | NUS SMS Corpus | ~67,000 | 1 (Ham) | JSON | [DOI](https://doi.org/10.1007/s10579-012-9197-9) |
| 9 | Kaggle Spam/Ham | ~10,800 | 2 (Spam, Ham) | CSV | Kaggle |
| 10 | Kaggle Phishing | 1,001 | 1 (Smishing) | CSV | Kaggle |
| 11 | Enron Spam | 33,716 | 2 (Spam, Ham) | CSV | Metsis et al. (2006) |
| 12 | Malicious-Benign SMS/MMS | ~813,000 | 2 (Benign, Spam) | CSV | HuggingFace |
| 13 | Spanish Spam/Ham | 1,209 | 2 (Spam, Ham) | CSV | HuggingFace |
| 14 | **Smishing-4C** | **120** | **4** (Bank/Finance, Dating, Rewards, SMS Service) | CSV | [Kaggle](https://www.kaggle.com/datasets/galactus007/sms-smishing-collection-data-set), [Mendeley](https://data.mendeley.com/datasets/f45bkkt8pr/1) |
| 15 | **MIMICS-3500** | **3,500** | **7 / 13** clases | CSV | Kaggle, [Mendeley](https://data.mendeley.com/datasets/f45bkkt8pr/1), INCIBE, [SmishTank](https://smishtank.com/), SpamHunter |

### Detalle de nuevos datasets

**Smishing-4C** — 120 muestras de smishing con 4 categorías temáticas. Incluye anotaciones de features: `SLANG`, `COMPANY`, `Length_value`, `Num_writing_errors`, `Phone`, `URL`.

**MIMICS-3500** — 3,500 muestras de smishing con doble etiquetado:
- **7 clases:** Accounts, Bank/Finance, Dating, Deliveries, Rewards, SMS Service, Lifestyle
- **13 clases:** Accounts, Bank, Customer service, Dating, Deliveries, Finances, Gifts, Offers, Other, Prizes, SMS service, Sexual, Spam
- **Fuentes:** Kaggle, Mendeley, INCIBE (Twitter y Facebook), SmishTank, SpamHunter

## Estructura del proyecto

```
tfm_smishing-detection/
├── data/
│   ├── raw/                  # Fuentes originales (inmutables)
│   └── processed/            # Meta-dataset final (MetaSMS-HSS)
├── src/
│   ├── build_metasms.py      # Pipeline principal de construcción
│   └── utils/
│       ├── anonymizer.py     # Anonimización de PII (regex + privacy-filter)
│       ├── cleaner.py        # Limpieza de ofuscación en SMS
│       ├── feature_extractor.py  # Extracción de features de URLs
│       ├── llm_enricher.py   # Enriquecimiento con Gemini API
│       └── smishtank.py      # Extractor de SmishTank API
├── docs/
│   ├── raw_data.md           # Documentación detallada de datasets
│   └── metasms_pipeline.md   # Documentación del pipeline
├── scripts/                  # Scripts de ejecución
├── requirements.txt
└── README.md
```

## Uso

### Construcción del meta-dataset

```bash
# Ejecución básica (sin LLM, con deduplicación)
python src/build_metasms.py --no-llm

# Solo las nuevas fuentes
python src/build_metasms.py --sources "smishing_4c,mimics_3500" --no-llm

# Con enriquecimiento LLM multi-modelo
python src/build_metasms.py --llm-models "gemini-2.5-flash,gemini-2.0-pro"

# Con privacy-filter para anonimización avanzada
python src/build_metasms.py --privacy-filter

# Reanudar una ejecución parcial
python src/build_metasms.py --resume
```

### Fuentes de datos

- **Kaggle Smishing:** https://www.kaggle.com/datasets/galactus007/sms-smishing-collection-data-set
- **Mendeley (Mishra & Soni):** https://doi.org/10.17632/f45bkkt8pr.1
- **SmishTank:** https://smishtank.com/
- **UCI SMS Spam Collection:** https://archive.ics.uci.edu/dataset/228/sms+spam+collection
- **Hosseinpour 2025:** https://doi.org/10.1145/3734477.3736147
- **Agarwal IMC 2025:** https://doi.org/10.1145/3730567.3764431
- **NUS SMS Corpus:** https://doi.org/10.1007/s10579-012-9197-9
- **INCIBE:** https://www.incibe.es/ (Twitter y Facebook)
- **SpamHunter:** Herramienta de recopilación de spam

## Requisitos

```bash
pip install -r requirements.txt
```

## Licencia

Uso académico. Consultar licencia individual de cada dataset en `docs/raw_data.md`.