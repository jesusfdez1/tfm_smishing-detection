# MetaSMS-HSS Dataset Build Pipeline

This document describes how the MetaSMS-HSS master dataset is built from heterogeneous public sources. It covers the transformation steps, labeling strategy, anonymization, deduplication, LLM enrichment, and how to run the pipeline in a reproducible way.

## Scope

The build script produces a single master CSV that includes:
- Unified schema across all sources
- Traceability fields (source, original label, mapping rule)
- Language detection
- Optional LLM enrichment
- Optional PII anonymization with a privacy model
- Cross-source deduplication
- Resume checkpoints for long runs

## Input sources

Raw datasets live under data/raw and include CSV, TSV, JSON, and JSONL formats. Each source is mapped to the unified schema in build_metadataset.py using a source-specific loader.

## Transformation steps

The pipeline applies the following steps in order:

1. Text cleaning and anonymization
   - Clean obfuscation artifacts.
   - Anonymize PII. This can use regex patterns or the privacy-filter model.
   - The anonymized text is used for language detection, dedupe, and LLM enrichment.

2. Label normalization
   - Map source labels to the canonical classes: ham, spam, smishing.
   - For unlabeled rows, optional LLM labeling can be enabled.

3. Language detection
   - Uses Meta's FastText neural network model (`lid.176.bin`) to infer language at high speed.
   - Retries using lowercase text in case of noisy/ALL-CAPS SMS messages.
   - Maps language codes to ISO 639-1 using `pycountry`.

4. LLM enrichment y Whois (Opcional - Fase 2)
   - Realizado por `enrich_metadataset.py`.
   - Añade `theme`, `urgency_level` y opcionalmente deduce la etiqueta.
   - Extrae dominios de las URLs y consulta Whois (edad, país, privacidad).

5. Deduplication
   - Hashes normalized anonymized text.
   - Removes duplicates across all sources.

6. Resume checkpoints
   - Writes a state file and a hash file.
   - Allows safe resume after cancellation.

## Output columns

The master CSV includes the original schema plus these extra fields:
- text_anonymized
- anonymization_status (raw, anonymized, pre_anonymized)
- whois_domain, whois_tld, whois_age_days, whois_hidden, whois_country (Añadidos en fase 2)
- theme, urgency_level, llm_model, prompt_version (Añadidos en fase 2)

## Running the build

**Fase 1: Construcción (Sin LLM)**
Basic run:
```bash
python src/dataset/build_metadataset.py
```

Enable privacy-filter anonymization:
```bash
python src/dataset/build_metadataset.py --privacy-filter --privacy-filter-model openai/privacy-filter
```

Resume a partial run:
```bash
python src/dataset/build_metadataset.py --resume
```

Disable dedupe:
```bash
python src/dataset/build_metadataset.py --no-dedupe
```

**Fase 2: Enriquecimiento con LLM y Whois**
```bash
python src/dataset/enrich_metadataset.py --input data/processed/metasms_v1.csv --output data/processed/metasms_v1_enriched.csv --model Qwen/Qwen3.5-35B-A3B-GPTQ-Int4
```

## Output naming

If `--output` is omitted in Phase 1, the filename encodes the main options, for example:

`metasms_v1_privacy.csv` o `metasms_v1_nodedupe_with_ai.csv`

## Notes on privacy-filter

The privacy-filter model is stronger than regex for contextual PII detection but requires transformers and torch. It should be evaluated on in-domain data and used with clear policies for masking and review.
