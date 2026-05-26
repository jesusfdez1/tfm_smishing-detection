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

Raw datasets live under data/raw and include CSV, TSV, JSON, and JSONL formats. Each source is mapped to the unified schema in build_metasms.py using a source-specific loader.

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
   - Uses specialized libraries (pycld3, langdetect, langid).
   - Falls back to dataset hints if the detector cannot decide.

4. LLM enrichment (optional)
   - Adds theme and urgency_level.
   - Supports multiple LLM models in a single run.
   - Full multi-model outputs are stored in llm_annotations.

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
- llm_models
- llm_annotations

## Running the build

Basic run:

python src/build_metasms.py

Multi-LLM run:

python src/build_metasms.py --llm-models "gemini-2.5-flash,gemini-2.0-pro"

Enable privacy-filter anonymization:

python src/build_metasms.py --privacy-filter --privacy-filter-model openai/privacy-filter

Resume a partial run:

python src/build_metasms.py --resume

Disable dedupe:

python src/build_metasms.py --no-dedupe

## Output naming

If --output is omitted, the filename encodes the main options, for example:

metasms_hss_master__llm=on__label=off__models=gemini-2-5-flash__privacy=on__pmodel=openai-privacy-filter__dedupe=on.csv

## Environment and API keys

LLM keys are loaded from a .env file:

GEMINI_API_KEYS=key_1,key_2,key_3

A single key is also supported via GEMINI_API_KEY.

## Notes on privacy-filter

The privacy-filter model is stronger than regex for contextual PII detection but requires transformers and torch. It should be evaluated on in-domain data and used with clear policies for masking and review.
