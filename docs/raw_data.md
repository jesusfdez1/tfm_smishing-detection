# Raw Datasets Documentation (`data/raw/`)

This document provides a comprehensive overview and data dictionary for all the raw datasets ingested into the `tfm_smishing-detection` project. These datasets serve as the foundation for our extraction, loading, and transformation (ELT) pipelines, and ultimately for training our machine learning models. 

To guarantee reproducibility and data lineage, all files located in the `data/raw/` directory are kept strictly **immutable**.

---

## 1. ExAIS SMS Spam Dataset

**Overview:**  
An indigenous SMS Spam corpus compiled at the Federal University of Agriculture, Abeokuta, Nigeria, uniquely featuring an African-English context. It was created to develop a reliable SMS corpus for improving spam detection schemes.

**Key Statistics:**  
- **Total Records:** 5,240 SMS messages  
- **Class Distribution:** 2,350 Spam | 2,890 Ham (legitimate)  
- **Source Population:** 20 participants (students, faculty, and administrative staff, aged 20–50)  
- **Authors/Institution:** Onashoga et al. / Federal University of Agriculture, Abeokuta  

**Storage & Format:**  
- **Location:** `data/raw/ExAIS_SMS Spam Dataset/`  
- **Format:** CSV (20 separate files)  
- **Structure:** One file per participant (`USER 1.csv` to `USER 20.csv`). Message counts per file range from 23 to 1,082 records.

**Data Dictionary:**  
| Attribute | Type | Description |
| :--- | :---: | :--- |
| `Sender Details` / `Contact` | *String* | Masked sender identifier or contact reference. |
| `Date` / `Timestamp` | *Date/Time* | Time of message reception. |
| `SMS Message` | *String* | The raw textual content of the message. |
| `Metadata` | *Mixed* | Additional anonymized context from users' contact lists. |

**Primary Citation & DOI:**  
> Onashoga, A. S., Abayomi-Alli, O. O., Sodiya, A. S., & Ojo, D. A. (2015). An Adaptive and Collaborative Server-Side SMS Spam Filtering Scheme Using Artificial Immune System. *Information Security Journal: A Global Perspective*, 24(4-6), 133-145.  
> **DOI/URL:** Not provided.

**Additional Notes:**  
Explicit consent was obtained from all participants. Sensitive details (e.g., bank accounts, phone numbers) have been strictly masked to preserve anonymity.

---

## 2. Smishtank Collections

**Overview:**  
A continuously updated repository of real-world "smishing" (SMS Phishing) threats obtained from the [SmishTank](https://smishtank.com/) platform. Given the rapidly evolving nature of SMS threats, this dataset provides modern, in-the-wild malicious payloads necessary for robust threat detection.

**Key Statistics:**  
- **Total Records:** Continuously updated (exact count varies by pipeline run)  
- **Class Distribution:** Primarily Smishing/Phishing payloads  
- **Source Population:** Public threat reports & automated web scraping  
- **Authors/Institution:** Timko & Rahman / SmishTank Platform  

**Storage & Format:**  
- **Location:** `data/raw/smishtank_raw.jsonl` (plus supplementary Dataset I files)  
- **Format:** JSONL / CSV  
- **Structure:** Raw message payloads appended programmatically via data ingestion pipelines.

**Data Dictionary:**  
| Attribute | Type | Description |
| :--- | :---: | :--- |
| `text` | *String* | Raw SMS payload content. |
| `label` | *Categorical* | Threat classification (Smishing/Spam). |
| `metadata` | *JSON* | Pipeline-extracted tags, source URLs, and timestamps. |

**Primary Citation & DOI:**  
> Timko, D., & Rahman, M. L. (2024). Smishing Dataset I: Phishing SMS Dataset from Smishtank.com. *Proceedings of the Fourteenth ACM Conference on Data and Application Security and Privacy*, 289–294.  
> Timko, D., & Rahman, M. L. (2023). Commercial Anti-Smishing Tools and Their Comparative Effectiveness Against Modern Threats. *Proceedings of the 16th ACM Conference on Security and Privacy in Wireless and Mobile Networks*, 1–12.  
> **DOI/URL:** https://smishtank.com/

**Additional Notes:**  
Dataset requires periodic pipeline updates to capture emerging threat patterns. Commercial anti-smishing tool comparisons are documented in the referenced literature.

---

## 3. SMS Phishing Dataset for Machine Learning & Pattern Recognition

**Overview:**  
Published on Mendeley Data, this is a comprehensively labeled dataset containing SMS messages curated specifically for deep learning and targeted attribute extraction research. The data was gathered partly by running OCR (Optical Character Recognition) via Python on internet-sourced images of SMS screens.

**Key Statistics:**  
- **Total Records:** 5,971 messages  
- **Class Distribution:** 4,844 Ham | 489 Spam | 638 Smishing  
- **Version:** 1 (June 20, 2022)  
- **Authors/Institution:** Sandhya Mishra & Devpriya Soni / Jaypee Institute of Information Technology  

**Storage & Format:**  
- **Location:** `data/raw/` (original Mendeley package)  
- **Format:** CSV  
- **Structure:** Single tabular file with pre-extracted boolean and textual features.

**Data Dictionary:**  
| Attribute | Type | Description |
| :--- | :---: | :--- |
| `LABEL` | *Categorical* | Ground truth classification: `Ham`, `Spam`, or `Smishing`. |
| `TEXT` | *String* | Raw, unprocessed textual content of the message. |
| `URL` | *Boolean* | Indicates presence of a hyperlinked URL (`True`/`False`). |
| `EMAIL` | *Boolean* | Indicates presence of an email address (`True`/`False`). |
| `PHONE` | *Boolean* | Indicates presence of a phone number (`True`/`False`). |

**Primary Citation & DOI:**  
> Mishra, S., & Soni, D. (2022). SMS Phishing Dataset for Machine Learning and Pattern Recognition. *Mendeley Data*, V1.  
> **DOI/URL:** https://doi.org/10.17632/f45bkkt8pr.1

**Additional Notes:**  
The original source package includes Python extraction scripts and frequency charts. Our pipeline relies primarily on the raw tabular data for consistent feature engineering.

---

## 4. Extended SMS Phishing Dataset (10,191 messages)

**Overview:**  
An extended dataset structurally identical to the Mendeley dataset (Mishra & Soni), expanded to contain 10,191 messages. It maintains the same schema of labeled text messages intended for SMS phishing classification and model generalization.

**Key Statistics:**  
- **Total Records:** 10,191 messages  
- **Class Distribution:** Follows original dataset proportions (exact split not specified)  
- **Version:** Extended release  
- **Authors/Institution:** Extension of Mishra & Soni dataset  

**Storage & Format:**  
- **Location:** `data/raw/mishra_extended_10191.csv`  
- **Format:** CSV  
- **Structure:** Single file, re-formatted for direct pipeline ingestion.

**Data Dictionary:**  
| Attribute | Type | Description |
| :--- | :---: | :--- |
| `LABEL` | *Categorical* | Ground truth classification: `Ham`, `Spam`, or `Smishing`. |
| `TEXT` | *String* | Raw, unprocessed textual content of the message. |
| `URL` | *Boolean* | Indicates presence of a hyperlinked URL (`True`/`False`). |
| `EMAIL` | *Boolean* | Indicates presence of an email address (`True`/`False`). |
| `PHONE` | *Boolean* | Indicates presence of a phone number (`True`/`False`). |

**Primary Citation & DOI:**  
> Derived from the base dataset by Mishra & Soni (2022).  
> **DOI/URL:** https://doi.org/10.17632/f45bkkt8pr.1 (base reference)

**Additional Notes:**  
Re-formatted and expanded to improve training diversity. Uses the exact same extraction attributes as Dataset 3 for seamless feature alignment.

---

## 5. Combined Labeled Smishing Dataset

**Overview:**  
A robust consolidated dataset merged from five distinct public sources. It features an extensive collection of SMS messages labeled for rigorous supervised learning tasks, with unified relabeling based on the prevalence of smishing-related keywords.

**Key Statistics:**  
- **Total Records:** Not specified (multi-source aggregation)  
- **Class Distribution:** Binary differentiation between general Spam and targeted Smishing  
- **Version:** 2025 release  
- **Authors/Institution:** Shaghayegh Hosseinpour  

**Storage & Format:**  
- **Location:** `data/raw/hosseinpour_2025_combined.csv`  
- **Format:** CSV  
- **Structure:** Unified schema with dual-labeling system for spam vs. smishing.

**Data Dictionary:**  
| Attribute | Type | Description |
| :--- | :---: | :--- |
| `message` | *String* | Full SMS payload content. |
| `spam label` | *Binary* | `0` = Ham, `1` = Spam. |
| `smishing label` | *Binary* | `0` = Non-smishing, `1` = Smishing. |

**Primary Citation & DOI:**  
> Hosseinpour, S. (2025). Combined Labeled Smishing Dataset. *ACM Digital Library*.  
> **DOI/URL:** https://doi.org/10.1145/3734477.3736147

**Additional Notes:**  
Specifically engineered to differentiate generic commercial spam from malicious smishing campaigns. Keyword-based relabeling ensures high precision for smishing-focused models.

---

## 6. Fishing for Smishing (IMC 2025)

**Overview:**  
A thoroughly documented dataset containing high-quality, real-world user reports on SMS phishing. This artifact stems from the IMC '25 paper focusing on SMS phishing infrastructure and scammer strategies, built by mining extensive public report repositories.

**Key Statistics:**  
- **Total Records:** Not specified (public report aggregation)  
- **Class Distribution:** Multi-language, multi-tactic smishing campaigns  
- **Version:** IMC 2025 artifact  
- **Authors/Institution:** Sharad Agarwal, Antonis Papasavva, Guillermo Suarez-Tangil, Marie Vasek  

**Storage & Format:**  
- **Location:** `data/raw/agarwal_2025_fishing_smishing.csv`  
- **Format:** CSV  
- **Structure:** Rich metadata schema optimized for NLP, linguistic analysis, and infrastructure tracking.

**Data Dictionary:**  
| Attribute | Type | Description |
| :--- | :---: | :--- |
| `time` | *Date* | Reported date of the message. |
| `text` / `translation` | *String* | Original message payload and its English translation. |
| `language` | *String* | Detected source language (e.g., Portuguese, Dutch, Spanish). |
| `scam_type` / `lure_principles` | *String* | Social engineering tactic category (e.g., delivery, urgency, financial). |

**Primary Citation & DOI:**  
> Agarwal, S., Papasavva, A., Suarez-Tangil, G., & Vasek, M. (2025). Fishing for Smishing: Understanding SMS Phishing Infrastructure and Strategies by Mining Public User Reports. *Proceedings of the 2025 ACM on Internet Measurement Conference*.
> **DOI/URL:** https://doi.org/10.1145/3730567.3764431

**Additional Notes:**  
This dataset is heavily focused on the social engineering aspects and the infrastructure of the attacks. Very useful for feature engineering related to attack types.

---

## 7. UCI SMS Spam Collection

**Overview:**  
The foundational benchmark dataset for SMS spam research. Curated by Almeida et al. from various sources including Grumbletext and the NUS SMS Corpus.

**Key Statistics:**  
- **Total Records:** 5,574 messages  
- **Class Distribution:** 4,827 Ham | 747 Spam  
- **Version:** v.1 (2011)  
- **Authors/Institution:** Tiago A. Almeida & José María Gómez Hidalgo  

**Storage & Format:**  
- **Location:** `data/raw/almeida_2011_uci_sms_spam/`  
- **Format:** TSV (tab-separated values in `SMSSpamCollection` file)  
- **Structure:** Two columns: label (`ham` or `spam`) and raw text.

**Data Dictionary:**  
| Attribute | Type | Description |
| :--- | :---: | :--- |
| `label` | *Categorical* | Ground truth classification: `ham` or `spam`. |
| `text` | *String* | Raw textual content of the message. |

**Primary Citation & DOI:**  
> Almeida, T.A., Gómez Hidalgo, J.M., Yamakami, A. (2011). Contributions to the Study of SMS Spam Filtering: New Collection and Results. *Proceedings of the 2011 ACM Symposium on Document Engineering (DOCENG'11)*.  
> **DOI/URL:** https://archive.ics.uci.edu/dataset/228/sms+spam+collection

---

## 8. NUS SMS Corpus

**Overview:**  
A large corpus of legitimate conversational SMS messages collected by the National University of Singapore, largely from students.

**Key Statistics:**  
- **Total Records:** ~67,000 messages  
- **Class Distribution:** 100% Ham (Legitimate)  
- **Version:** 1.2 (2015)  
- **Authors/Institution:** Tao Chen & Min-Yen Kan / National University of Singapore  

**Storage & Format:**  
- **Location:** `data/raw/nus_2015_sms_corpus_en.json`  
- **Format:** JSON  
- **Structure:** Deeply nested JSON structure containing message details, sender metadata, and timestamps.

**Data Dictionary:**  
| Attribute | Type | Description |
| :--- | :---: | :--- |
| `text` | *String* | Raw textual content of the message. |
| `source` | *JSON* | Metadata about the sender (anonymized phone model, etc.). |

**Primary Citation & DOI:**  
> Chen, T. and Kan, M.-Y. (2013). Creating a Live, Public Short Message Service Corpus: The NUS SMS Corpus. *Language Resources and Evaluation*, 47(2), 299-355.  
> **DOI/URL:** https://doi.org/10.1007/s10579-012-9197-9

---

## 9. Kaggle Combined Spam/Ham Dataset

**Overview:**  
An aggregation of SMS spam and ham messages sourced from Kaggle. Likely a derivative work combining datasets like the UCI SMS Spam Collection with other sources.

**Key Statistics:**  
- **Total Records:** ~10,800 messages  
- **Class Distribution:** Spam and Ham  
- **Authors/Institution:** Unknown (Kaggle Community)  

**Storage & Format:**  
- **Location:** `data/raw/kaggle_unknown_combined_spam_ham.csv`  
- **Format:** CSV  
- **Structure:** Simple tabular data.

**Data Dictionary:**  
| Attribute | Type | Description |
| :--- | :---: | :--- |
| `target` | *Categorical* | Ground truth classification: `spam` or `ham`. |
| `text` | *String* | Message content. |

---

## 10. Kaggle Phishing Dataset with Categories

**Overview:**  
A small, specialized dataset categorizing phishing/smishing messages by the social engineering tactic used (e.g., urgency, authority).

**Key Statistics:**  
- **Total Records:** 1,001 messages  
- **Class Distribution:** Phishing  
- **Authors/Institution:** Unknown (Kaggle Community)  

**Storage & Format:**  
- **Location:** `data/raw/kaggle_unknown_phishing_categories.csv`  
- **Format:** CSV  
- **Structure:** Text and tactical category labels.

**Data Dictionary:**  
| Attribute | Type | Description |
| :--- | :---: | :--- |
| `text` | *String* | Message content. |
| `category` | *Categorical* | Tactic used (e.g., urgency, authority). |
| `label` | *Categorical* | Base classification (`phishing`). |

---

## 11. Enron Spam Dataset

**Overview:**  
A massive collection of spam and ham *emails* from the Enron corpus. While not SMS, the textual patterns are often used in generic spam classification models.

**Key Statistics:**  
- **Total Records:** 33,716 emails  
- **Class Distribution:** 17,171 Spam | 16,545 Ham  
- **Version:** 2006 (Metsis et al.)  
- **Authors/Institution:** V. Metsis, I. Androutsopoulos, and G. Paliouras  

**Storage & Format:**  
- **Location:** `data/raw/metsis_2006_enron_spam.csv`  
- **Format:** CSV  
- **Structure:** Emails organized into subject, message body, label, and date.

**Data Dictionary:**  
| Attribute | Type | Description |
| :--- | :---: | :--- |
| `Subject` | *String* | Email subject line. |
| `Message` | *String* | Email body content. |
| `Spam/Ham` | *Categorical* | Label: `spam` or `ham`. |
| `Date` | *Date* | Arrival date. |

**Primary Citation & DOI:**  
> Metsis, V., Androutsopoulos, I. and Paliouras, G. (2006). Spam Filtering with Naive Bayes - Which Naive Bayes? *CEAS*.  

---

## 12. Malicious-Benign SMS/MMS Dataset

**Overview:**  
A large-scale, highly engineered dataset designed for DeBERTa training. It combines real data (Discord messages, UCI, SpamDam) with heavily generated synthetic data using various LLMs to create balanced smishing classes.

**Key Statistics:**  
- **Total Records:** Up to 813,546 rows (depends on specific CSV used)  
- **Class Distribution:** Highly structured (base: ~365K Benign, ~76K Spam)  
- **Version:** v3  
- **Authors/Institution:** Unknown (HuggingFace Community Source)  

**Storage & Format:**  
- **Location:** `data/raw/malicious_benign_sms_mms/`  
- **Format:** Directory containing multiple CSVs and synthetic generation scripts.  
- **Structure:** Includes base text features, labels, and pre-computed engineered features (e.g., character entropies).

**Data Dictionary:**  
| Attribute | Type | Description |
| :--- | :---: | :--- |
| `message` | *String* | Full payload content. |
| `label` | *Binary* | `0` = Benign, `1` = Spam/Smishing. |
| `ai_generated` | *Binary* | `0` = Human-written, `1` = AI-generated. |
| `[features]` | *Various* | Dozens of engineered NLP features (e.g., `char_count`, `urgency_score`). |

---

## 13. Spanish Spam/Ham Dataset

**Overview:**  
A small dataset specifically containing messages in Spanish. Rare resource useful for multi-lingual model tuning.

**Key Statistics:**  
- **Total Records:** 1,209 messages (999 train + 210 test)  
- **Class Distribution:** Spam and Ham  
- **Authors/Institution:** Unknown (HuggingFace Community Source)  

**Storage & Format:**  
- **Location:** `data/raw/spanish_spam_ham/`  
- **Format:** Directory with `train.csv` and `test.csv`  
- **Structure:** Basic text and label layout.

**Data Dictionary:**  
| Attribute | Type | Description |
| :--- | :---: | :--- |
| `mensaje` | *String* | Message content in Spanish. |
| `tipo` | *Categorical* | Label: `spam` or `ham`. |