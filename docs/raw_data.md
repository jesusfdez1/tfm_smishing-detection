# Raw Datasets Documentation (`data/raw/`)

This document provides a comprehensive overview and data dictionary for all the raw datasets ingested into the `tfm_smishing-detection` project. These datasets serve as the foundation for our extraction, loading, and transformation (ELT) pipelines, and ultimately for training our machine learning models. 

To guarantee reproducibility and data lineage, all files located in the `data/raw/` directory are kept strictly **immutable**.

---

## 1. ExAIS SMS Spam Dataset

**Overview:**
An indigenous SMS Spam corpus compiled at the Federal University of Agriculture, Abeokuta, Nigeria, uniquely featuring an African-English context. It was created to develop a reliable SMS corpus for improving spam detection schemes.

*   **Source Population:** Collected from 20 participants (students, faculty, and administrative staff, aged 20-50).
*   **Total Records:** 5,240 SMS messages.
*   **Class Distribution:** 2,350 Spam | 2,890 Ham (legitimate).
*   **Privacy & Ethics:** Explicit consent was obtained. Sensitive details (e.g., bank accounts, phone numbers) have been strictly masked to preserve anonymity.

**Storage Structure:**
The data is segregated into 20 distinct CSV files inside the `raw/ExAIS_SMS Spam Dataset/` folder, corresponding to each participant (e.g., `USER 1.csv` ... `USER 20.csv`). Message counts per file range dynamically between 23 and 1,082 records.

**Standard Features (Columns):**
While exact column names may vary slightly across the CSVs, they generally contain:
*   `Sender Details` / `Contact`
*   `Date` / `Timestamp`
*   `SMS Message` (The raw texted content)
*   *Metadata related to the users' masked contact lists.*

**Primary Citation:**
> Onashoga, A. S., Abayomi-Alli, O. O., Sodiya, A. S., & Ojo, D. A. (2015). An Adaptive and Collaborative Server-Side SMS Spam Filtering Scheme Using Artificial Immune System. *Information Security Journal: A Global Perspective*, 24(4-6), 133-145.

---

## 2. Smishtank Collections

**Overview:**
A continuously updated repository of real-world "smishing" (SMS Phishing) threats obtained from the [SmishTank](https://smishtank.com/) platform. Given the rapidly evolving nature of SMS threats, this dataset provides modern, in-the-wild malicious payloads necessary for robust threat detection.

Currently, this includes **Dataset I** and data gathered programmatically via our data pipelines (`smishtank_raw.jsonl`).

**Primary Citations:**
Please ensure proper academic attribution when processing this specific data:

> **Smishing Dataset I:**
> Timko, D., & Rahman, M. L. (2024). Smishing Dataset I: Phishing SMS Dataset from Smishtank.com. *Proceedings of the Fourteenth ACM Conference on Data and Application Security and Privacy*, 289–294.

> **General Smishtank References:**
> Timko, D., & Rahman, M. L. (2023). Commercial Anti-Smishing Tools and Their Comparative Effectiveness Against Modern Threats. *Proceedings of the 16th ACM Conference on Security and Privacy in Wireless and Mobile Networks*, 1–12.

---

## 3. SMS Phishing Dataset for Machine Learning & Pattern Recognition

**Overview:**
Published on Mendeley Data (June 20, 2022), this is a comprehensively labeled dataset containing SMS messages curated specifically for deep learning and targeted attribute extraction research. The data was gathered partly by running OCR (Optical Character Recognition) via Python on internet-sourced images of SMS screens.

*   **Version:** 1
*   **Authors:** Sandhya Mishra, Devpriya Soni
*   **Total Records:** 5,971 messages.
*   **Class Distribution:** 4,844 Ham | 489 Spam | 638 Smishing.
*   **DOI:** [10.17632/f45bkkt8pr.1](https://doi.org/10.17632/f45bkkt8pr.1)

**Data Dictionary & Attributes:**
The dataset provides pre-extracted and categorized boolean/textual features to assist early classification stages:

| Attribute | Type | Description |
| :--- | :---: | :--- |
| `LABEL` | *Categorical* | The ground truth target variable classifying the message as `Ham`, `Spam`, or `Smishing`. |
| `TEXT` | *String* | The raw, unprocessed textual content of the message. |
| `URL` | *Boolean* | Indicates whether the message payload contains a hyperlinked URL (`True`/`False`). |
| `EMAIL` | *Boolean* | Indicates whether an email address is present in the payload. |
| `PHONE` | *Boolean* | Indicates whether a phone number string is detected in the payload. |

*(Note: The original source package also included Python extraction scripts and frequency charts, though we rely primarily on the raw tabular data for our own pipeline).*
