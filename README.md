# Sparse vs. Dense Representations for Short Social Media Text

**Unsupervised Sentiment and Topic Structure Across TF-IDF, Doc2Vec, and Sentence-BERT Embeddings**

Max Chalekson · Northwestern University MSDS 453 · Spring 2026

---

## Overview

This project investigates whether pretrained sentence embeddings produce meaningfully richer unsupervised structure than sparse bag-of-words representations on short social-media text — and whether the practical gains justify the added computational overhead. Three representation strategies (TF-IDF, Doc2Vec, Sentence-BERT) are applied to two distinct corpora under matched experimental conditions. Clustering quality is evaluated both against ground-truth sentiment labels (where available) and via unsupervised intrinsic metrics.

---

## Research Question

> Does moving from sparse (TF-IDF) to dense (Doc2Vec, SBERT) representations produce meaningfully better unsupervised cluster or topic structure on short social-media text?

---

## Datasets

| Corpus | Size (working) | Labels | Source |
|--------|---------------|--------|--------|
| [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140) | 99,822 tweets | Binary sentiment (emoticon-based distant supervision) | Kaggle |
| [Bluesky Posts](https://huggingface.co/datasets/withalim/bluesky-posts) | 99,556 posts | None (unlabeled) | Hugging Face |

**Sentiment140** is a 1.6-million-tweet corpus balanced across positive and negative classes; 50,000 tweets were drawn per class via stratified random sampling (seed 42). It serves as the primary evaluation corpus because ground-truth labels allow supervised-metric assessment of unsupervised clustering.

**Bluesky Posts** is a contemporary unlabeled corpus of 7.8 million posts from the Bluesky decentralized social network. A 300,000-post buffer was sampled, filtered to English using `langdetect` (56.2% retention), and then downsampled to ~100,000 posts. It provides a real-world exploratory surface free from annotation artifacts.

---

## Methods

### Text Representations

| Method | Details |
|--------|---------|
| **TF-IDF** | scikit-learn `TfidfVectorizer`; unigrams + bigrams; `min_df=3`, `max_df=0.95`, `sublinear_tf=True`; 15,529 features (S140) / 20,000 features (Bluesky) |
| **Doc2Vec** | Gensim PV-DM; `vector_size=200`, `window=5`, `min_count=2`, 20 epochs; trained from scratch on each corpus |
| **Sentence-BERT** | `all-MiniLM-L6-v2` checkpoint; 384-dimensional mean-pooled embeddings; no fine-tuning |

### Clustering

K-means applied to each representation with k selected by maximizing silhouette coefficient over k ∈ {2, …, 10}. Silhouette scores computed on a 10,000-document random subsample. ARI and NMI computed against Sentiment140 ground-truth labels.

### Topic Modeling

LDA (`sklearn.decomposition.LatentDirichletAllocation`) with `CountVectorizer` front-end. Topic count T selected by maximizing Cv coherence (Gensim `CoherenceModel`) over T ∈ {5, 10, 15, 20, 25, 30}.

### Sentiment Baseline

VADER applied via `vaderSentiment`; compound score threshold ±0.05 for binary classification; neutral-zone posts excluded from Sentiment140 evaluation metrics (26.7% excluded).

---

## Key Findings

### Clustering (K-means)

| Representation | Corpus | Best k | Silhouette | ARI | NMI |
|---------------|--------|--------|-----------|-----|-----|
| TF-IDF | Sentiment140 | 9 | 0.007 | 0.01 | 0.02 |
| TF-IDF | Bluesky | 10 | 0.008 | — | — |
| Doc2Vec | Sentiment140 | 2 | 0.034 | 0.00 | 0.00 |
| Doc2Vec | Bluesky | 2 | 0.037 | — | — |
| SBERT | Sentiment140 | 3 | 0.019 | 0.00 | 0.00 |
| SBERT | Bluesky | 2 | 0.023 | — | — |

- **All representations produce near-zero silhouette scores**, confirming that unsupervised clustering is a genuinely hard problem on short social-media text regardless of representation choice.
- **Doc2Vec achieves the highest peak silhouette (0.034–0.037) but only at k=2**; scores collapse into negative territory for k ≥ 3, indicating the model captures one coarse binary axis but no finer structure.
- **SBERT is marginally better than TF-IDF** in absolute terms (0.019 vs. 0.007 on Sentiment140), but neither representation recovers the known sentiment partition (ARI ≈ 0).
- **No representation recovers ground-truth sentiment structure** via unsupervised clustering.

### Topic Modeling (LDA)

| Corpus | Optimal T | Cv Coherence |
|--------|-----------|-------------|
| Sentiment140 | 30 | 0.49 |
| Bluesky | 10 | 0.62 |

Bluesky yields substantially higher coherence (0.62), reflecting its broader topical diversity. Sentiment140 coherence is flat across all T values (0.47–0.49), consistent with co-occurrence sparsity in an emoticon-filtered corpus.

### VADER Sentiment Baseline (Sentiment140)

| Precision | Recall | F1 |
|-----------|--------|----|
| 0.66 | 0.86 | 0.75 |

VADER achieves F1 = 0.75 with no training, no learned representations, and no hyperparameter tuning. For binary sentiment classification on Twitter-like text, this sets a strong benchmark that unsupervised clustering does not approach.

---

## Interpretation

1. **Short social-media text resists unsupervised clustering** across all representation types. The volume of noise relative to signal — abbreviations, slang, platform conventions — dilutes the semantic structure that clustering exploits.
2. **Dense embeddings do not substantially outperform TF-IDF** for K-means clustering on this text class. The SBERT advantage exists but is small in absolute terms and practically irrelevant when ARI ≈ 0 across the board.
3. **LDA coherence tracks corpus topical diversity, not representation quality.** The gap between Bluesky (0.62) and Sentiment140 (0.49) reflects corpus character, not model differences.
4. **VADER as a no-training baseline outperforms unsupervised clustering** for the specific task of binary sentiment recovery, suggesting that the overhead of dense embeddings is difficult to justify for this use case.

---

## Repository Structure

```
453-final-project/
├── FP-DataSources/
│   └── 485-DL-FinalProject.ipynb   # Main analysis notebook (6 sections)
├── FP-ReferenceSources/             # Reference PDFs (key papers)
│   ├── 1-s2.0-S2666307424000482-main.pdf     # Petukhova et al. 2025
│   ├── 14550-Article Text-18068-1-2-20201228.pdf  # Qiang et al. 2020
│   ├── 1904.07695v1.pdf                       # Sentence-BERT (Reimers & Gurevych 2019)
│   ├── 2022.acl-long.60.pdf
│   ├── TwitterDistantSupervision09.pdf        # Go et al. 2009 (Sentiment140 original)
│   ├── fsoc-07-886498.pdf                     # Egger & Yu 2022
│   ├── s12652-020-02771-9.pdf
│   └── unnisa-2016-ijca-911317.pdf
└── README.md
```

The notebook is structured in six self-contained sections:
1. Data loading and preprocessing (Sentiment140 + Bluesky)
2. TF-IDF vectorization and K-means clustering
3. Doc2Vec training and K-means clustering
4. Sentence-BERT encoding and K-means clustering
5. LDA topic modeling with coherence-score selection
6. VADER sentiment baseline

---

## Dependencies

```
datasets
gensim
langdetect
matplotlib
numpy
pandas
scikit-learn
sentence-transformers
tqdm
vaderSentiment
```

Install all dependencies:

```bash
pip install datasets gensim langdetect matplotlib numpy pandas scikit-learn sentence-transformers tqdm vaderSentiment
```

**Note:** Sentiment140 must be downloaded separately from [Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140) and placed at the path configured in Section 1 of the notebook (`S140_PATH`). The Bluesky dataset streams directly from Hugging Face via the `datasets` library.

---

## Citation

If referencing this work, please cite:

> Chalekson, Max. 2026. "Sparse vs. Dense Representations for Short Social Media Text: Unsupervised Sentiment and Topic Structure Across TF-IDF, Doc2Vec, and Sentence-BERT Embeddings." MSDS 453 Final Project, Northwestern University.
