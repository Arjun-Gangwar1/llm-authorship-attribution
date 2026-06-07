llm-text-classifier/                         ← Git root
│
├── README.md                                ← Project overview + results table
├── requirements.txt                         ← pip packages
├── environment.yml                          ← conda env export
├── setup.py                                 ← makes src/ importable
├── .gitignore                               ← excludes data/, models/, cache/
├── test_gpu.py                              ← GPU verification script
│
├── data/
│   ├── raw/                                 ← ❌ gitignored
│   │   ├── train/train.parquet              (960,000 rows)
│   │   ├── val/val.parquet                  (99,600 rows)
│   │   └── test/test.parquet                (99,600 rows)
│   ├── processed/                           ← cleaned/tokenized data
│   └── external/                            ← GloVe/FastText vectors
│
├── notebooks/
│   ├── 01_eda_and_preprocessing.ipynb
│   ├── 02_tfidf_classical_models.ipynb
│   ├── 03_word_embeddings.ipynb             ← Word2Vec, GloVe, FastText
│   ├── 04_sentence_embeddings.ipynb         ← MiniLM, MPNet, BGE, E5
│   ├── 05_deep_learning_rnn_cnn.ipynb       ← LSTM, BiLSTM, GRU, TextCNN
│   ├── 06_transformers/
│   │   ├── 06a_frozen_classifier.ipynb
│   │   ├── 06b_unfreeze_2layers.ipynb
│   │   ├── 06c_unfreeze_4layers.ipynb
│   │   ├── 06d_unfreeze_6layers.ipynb
│   │   └── 06e_full_finetune.ipynb
│   ├── 07_hybrid_features.ipynb             ← SBERT + TF-IDF + Stylometric
│   ├── 08_ensemble.ipynb
│   ├── 09_advanced_techniques.ipynb         ← Contrastive, perplexity features
│   └── 10_results_analysis.ipynb
│
├── src/                                     ← ✅ committed
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── config.yaml
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load_data.py
│   │   └── preprocess.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── tfidf.py
│   │   ├── word_embeddings.py               ← Word2Vec, GloVe, FastText
│   │   ├── sentence_embeddings.py           ← SBERT wrapper + caching
│   │   ├── stylometric.py                   ← 40+ feature extractor
│   │   └── hybrid.py                        ← combines feature sets
│   ├── models/
│   │   ├── __init__.py
│   │   ├── classical/
│   │   │   ├── __init__.py
│   │   │   ├── sklearn_models.py            ← NB, LR, SVM, RF, ExtraTrees
│   │   │   └── boosting.py                  ← LightGBM, XGBoost, CatBoost
│   │   ├── deep/
│   │   │   ├── __init__.py
│   │   │   ├── mlp.py                       ← Deep MLP with ResBlocks
│   │   │   ├── rnn_models.py                ← LSTM, BiLSTM, GRU
│   │   │   └── textcnn.py                   ← TextCNN
│   │   └── transformers/
│   │       ├── __init__.py
│   │       ├── deberta.py                   ← DeBERTa trainer
│   │       ├── roberta.py
│   │       └── distilbert.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_classical.py
│   │   └── train_neural.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py					 ← All evaluation metrics
│   │   └── visualize.py				← Confusion matrix, ROC plots
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       ├── seed.py
│       └── model_saver.py			 ← Save/load any model type

│
├── experiments/
│   ├── logs/                                ← ❌ gitignored
│   ├── results/                             ← ✅ committed
│   │   ├── classical_results.csv
│   │   ├── embedding_results.csv
│   │   ├── transformer_results.csv
│   │   └── final_comparison.csv
│   └── configs/
│       ├── deberta_base.yaml
│       └── lgbm_best.yaml
│
├── models/
│   ├── saved_models/                        ← ❌ gitignored
│   ├── checkpoints/                         ← ❌ gitignored
│   └── deberta_classifier/                  ← ❌ gitignored
│
├── transformers_experiments/
│   ├── README.md
│   ├── experiment_01_frozen.ipynb
│   ├── experiment_02_unfreeze2.ipynb
│   ├── experiment_03_unfreeze4.ipynb
│   ├── experiment_04_unfreeze6.ipynb
│   └── experiment_05_full_finetune.ipynb
│
├── scripts/
│   ├── run_all_classical.py
│   ├── run_all_embeddings.py
│   ├── run_all_transformers.py
│   └── generate_report.py
│
├── reports/
│   ├── figures/                             ← PNG/SVG plots ✅ committed
│   ├── tables/                              ← Excel/CSV tables ✅ committed
│   └── final_report.pdf
│
├── plots/                                   ← auto-generated during runs
└── hf_cache/                                ← ❌ gitignored (HuggingFace models)

