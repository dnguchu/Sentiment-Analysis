# Day 6 Analysis — instructions

Run `notebooks/Day6_Evaluation_and_Analysis.ipynb` from repo root.

Required input files (place under `feature_engineering_output/`):
- tfidf_X_train.npz, tfidf_X_test.npz, y_train.npy, y_test.npy
- tfidf_vectorizer.pkl (recommended) or tfidf_vocabulary.txt

Optional (if present):
- models/*.pkl            -> evaluated automatically if present
- train_split.csv, test_split.csv -> used to map misclassified rows to raw text

Outputs created:
- analysis_results/model_evaluation_summary.csv
- analysis_results/top_features_*.csv
- analysis_results/cosine_similarity_top_words.csv
- analysis_results/false_pos_examples.csv (if mapping available)
- figs/*.png
