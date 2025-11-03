
#!/usr/bin/env bash
set -e
python fetch_and_merge.py --include_top5 --add_local "psl_2024_25_results_rounds1_10.csv" "nedbank_cup_2024_25_from_user_paste.csv"
python train_model.py --csv merged_matches.csv --out model.pkl
