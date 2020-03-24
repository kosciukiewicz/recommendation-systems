setlocal
set PYTHONPATH=%CD%
dvc run^
 -d prepare_dataset.py^
 -o dataset^
 -f prepare_dataset.dvc^
 python prepare_dataset.py
dvc run^
 -d evaluation/scripts/run_experiments.py^
 -d dataset -d clustering -d collaborative_filtering -d content_based_recomendation -d hybrid -d deep_learning -d data -d utils -d sequence^
 -o results/run_experiments.csv^
 -f run_experiments.dvc^
 python evaluation/scripts/run_experiments.py
dvc run^
 -d evaluation/scripts/run_deep_learning_experiments.py^
 -d dataset -d clustering -d collaborative_filtering -d content_based_recomendation -d hybrid -d deep_learning -d data -d utils -d sequence^
 -o results/run_deep_learning_experiments.csv^
 -f run_deep_learning_experiments.dvc^
 python evaluation/scripts/run_deep_learning_experiments.py
dvc run^
 -d evaluation/scripts/run_clustering_experiments.py^
 -d dataset -d clustering -d collaborative_filtering -d content_based_recomendation -d hybrid -d deep_learning -d data -d utils -d sequence^
 -o results/run_clustering_experiments.csv^
 -f run_clustering_experiments.dvc^
 python evaluation/scripts/run_clustering_experiments.py
endlocal
pause