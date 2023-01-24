.PHONY: setup pbmc

setup:
	pip install -r requirements.txt
	./src/setup/Dependencies.R

pbmc:
	./src/pbmc_experiment/gen_umap_concordance_data.py
	./src/pbmc_experiment/plot_umap_concordance.R
	./src/pbmc_experiment/calc_dist_concordance.R
