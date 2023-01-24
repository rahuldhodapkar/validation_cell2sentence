.PHONY: setup pbmc sim

setup:
	pip install -r requirements.txt
	./src/setup/Dependencies.R

pbmc:
	./src/pbmc_experiment/gen_umap_concordance_data.py
	./src/pbmc_experiment/plot_umap_concordance.R
	./src/pbmc_experiment/calc_dist_concordance.R

sim: # simulated clustering benchmark
	./src/sim_experiment/simulate_data.R
	./src/sim_experiment/plot_sim_results.R
