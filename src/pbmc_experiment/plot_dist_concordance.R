#!/usr/bin/env Rscript
#
# @author Rahul Dhodapkar
#

library(ggplot2)
library(cowplot)

################################################################################
# Generate Plots
################################################################################

plot.df <- read.csv('./calc/pbmc_experiment/mantel_test_results.csv')

ggplot(plot.df, aes(x=dist_name, y=mantel_obs, fill=dist_name)) +
    geom_col() +
    theme_cowplot()
ggsave('./fig/pbmc_experiment/mantel_test_results.png', width=5, height=5)

print('All done!')
