#!/usr/bin/env Rscript
#
# @author Rahul Dhodapkar
#

library(ggplot2)
library(cowplot)

plot.df <- read.csv('./calc/sim_experiment/sim_results.csv')

ggplot(plot.df, aes(y=ARI, x=separation, color=distance_type)) +
    geom_line() +
    facet_grid(~clustering_type) +
    theme_cowplot() +
    background_grid()

ggsave('./fig/sim_experiment/sim_results.png', width=12, height=5)

print('All done!')
