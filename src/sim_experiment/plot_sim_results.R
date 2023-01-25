#!/usr/bin/env Rscript
#
# @author Rahul Dhodapkar
#

library(ggplot2)
library(cowplot)
library(dplyr)

################################################################################
# Generate Plots
################################################################################

plot.df.agg <- read.csv('./calc/sim_experiment/sim_results.csv')

plot.df <- plot.df.agg %>%
                group_by(separation, distance_type, clustering_type) %>%
                summarize(
                    mean.ARI = mean(ARI),
                    sd.ARI = sd(ARI),
                    .groups='keep') %>%
                as.data.frame()

ggplot(plot.df, aes(y=mean.ARI, x=separation, color=distance_type)) +
    geom_line() +
    facet_grid(~clustering_type) +
    theme_cowplot() +
    background_grid()
ggsave('./fig/sim_experiment/sim_results.png', width=20, height=5)

print('All done!')
