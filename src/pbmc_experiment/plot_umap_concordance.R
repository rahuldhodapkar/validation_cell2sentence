#!/usr/bin/env Rscript
#
# @author Rahul Dhodapkar
#

library(ggplot2)
library(cowplot)
library(stringr)

################################################################################
# Generate Plots
################################################################################

distance_types <- c(
    'levenshtein',
    'damerau_levenshtein',
    'jaro',
    'jaro_winkler',
    'zlib_ncd'
)

for (d in distance_types) {
    plot.df <- read.csv(paste0('./calc/pbmc_experiment/',d,'_umap_data.csv'))
    ggplot(plot.df, aes(x=edit_UMAP, y=orig_UMAP, color=celltype)) +
        geom_point() +
        theme_cowplot()
    ggsave(paste0('./fig/pbmc_experiment/', d, '_umap_concordance.png'),
            width=8, height=6)
}

print('All done!')
