#!/usr/bin/env Rscript
#
# @author Rahul Dhodapkar
#

library(ade4)
library(stringr)

################################################################################
# Run Statistics
################################################################################

symmetrize.dist.mat <- function(m) { return( (m + t(m)) / 2 ) }

raw.dist <- read.table('./calc/pbmc_experiment/raw_dist_mat.csv',
    sep=',', header=F)

distance.types <- c(
    'levenshtein',
    'damerau_levenshtein',
    'jaro',
    'jaro_winkler',
    'zlib_ncd'
)

dist.tests <- list()
for (i in 1:length(distance.types)) {
    d <- distance.types[i]
    print(paste0('===== Processing ', d, ' ====='))

    tmp.dist <- read.table(
        paste0('./calc/pbmc_experiment/',d,'_dist_mat.csv'),
        sep=',', header=F)
    # symmetrize distance matrix (NOOP if already symmetric)
    tmp.dist <- symmetrize.dist.mat(tmp.dist)

    dist.tests[[i]] <- mantel.rtest(
        as.dist(raw.dist), as.dist(tmp.dist), nrepet = 999)
}
names(dist.tests) <- distance.types

plot.df <- data.frame(
    dist_name=names(dist.tests),
    mantel_obs=sapply(1:length(dist.tests), function(i){dist.tests[[i]]$obs}),
    mantel_pval=sapply(1:length(dist.tests), function(i){dist.tests[[i]]$pvalue})
)
write.csv(plot.df, './calc/pbmc_experiment/mantel_test_results.csv', row.names=F)

print(dist.tests)
print('All done!')
