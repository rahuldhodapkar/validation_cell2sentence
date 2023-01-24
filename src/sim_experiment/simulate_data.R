#!/usr/bin/env Rscript
#
# @author Rahul Dhodapkar
#

suppressPackageStartupMessages({
  library(splatter)
})
library(igraph)
library(cccd)
library(pdfCluster)
library(stringr)

library(reticulate)
use_condaenv('c2s')
cs <- import('cell2sentence')
ad <- import('anndata')

################################################################################
# Create Output Scaffolding
################################################################################

s <- ifelse(!dir.exists('./fig'), dir.create('./fig'), FALSE)
s <- ifelse(!dir.exists('./fig/sim_experiment'), dir.create('./fig/sim_experiment'), FALSE)

s <- ifelse(!dir.exists('./calc'), dir.create('./calc'), FALSE)
s <- ifelse(!dir.exists('./calc/sim_experiment'), dir.create('./calc/sim_experiment'), FALSE)

################################################################################
# Run Simulations
################################################################################

upsert <- function(df, update) {
    if (is.null(df)) {
        return(update)
    } else {
        return(rbind(df, update))
    }
}

# Define data simulation function using splatter
simulate <- function(de.facScale, nGenes=200, nCells=100) {
    sim.groups <- splatSimulate(
        nGenes=nGenes,
        batchCells=nCells,
        group.prob = c(0.5, 0.5),
        method = "groups",
        de.facScale = de.facScale,
        lib.loc = 5,
        verbose = FALSE)
    meta.df <- data.frame(
        Group=sim.groups$Group,
        Cell=sim.groups$Cell
    )
    return(list(t(sim.groups@assays@data$counts), meta.df))
}

# define igraph clustering methods for general use
graph.clustering.methods <- list(
'spinglass'=function(x, dx) {
    knn.graph <- nng(dx=as.matrix(dx), k=15)
    clust.res <- cluster_spinglass(as.undirected(knn.graph))
    return(clust.res$membership)
},
'walktrap'=function(x, dx) {
    knn.graph <- nng(dx=as.matrix(dx), k=15)
    clust.res <- cluster_walktrap(as.undirected(knn.graph))
    return(clust.res$membership)
},
'leading_eigen'=function(x, dx) {
    knn.graph <- nng(dx=as.matrix(dx), k=15)
    clust.res <- cluster_leading_eigen(as.undirected(knn.graph))
    return(clust.res$membership)
},
'louvain'=function(x, dx) {
    knn.graph <- nng(dx=as.matrix(dx), k=15)
    clust.res <- cluster_louvain(as.undirected(knn.graph))
    return(clust.res$membership)
})

# define expression-matrix based methods of clustering
expr.funcs <- list(
    'euclidean'=c(list(
        'kmeans'=function(x, dx) {
            c <- kmeans(x, centers=2)
            return(c$cluster)
        }), graph.clustering.methods),
    'binary'=graph.clustering.methods,
    'manhattan'=graph.clustering.methods
)

# define edit-distance based methods of clustering
edit.funcs <- list(
    'levenshtein'=graph.clustering.methods,
    'damerau_levenshtein'=graph.clustering.methods,
    'jaro'=graph.clustering.methods,
    'jaro_winkler'=graph.clustering.methods
)

funcs <- c(expr.funcs, edit.funcs)

# define interpoint distance calculation methods
calc.dx <- function(x, method) {
    if (method %in% names(expr.funcs)) {
        # need to normalize expression matrix first
        x <- log1p(1e4 * x / rowSums(x))
        return(dist(x, method=method))
    } else if (method %in% names(edit.funcs)) {
        adata <- ad$AnnData(X=x)
        csdata <- cs$transforms$csdata_from_adata(adata)
        return(as.dist(csdata$create_distance_matrix(dist_type=method)))
    }
}

set.seed(42)
sep.trials <- seq(from=0, to=2, by=0.1)
plot.df <- NULL
for (separation in sep.trials) {
    sim.data <- simulate(separation)
    ref.clusters <- sim.data[[2]]$Group

    for (s in names(funcs)) {
        x = sim.data[[1]]
        dx = calc.dx(x, method=s)

        for(clust.method in names(funcs[[s]])) {
            ari <- NaN
            tryCatch({
                exp.clusters <- funcs[[s]][[clust.method]](x, dx)

                # evaluation metric (adjusted Rand index)
                ari <- adj.rand.index(exp.clusters, ref.clusters)
            }, error=function(e) {
                message(paste0(separation,'|', s,'|', clust.method, ', failed'))
                print(e)
            }, warning=function(w) {
                message(paste0(separation,'|', s,'|', clust.method, ', failed'))
                print(w)
            })

            plot.df <- upsert(plot.df, data.frame(
                'ARI'=ari,
                'separation'=separation,
                'distance_type'=s,
                'clustering_type'=clust.method
            ))
        }
    }
}

write.csv(plot.df, './calc/sim_experiment/sim_results.csv', row.names=F)

print('All done!')
