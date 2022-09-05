#!/usr/bin/env Rscript
## VizWord2VecEmbeddings.R
#
# visualize embeddings from word2vec on genes.
#

library(ggplot2)
library(word2vec)
library(umap)
library(stringr)
library(cccd)
library(igraph)
library(msigdbr)
library(lsa)
library(cowplot)
library(RColorBrewer)

################################################################################
## Create Output Scaffolding
################################################################################

s <- ifelse(!dir.exists('./fig'), dir.create('./fig'), FALSE)
s <- ifelse(!dir.exists('./fig/word2vec'), dir.create('./fig/word2vec'), FALSE)

s <- ifelse(!dir.exists('./calc'), dir.create('./calc'), FALSE)
s <- ifelse(!dir.exists('./calc/word2vec'), dir.create('./calc/word2vec'), FALSE)

################################################################################
## Load Data
################################################################################

species.idx <- 4

species.long.names <- c('human', 'mouse', 'zebrafish', 'chick')
species.short.names <- c('hu', 'mu', 'zf', 'ch')

species.long <- species.long.names[[species.idx]]
species.short <- species.short.names[[species.idx]]

raw.expression <- read.csv(paste0('./data/raw_exp/', species.long ,'_hvg_expression.csv'))
#rownames(raw.expression) <- raw.expression$X
raw.expression$X <- NULL

raw.pca <- prcomp(t(as.matrix(raw.expression)),
                  center = T, scale = T,
                  rank=200)
pca.emb <- raw.pca$x

custom.settings = umap.defaults
custom.settings$n_components = 200
umap.emb <- umap(raw.pca$x, config=custom.settings)$layout

w2v.embeddings <- read.word2vec(paste0('./data/word2vec_embs/',species.short,'_gene_vec.bin'))
w2v.emb <- as.matrix(w2v.embeddings)

################################################################################
## Visualize Embeddings
################################################################################

emb.list <- list(pca.emb, umap.emb, w2v.emb)
emb.names <- c('PCA', 'UMAP', 'word2vec')

stats.df <- NULL

for (i in 1:length(emb.list)) {
  emb <- emb.list[[i]]
  emb.name <- emb.names[[i]]
  
  emb.umap.proj <- umap(emb)
  
  species.meta <- str_match(rownames(emb), '^([A-Za-z]*)_')
  
  gene.metadata <- data.frame(
    species = species.meta[,2],
    gene = str_to_lower(str_replace(rownames(emb), species.meta[,1], '')),
    umap1 = emb.umap.proj$layout[,1],
    umap2 = emb.umap.proj$layout[,2]
  )
  
  ggplot(gene.metadata, aes(x=umap1, y=umap2)) +
    geom_point()
  
  # calculate nearest neighbor graph
  knn.graph <- nng(x=emb, k=15)
  clust.res <- cluster_louvain(as.undirected(knn.graph))
  
  gene.metadata$cluster <- as.factor(paste("C", clust.res$membership, sep=''))
  
  ggplot(gene.metadata, aes(x=umap1, y=umap2, color=cluster)) +
    geom_point()
  ggsave(paste0('./fig/word2vec/', species.short, '_', emb.name, '_gene_embedding.png'), width=8, height=8)
  
  ################################################################################
  ## Formally Evaluate Embedding Coherence
  ################################################################################
  
  h_gene_sets_hu <- msigdbr(species = "human", category = "H")
  sets <- unique(h_gene_sets_hu$gs_name)
  
  df.to.plot <- gene.metadata
  set.name <- 'HALLMARK_REACTIVE_OXYGEN_SPECIES_PATHWAY'
  
  h.genes <- str_to_lower(h_gene_sets_hu[h_gene_sets_hu$gs_name == set.name,]$gene_symbol)
  df.to.plot$in.set <- df.to.plot$gene %in% h.genes
  ggplot(df.to.plot, aes(x=umap1, y=umap2, color=in.set)) +
    geom_point(alpha=0.4)
  
  #################
  # across all sets
  #################
  set.seed(42)
  
  euclidean <- function(a, b) sqrt(sum((a - b)^2))
  cos.dist <- function(a, b) 1-cosine(a,b)
  
  sets <- unique(h_gene_sets_hu$gs_name)
  
  calc.dist.and.freq <- function(i1, i2) {
    return(list(
      'dist' = cos.dist(emb[i1,], emb[i2,]),
      'freq' = gene.metadata$freq[i1] + gene.metadata$freq[i2]
    ))
  }
  
  combined.within.set.dists <- c()
  combined.between.set.dists <- c()
  
  per.set.combined.within.set.dists <- c()
  per.set.combined.between.set.dists <- c()
  
  n.dist.replicates <- 100
  for (i in 1:length(sets)) {
    set.name <- sets[[i]]
    h.genes <- str_to_lower(h_gene_sets_hu[h_gene_sets_hu$gs_name == set.name,]$gene_symbol)
  
    genes.in.set <- which(gene.metadata$gene %in% h.genes)
    genes.not.in.set <- which(!gene.metadata$gene %in% h.genes)
    
    if(length(genes.in.set) < 5) { print(paste0("skipping ", set.name)); next; }
    
    within.set.pair.from <- rep(-1, n.dist.replicates)
    within.set.pair.to <- rep(-1, n.dist.replicates)
    for (i in 1:n.dist.replicates) {
      x <- sample(1:length(genes.in.set), 2, replace=T)
      within.set.pair.from[i] <- x[1]
      within.set.pair.to[i] <- x[2]
    }
    within.set.dists <- c()
    within.set.freqs <- c()
    for (j in 1:n.dist.replicates) {
      x <- calc.dist.and.freq(within.set.pair.from[j], within.set.pair.to[j])
      within.set.dists <- c(within.set.dists, x$dist)
      within.set.freqs <- c(within.set.freqs, x$freq)
    }
    
    between.set.pair.from <- sample(1:length(genes.in.set), n.dist.replicates, replace=T)
    between.set.pair.to <- sample(1:length(genes.not.in.set), n.dist.replicates, replace=T)
    between.set.dists <- c()
    between.set.freqs <- c()
    for (j in 1:n.dist.replicates) {
      x <- calc.dist.and.freq(between.set.pair.from[j], between.set.pair.to[j])
      between.set.dists <- c(between.set.dists, x$dist)
      between.set.freqs <- c(between.set.freqs, x$freq)
    }
    
    combined.within.set.dists <- c(combined.within.set.dists, within.set.dists)
    combined.between.set.dists <- c(combined.between.set.dists, between.set.dists)
    
    per.set.combined.within.set.dists <- c(
      per.set.combined.within.set.dists, mean(within.set.dists)
    )
    per.set.combined.between.set.dists <- c(
      per.set.combined.between.set.dists, mean(between.set.dists)
    )
  }
  mean(combined.within.set.dists)
  mean(combined.between.set.dists)
  
  wilcox.test(combined.within.set.dists, combined.between.set.dists)
  
  wilcox.test(per.set.combined.within.set.dists, per.set.combined.between.set.dists)
  
  plot.df <- rbind(
    data.frame(
      type='Within Set',
      mean.dist=per.set.combined.within.set.dists,
      id=1:length(per.set.combined.within.set.dists)
    ), data.frame(
      type='Between Set',
      mean.dist=per.set.combined.between.set.dists,
      id=1:length(per.set.combined.between.set.dists)
    )
  )
  plot.df$embedding <- emb.name
  
  if (is.null(stats.df)) {
    stats.df <- plot.df
  } else {
    stats.df <- rbind(stats.df, plot.df)
  }
  
  # show paired points on boxplot for manuscript figure.
  ggplot(plot.df, aes(x=type, y=mean.dist)) +
    geom_boxplot(width=0.3, size=1.5, fatten=1.5, colour="grey70") +
    geom_point(colour="red", size=2, alpha=0.5) +
    geom_line(aes(group=id), colour="red", linetype="11") +
    theme_classic()
}

ggplot(stats.df, aes(x=type, y=mean.dist, color=embedding)) +
  geom_boxplot(width=0.3, size=1, fatten=1, colour="black") +
  geom_point(size=2, alpha=0.5) +
  geom_line(aes(group=id), colour="grey70", linetype="11") +
  theme_classic() + background_grid() +
  facet_grid(.~embedding) +
  scale_color_manual(values=brewer.pal(n=3, name='Set2'))
ggsave(paste0('./fig/word2vec/', species.short, '_hallmark_boxplot.png'), width=8, height=8)

# get stats
for (i in 1:length(emb.list)) {
  emb.name <- emb.names[[i]]
  print(paste('Wilcox Test for [', emb.name, ']', sep=''))
  test <- wilcox.test(
    stats.df$mean.dist[
      stats.df$type == 'Between Set'
      & stats.df$embedding == emb.name], 
    stats.df$mean.dist[
      stats.df$type == 'Within Set'
      & stats.df$embedding == emb.name])
  print(test)
}
write.csv(stats.df, paste0('./calc/word2vec/', species.short, '_stats_df.csv'), row.names = F)

print('All done!')
