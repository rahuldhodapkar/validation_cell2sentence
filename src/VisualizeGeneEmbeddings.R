#!/usr/bin/env Rscript
# 
# Visualize gene embeddings for both human and zebrafish genes 
#
# @author Rahul Dhodapkar
#

# load generics
library(stringr)
library(ggplot2)
library(umap)

# cross species homology
library(homologene)
library(biomaRt)
library(lsa)

# gene set enrichment analysis
library(msigdbr)

# load data
gene.metadata <- read.csv('./data/vocab', sep = ' ', header = F)
colnames(gene.metadata) <- c('gene_id', 'freq')

species.meta <- str_match(gene.metadata$gene_id, '^([A-Za-z]*)_')
gene.metadata$species <-  species.meta[,2]
gene.metadata$gene <- str_to_lower(str_replace(gene.metadata$gene_id, species.meta[,1], ''))

# get homology map
ensembl <- useEnsembl(biomart = "genes", dataset = "drerio_gene_ensembl")
ncbi.ids.map <- getBM(attributes = c('ensembl_gene_id','entrezgene_id'),
                      filters = c('ensembl_gene_id'),
                      values = gene.metadata$gene[gene.metadata$species == 'zebrafish'],
                      mart = ensembl)

human.gene.ids <- homologene(ncbi.ids.map$entrezgene_id, inTax=7955, outTax=9606)
human.gene.ids$zf_entrezgene_id <- human.gene.ids$X7955_ID

human.gene.ids$zf_gene_name <- str_to_lower(human.gene.ids[['7955']])
human.gene.ids$human_gene_name <- str_to_lower(human.gene.ids[['9606']])

# read in gene embedding
gene.emb <- read.csv('./data/gene_embeddings.csv', header = F)
gene.emb <- as.matrix(gene.emb)

gene.emb.pca <- prcomp(gene.emb, scale = TRUE)
var.explained = gene.emb.pca$sdev^2 / sum(gene.emb.pca$sdev^2)
scree.bounds <- c(1:10)
#create scree plot
qplot(scree.bounds, var.explained[scree.bounds]) + 
  geom_line() + 
  xlab("Principal Component") + 
  ylab("Variance Explained") +
  ggtitle("Scree Plot") +
  ylim(0, 1)

# plot by pcs

df.to.plot <- data.frame(
  comp1 = gene.emb.pca$x[,1],
  comp2 = gene.emb.pca$x[,2],
  name = gene.metadata$gene,
  species = gene.metadata$species,
  freq = gene.metadata$freq
)
ggplot(df.to.plot, aes(x=comp1, y=comp2, color=species)) +
  geom_point()

# generate umap
umap(gene.emb)

### evaluate homologous gene embedding space distance
#gene.metadata <- subset(gene.metadata, freq > 20000)
euclidean <- function(a, b) sqrt(sum((a - b)^2))
cos.dist <- function(a, b) 1-cosine(a,b)

# identify homologs by string matching
human.gene.ixs <- which(gene.metadata$species == 'human')
zf.gene.ixs <- which(gene.metadata$species == 'zebrafish')

n.homologs <- 0
homolog.hu <- c()
homolog.zf <- c()
for (i in human.gene.ixs) {
  if (gene.metadata$gene[[i]] %in% gene.metadata$gene[zf.gene.ixs]) {
    n.homologs <- n.homologs + 1
    homolog.hu <- c(homolog.hu, i)
    homolog.zf <- c(homolog.zf, 
                    zf.gene.ixs[which(gene.metadata$gene[zf.gene.ixs] == gene.metadata$gene[[i]])[[1]]])
  }
}

for (i in 1:5) {
  print(gene.metadata$gene[homolog.hu[i]])
  print(gene.metadata$gene[homolog.zf[i]])
}

homolog.dists <- c()
homolog.freqs <- c()
for (i in 1:n.homologs) {
  v.human <- gene.emb[homolog.hu[i],]
  v.zf <- gene.emb[homolog.zf[i],]
  homolog.dists <- c(homolog.dists, euclidean(v.human, v.zf))
  homolog.freqs <- c(homolog.freqs,
    gene.metadata$freq[homolog.zf[[i]]] + gene.metadata$freq[homolog.hu[[i]]])
}
homolog.dist.df <- data.frame(
  dist = homolog.dists,
  freq = homolog.freqs,
  name = gene.metadata$gene[homolog.hu]
)

ggplot(homolog.dist.df, aes(x=freq, y=dist)) +
  geom_point()


# comparison set
set.seed(7)
comparison.replicates <- n.homologs

from <- human.gene.ixs[sample(1:length(human.gene.ixs), comparison.replicates)]
to <- zf.gene.ixs[sample(1:length(zf.gene.ixs), comparison.replicates)]

comparison.dists <- c()
comparison.freqsum <- c()
for (i in 1:comparison.replicates) {
  comparison.dists <- c(comparison.dists, euclidean(gene.emb[from[[i]],],gene.emb[to[[i]],]))
  comparison.freqsum <- c(comparison.freqsum,
    gene.metadata$freq[from[[i]]] + gene.metadata$freq[to[[i]]])
}
comparison.dist.df <- data.frame(
  dist = comparison.dists,
  freq = comparison.freqsum,
  name = paste(gene.metadata$gene[from], gene.metadata$gene[to], sep='_')
)
ggplot(comparison.dist.df, aes(x=freq, y = dist)) +
  geom_point()

homolog.dist.df$experiment <- 'homologs'
comparison.dist.df$experiment <- 'random'

plot.df <- rbind(homolog.dist.df, comparison.dist.df)

ggplot(plot.df, aes(x=freq, y=dist, color=experiment)) +
  geom_point(alpha=0.4)

freq.cutoff <- quantile(c(homolog.dist.df$freq, comparison.dist.df$freq), probs=c(0.95))

mean(homolog.dist.df[homolog.dist.df$freq > freq.cutoff,]$dist)
mean(comparison.dist.df[comparison.dist.df$freq > freq.cutoff,]$dist)

wilcox.test(homolog.dist.df[homolog.dist.df$freq > freq.cutoff,]$dist,
            comparison.dist.df[comparison.dist.df$freq > freq.cutoff,]$dist)

ggplot(subset(plot.df, freq > freq.cutoff), aes(x=freq, y=dist, color=experiment)) +
  geom_point(alpha=0.4)

################################################################################
## UMAP embedding
################################################################################
set.seed(42)

high.freq.gene.cutoff <- quantile(gene.metadata$freq, probs = c(0.95))
high.freq.gene.ixs <- which(gene.metadata$freq > high.freq.gene.cutoff)

# high.freq.gene.ixs <- sample(1:nrow(gene.metadata), 1000)


gene.umap.emb <- umap(d=gene.emb[high.freq.gene.ixs,])

df.to.plot <- data.frame(
  umap1 = gene.umap.emb$layout[,1],
  umap2 = gene.umap.emb$layout[,2],
  name = gene.metadata$gene[high.freq.gene.ixs],
  species = gene.metadata$species[high.freq.gene.ixs],
  freq = gene.metadata$freq[high.freq.gene.ixs]
)
ggplot(df.to.plot, aes(x=umap1, y=umap2, color=freq)) +
  geom_point(alpha=0.4)

# color plot by hallmark gene sets
h_gene_sets_zf <- msigdbr(species = "zebrafish", category = "H")
h_gene_sets_hu <- msigdbr(species = "human", category = "H")

sets <- unique(h_gene_sets_hu$gs_name)
set.name <- 'HALLMARK_APOPTOSIS'
h.genes <- str_to_lower(
  c(h_gene_sets_zf[h_gene_sets_zf$gs_name == set.name,]$gene_symbol,
    h_gene_sets_hu[h_gene_sets_hu$gs_name == set.name,]$gene_symbol))
df.to.plot$in.set <- df.to.plot$name %in% h.genes
ggplot(df.to.plot, aes(x=umap1, y=umap2, color=in.set)) +
  geom_point(alpha=0.4)

################################################################################
## Formal testing for within-group and between-group distance
################################################################################
sets <- unique(h_gene_sets_hu$gs_name)

calc.dist.and.freq <- function(i1, i2) {
  return(list(
    'dist' = euclidean(gene.emb[i1,], gene.emb[i2,]),
    'freq' = gene.metadata$freq[i1] + gene.metadata$freq[i2]
  ))
}

combined.within.set.dists <- c()
combined.between.set.dists <- c()

n.dist.replicates <- 100
for (i in 1:length(sets)) {
  set.name <- sets[[i]]
  h.genes <- str_to_lower(
    c(h_gene_sets_zf[h_gene_sets_zf$gs_name == set.name,]$gene_symbol,
      h_gene_sets_hu[h_gene_sets_hu$gs_name == set.name,]$gene_symbol))
  
  genes.in.set <- which(gene.metadata$gene %in% h.genes)
  genes.not.in.set <- which(!gene.metadata$gene %in% h.genes)
  
  within.set.pair.from <- rep(-1, n.dist.replicates)
  within.set.pair.to <- rep(-1, n.dist.replicates)
  for (i in 1:n.dist.replicates) {
    x <- sample(1:length(genes.in.set), 2)
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

  between.set.pair.from <- sample(1:length(genes.in.set), n.dist.replicates)
  between.set.pair.to <- sample(1:length(genes.not.in.set), n.dist.replicates)
  between.set.dists <- c()
  between.set.freqs <- c()
  for (j in 1:n.dist.replicates) {
    x <- calc.dist.and.freq(between.set.pair.from[j], between.set.pair.to[j])
    between.set.dists <- c(between.set.dists, x$dist)
    between.set.freqs <- c(between.set.freqs, x$freq)
  }

  combined.within.set.dists <- c(combined.within.set.dists, within.set.dists)
  combined.between.set.dists <- c(combined.between.set.dists, between.set.dists)

}
mean(combined.within.set.dists)
mean(combined.between.set.dists)

wilcox.test(combined.within.set.dists, combined.between.set.dists)
