#!/usr/bin/env Rscript
# 
# Visualize cell embeddings for validation datasets.
#
# @author Rahul Dhodapkar
#

library(stringr)
library(ggplot2)
library(umap)
library(dplyr)

################################################################################
## Create Output Scaffolding
################################################################################

ifelse(!dir.exists("./fig"), dir.create("./fig"), FALSE)
ifelse(!dir.exists("./fig/cell_embeddings"), dir.create("./fig/cell_embeddings"), FALSE)

ifelse(!dir.exists("./calc"), dir.create("./calc"), FALSE)
ifelse(!dir.exists("./calc/cell_embeddings"), dir.create("./calc/cell_embeddings"), FALSE)

################################################################################
## Load Cell Embeddings
################################################################################

hu.cell.emb <- read.csv('./data/human_cell_embeddings.csv', header = F)
zf.cell.emb <- read.csv('./data/zebrafish_cell_embeddings.csv', header = F)
mu.cell.emb <- read.csv('./data/mouse_cell_embeddings.csv', header = F)
ch.cell.emb <- read.csv('./data/chick_cell_embeddings.csv', header = F)

species.labels <- c(rep('human', nrow(hu.cell.emb)),
                    rep('zebrafish', nrow(zf.cell.emb)),
                    rep('mouse', nrow(mu.cell.emb)),
                    rep('chick', nrow(ch.cell.emb)))
comb.emb <- as.matrix(rbind(hu.cell.emb,
                            zf.cell.emb,
                            mu.cell.emb,
                            ch.cell.emb))

n.cells.to.sample <- 2000
sample.ixs <- sample(1:nrow(comb.emb), n.cells.to.sample)

comb.emb.umap <- umap(d=comb.emb[sample.ixs,])

df.to.plot <- data.frame(
  umap1=comb.emb.umap$layout[,1],
  umap2=comb.emb.umap$layout[,2],
  species=species.labels[sample.ixs]
)

ggplot(df.to.plot, aes(x=umap1,y=umap2,color=species)) +
  geom_point()

################################################################################
## Generate ZF Embedding Separately
################################################################################

n.cells.to.sample <- 2000
zf.sample.ixs <- sample(1:nrow(zf.cell.emb), n.cells.to.sample)

zf.emb.umap <- umap(d=zf.cell.emb[zf.sample.ixs,])

zf.plot.df <- data.frame(
  umap1=zf.emb.umap$layout[,1],
  umap2=zf.emb.umap$layout[,2]
)

ggplot(zf.plot.df, aes(x=umap1,y=umap2)) +
  geom_point()

################################################################################
## Generate HU Embedding Separately
################################################################################

hu.sample.barcodes <- read.csv('./data/valid_barcodes.human', header=F)
hu.sample.metadata <- read.csv('./data/Human_retina_combined_all_meta.csv', skip=1, header=T)

hu.merged.metadata <- left_join(hu.sample.barcodes, hu.sample.metadata,
                                by=c('V1' = 'TYPE'))

matches <- str_match(hu.merged.metadata$V1, '^(H\\d)([A-Za-z0-9]*)(S\\d)_')
hu.merged.metadata$donor <- matches[,2]
hu.merged.metadata$condition <- matches[,3]
hu.merged.metadata$sample <- matches[,4]

n.cells.to.sample <- 2000
hu.sample.ixs <- sample(1:nrow(hu.cell.emb), n.cells.to.sample)

hu.emb.umap <- umap(d=hu.cell.emb[hu.sample.ixs,])

hu.plot.df <- data.frame(
  umap1=hu.emb.umap$layout[,1],
  umap2=hu.emb.umap$layout[,2],
  celltype=hu.merged.metadata$group[hu.sample.ixs],
  donor=hu.merged.metadata$donor[hu.sample.ixs],
  condition=hu.merged.metadata$condition[hu.sample.ixs],
  sample=hu.merged.metadata$sample[hu.sample.ixs]
)

ggplot(hu.plot.df, aes(x=umap1,y=umap2, color=celltype)) +
  geom_point()



