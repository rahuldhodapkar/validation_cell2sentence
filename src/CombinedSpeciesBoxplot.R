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
library(tidyr)


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

species.long.names <- c('human', 'mouse', 'chick', 'macaque')
species.short.names <- c('hu', 'mu', 'ch', 'ma')

################################################################################
## Load Data
################################################################################

combined.stats.df <- NULL
for (i in 1:length(species.short.names)) {
  stats.df <- read.csv(paste0('./calc/word2vec/', species.short.names[[i]], '_stats_df.csv'))  
  stats.df$species <- species.long.names[[i]]
  
  if (is.null(combined.stats.df)) {
    combined.stats.df <- stats.df
  } else {
    combined.stats.df <- rbind(combined.stats.df, stats.df)
  }
}

combined.stats.df$species <- factor(combined.stats.df$species,
                                    levels=c('human', 'macaque', 'mouse', 'chick'),
                                    ordered=T)

embeddings <- c('PCA', 'UMAP', 'word2vec')

for (i in 1:length(embeddings)) {
  ggplot(subset(combined.stats.df, embedding == embeddings[[i]]), aes(x=type, y=mean.dist / 2, fill=species)) +
    geom_boxplot(width=0.7, size=1, fatten=1, outlier.shape = NA) +
    geom_point(size=0.5, alpha=0.5, shape=21, color='black') +
    geom_line(aes(group=id), colour="black", linetype="11") +
    theme_minimal_hgrid() +
    ylim(c(0,1)) +
    facet_grid(.~species) +
    scale_fill_manual(values=brewer.pal(n=4, name='Set2'))
  ggsave(paste0('./fig/word2vec/combined_',embeddings[[i]],'_hallmark_boxplot.png'), width=6, height=8)
}

######
## Run Stats
######

for (i in 1:length(species.short.names)) {
  for (j in 1:length(embeddings)) {
    sub.1 = subset(combined.stats.df, species == species.long.names[[i]]
                                      & embedding == embeddings[[j]]
                                      & type == 'Between Set')
    sub.2 = subset(combined.stats.df, species == species.long.names[[i]]
                   & embedding == embeddings[[j]]
                   & type == 'Within Set')
    x = wilcox.test(sub.1$mean.dist, sub.2$mean.dist, paired=TRUE)
    print("########################################################")
    print(paste0(species.long.names[[i]], ' - ', embeddings[[j]]))
    print(x)
  }
}





######
## Alternative visualization
######


sub.1 <- subset(combined.stats.df, type=='Between Set')
sub.2 <- subset(combined.stats.df, type=='Within Set')

combined.stats.jointype <- full_join(
  sub.1, sub.2, by=c('embedding'='embedding',
                     'species'='species',
                     'id'='id'), suffix=c('.between', '.within'))
combined.stats.jointype$diff.dist <- (
  combined.stats.jointype$mean.dist.between
  - combined.stats.jointype$mean.dist.within)

ggplot(combined.stats.jointype, aes(x=species, y = diff.dist, fill=species)) +
  facet_grid(.~embedding) +
  geom_boxplot()



