#!/usr/bin/env Rscript
## Dependecies.R
##
# Small utility script to consolidate the requirements for data analysis
# and ensure easy installation for replication and extension of findings.
#
# Note that on macOS, additional dependencies are required, including
# - Xcode command line utilities
# - `brew install openssl curl libgit2`
#
# @author Rahul Dhodapkar <rahul.dhodapkar@yale.edu>
# @version 2021.02.10
#
# Last tested on 2022.01.28
#

## set local CRAN mirror
local({r <- getOption("repos")
       r["CRAN"] <- "http://cran.r-project.org" 
       options(repos=r)
})


##############################################
# INSTALL FROM CRAN
##############################################

install.packages('word2vec')
install.packages('cccd')
install.packages('igraph')
install.packages('ade4')
install.packages('reticulate')
install.packages('igraph')
install.packages('cccd')
install.packages('pdfCluster')
install.packages('stringr')
install.packages('dplyr')

##############################################
# INSTALL FROM BIOCONDUCTOR
##############################################

if(!requireNamespace("BiocManager", quietly = TRUE)) {
 install.packages("BiocManager") 
}

BiocManager::install("splatter")

##############################################
# INSTALL FROM GITHUB
##############################################

if (!"devtools" %in% installed.packages()[,1]) {
    install.packages("devtools")
}


