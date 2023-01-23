#!/usr/bin/env Rscript
#
# Plot analytifcal bounds on information lost
#

library(HMMcopula)
library(ggplot2)

vals <- c()
a.minus.b <- seq(1,8000,by=10)
A <- 1
c <- 3.14 * sqrt(2/3)


for (i in 1:length(a.minus.b)) {
  k <- 2000
  n <- a.minus.b[i]
  vals <- c(vals,
    log(A * exp(c * sqrt(n)) / n^(3/4) * exp(-2*sqrt(n) / c * dilog(exp(-1*c*(k + 1/2) / (2*sqrt(n))))))
  )
}

plot.df <- data.frame(
  x = ixs,
  v = vals,
  series = 'cell2sentence'
)

vals <- c()
a.minus.b <- seq(1,8000,by=10)

for (i in 1:length(a.minus.b)) {
  k <- 2000
  n <- a.minus.b[i]
  vals <- c(vals,
            log(k) * n
  )
}

plot.df <- rbind(
  plot.df, data.frame(
    x = ixs,
    v = vals,
    series = 'boolean'
  )
)

ggplot(plot.df, aes(x=x, y=v, color=series)) + 
  geom_line()

