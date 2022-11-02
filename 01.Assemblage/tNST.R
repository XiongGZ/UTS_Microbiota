rm(list = ls())
gc()

library(ape)
library(iCAMP)
library(NST)
library(dplyr)

abundance = t(read.csv("abundance.csv", header = T, check.names = F, row.names = 1, as.is = T, stringsAsFactors = F))

metadata = read.csv("metadata.csv", header = T, row.names = 1, as.is = T, stringsAsFactors = F)

samp.ck = NST::match.name(rn.list = list(comm = abundance, group = metadata))
abundance = samp.ck$comm
abundance = abundance[, colSums(abundance) > 0, drop = F]

metadata = samp.ck$group
metadata$group = "total"
group = metadata[, "total", drop = F]

# calculate tNST
tnst = tNST(comm = abundance, group = group, meta.group = NULL, meta.com = NULL,
            dist.method = "bray", abundance.weighted = T, rand = 1000,
            output.rand = T, nworker = 20, LB = F, null.model = "PF",
            between.group = F, SES = T, RC = T)

write.table(tnst$index.grp, file = "summary.csv", quote = F, sep = ",")
write.table(tnst$index.pair.grp, file = "pairwise.csv", quote = F, sep = ",")
