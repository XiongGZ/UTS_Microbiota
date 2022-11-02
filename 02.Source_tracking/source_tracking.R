rm(list = ls())
gc()

args <- commandArgs(T)

library(FEAST)

metadata = Load_metadata(metadata_path = args[1])
count_matrix = Load_CountMatrix(CountMatrix_path = args[2])

FEAST_output = FEAST(C = count_matrix, metadata = metadata, different_sources_flag = 0, dir_path = args[3], outfile = args[4])
