# COGSCI23
include("configuration.jl")
using .cn, Printf, Statistics, Distributions, DelimitedFiles, CSV, DataFrames, IterTools, StatsBase, Chain, FStrings, Base.Threads

# load shit 
configuration_probabilities = readdlm("/home/vpoulsen/humanities-glass/data/analysis/configuration_probabilities.txt")
configurations = readdlm("/home/vpoulsen/humanities-glass/data/analysis/configurations.txt", Int)
## I need an array, rather than a matrix 
configurations = cn.slicematrix(configurations)

# load all maximum likelihood configurations 
entry_config_filename = "/home/vpoulsen/humanities-glass/data/analysis/entry_maxlikelihood.csv"
entry_maxlikelihood = DataFrame(CSV.File(entry_config_filename))
config_ids = @chain entry_maxlikelihood begin _.config_id end
unique_configs = unique(config_ids) # think right, but double check 
unique_configs = unique_configs .+ 1 # because of 0-indexing in python 
sample_list = []

for unique_config in unique_configs 
    ConfObj = cn.Configuration(unique_config, configurations, configuration_probabilities)
    p_move = ConfObj.p_move(configurations, configuration_probabilities)
    p_stay = 1 .- p_move 
    p_raw = ConfObj.p 
    push!(sample_list, [p_raw, p_stay])
end 

println("saving file")
d = DataFrame(
config_id = [x-1 for x in unique_configs],
config_prob = [x for (x, y) in sample_list],
remain_prob = [y for (x, y) in sample_list] 
)
CSV.write(f"/home/vpoulsen/humanities-glass/data/COGSCI23/evo_stability/maxlik_evo_stability.csv", d)