# install these packages into environment (e.g. enter Pkg REPL and use 'add [PKG]').

module cn # begin module
using Printf, Statistics, Distributions, DelimitedFiles, CSV, DataFrames, IterTools, StatsBase, Chain, FStrings

# little function here 
function slicematrix(A::AbstractMatrix)
    return [A[i, :] for i in 1:size(A,1)]
end

# class
mutable struct Configuration 
    # all the input variables we take 
    id::Int # int something perhaps 
    configurations::Vector{Vector{Int64}} # some kind of array
    configuration_probabilities::Matrix{Float64} # some kind of array

    # all the attributes that can be defined  
    configuration::Vector{Int64}
    p::Float64
    id_neighbor::Vector{Int64}
    p_neighbor::Vector{Float64}
    transition::Bool
    len::Int64
    prob_targets
    prob_move
    targets 

    # the functions as well (which is a bit quirky)
    get_probability 
    get_configuration
    match_row 
    flip 
    flip_index 
    flip_indices
    normalize 
    hamming_neighbors 
    pid_neighbors
    p_move 
    move 

    # the main body 
    function Configuration(id, configurations, configuration_probabilities)
        self = new()
        self.id = id 
        self.transition = false
        self.prob_move = false 

        # functions to run on init 
        ## get probability function
        self.get_probability = function(configuration_probabilities)
            return configuration_probabilities[self.id]
        end 

        ## get configuration function 
        self.get_configuration = function(configurations)
            return configurations[self.id]
        end

        ## should be static 
        self.match_row = function(x, configurations) 
            return findfirst(isequal(x), configurations)
        end 
        
        self.configuration = self.get_configuration(configurations) 
        self.p = self.get_probability(configuration_probabilities)
        self.len = length(self.configuration)

        # works 
        self.flip = function(x)
            return x == 1 ? -1 : 1
        end 

        # works 
        self.flip_index = function(index)
            new_arr = copy(self.configuration)
            new_arr[index] = self.flip(new_arr[index])
            return new_arr 
        end 

        self.flip_indices = function(indices)
            new_arr = copy(self.configuration)
            for ind in indices 
                new_arr[ind] = self.flip(new_arr[ind])
            end 
            return new_arr 
        end 

        # normalize array 
        self.normalize = function(arr)
            return arr./sum(arr, dims = 1)
        end 

        # hamming distance 
        self.hamming_neighbors = function()
            hamming_list = []
            for (num, _) in enumerate(self.configuration)
                tmp_arr = self.flip_index(num)
                append!(hamming_list, [tmp_arr])
            end 
            return Vector(hamming_list)
        end 

        # transition probabilities 
        self.pid_neighbors = function(configurations, configuration_probabilities)
            
            # need to implement that it should not recompute 
            # would be nice to speed test to verify 
            if self.transition == true
                return self.id_neighbor, self.p_neighbor 
            end 

            # get the hamming array 
            hamming_array = self.hamming_neighbors()

            # get configuration ids, and configuration probabilities 
            self.id_neighbor = [self.match_row(x, configurations) for x in hamming_array] # works 
            self.p_neighbor = configuration_probabilities[self.id_neighbor]
            self.transition = true  

            # return 
            return self.id_neighbor, self.p_neighbor 
        end 

        self.p_move = function(configurations, configuration_probabilities, summary = true)
            # do not recompute if already done 
            _, p_neighbor = self.pid_neighbors(configurations, configuration_probabilities)
            # broadcasting Julia style 
            self.prob_move = 1 .- (self.p ./ (self.p .+ p_neighbor))
            # return array 
            if summary == false
                return self.prob_move
            # return mean 
            else 
                return mean(self.prob_move) 
            end 
        end 

        # we could save more stuff still 
        # to optimize the shit out of this 
        # i.e. if we same prob_targets then we just have to throw the dice 
        self.move = function(configurations, configuration_probabilities, n, conf_list = false)
            
            # this has to be sampled every time 
            if self.prob_move == false
                self.prob_move = self.p_move(configurations, configuration_probabilities, false)
            end 

            ## binary move 
            self.targets = rand(1:self.len, n)
            self.prob_targets = self.prob_move[self.targets]
            move_bin = self.prob_targets .>= rand(n)

            ## if no moves then return current configuration
            if !any(move_bin)
                return self 
            end 

            ## if n == 1 then we can just take the neighbor 
            if n == 1
                new_id = self.id_neighbor[self.targets][1]
                
                ## not super pretty 
                if conf_list != false 
                    x = findfirst(isequal(new_id), [x for (x, y) in conf_list])
                    if x isa Number 
                        return conf_list[x][2]
                    end 
                end 

                return Configuration(new_id, configurations, configuration_probabilities)
            
            ## if n > 1 move is not necessarily to a neighbor 
            else 
                feature_changes = [x for (x, y) in zip(self.targets, move_bin) if y]
                new_configuration = self.flip_indices(feature_changes)
                new_id = self.match_row(new_configuration, configurations)

                ## not super pretty 
                if conf_list != false 
                    x = findfirst(isequal(new_id), [x for (x, y) in conf_list])
                    if x isa Number 
                        return conf_list[x][2]
                    end 
                end 

                return Configuration(new_id, configurations, configuration_probabilities)
            end 
        end 
        return self 
    end 
end 
end # end module 
