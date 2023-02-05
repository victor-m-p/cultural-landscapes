import pandas as pd 
import numpy as np
import random

class Configuration: 
    def __init__(self, 
                 id, 
                 states, 
                 probabilities):
        self.id = id
        self.configuration = self.get_configuration(states)
        self.p = self.get_probability(probabilities)
        self.len = len(self.configuration)
        self.transition = False 
        
    # consider: is entry something we NEED to have
    # or is entry something that we can add when needed?
   
    # as long dataframe with columns (configuration_id, question_id, question_name, question_name_drh) 
    def to_long():
        pass
    
    # as a wide dataframe where questions are columns
    def to_wide(): 
        pass 
    
    # as a matrix 
    def to_matrix(): 
        pass 
    
    # as a string 
    def to_string(self):
        return "".join([str(x) if x == 1 else str(0) for x in self.configuration])
    
    # depends on whether we require this property
    # otherwise we might include to possibility
    # to add this attribute 
    def get_probability(self, probabilities): 
        probability = probabilities[self.id]
        return probability

    # again depends on whether we require this 
    # otherwise we might include the possibility
    # to add this attribute
    def get_configuration(self, configurations): 
        configuration = configurations[self.id]
        return configuration

    # flip a bit (static)
    @staticmethod
    def flip(x): 
        return -1 if x == 1 else 1 
    
    # flip a specific index in array 
    def flip_index(self, index): 
        new_arr = np.copy(self.configuration)
        new_arr[index] = self.flip(new_arr[index]) # not sure whether this should be "flip" or "self.flip"
        return new_arr 

    def flip_indices(self, indices): 
        new_arr = np.copy(self.configuration)
        for ind in indices: 
            new_arr[ind] = self.flip(new_arr[ind])
        return new_arr 

    # sum array to one (static)
    @staticmethod
    def normalize(array): 
        array = array / array.min()
        array = array / array.sum()
        return array 

    # hamming neighbors
    def hamming_neighbors(self): # for now only immediate neighbors
        hamming_lst = [] 
        for num, _ in enumerate(self.configuration): 
            tmp_arr = self.flip_index(num)
            hamming_lst.append(tmp_arr)
        hamming_array = np.array(hamming_lst)
        return hamming_array 

    # flip probabilities (including or excluding self)
    # make sure that this checks as well whether it is already computed 
    def pid_neighbors(self, configurations, 
                       configuration_probabilities):
        
        # if already computed do not recompute 
        if self.transition:
            return self.id_neighbor, self.p_neighbor 
        
        # check whether it is already computed 
        hamming_array = self.hamming_neighbors()

        # get configuration ids, and configuration probabilities
        self.id_neighbor = np.array([np.where((configurations == i).all(1))[0][0] for i in hamming_array])
        self.p_neighbor = configuration_probabilities[self.id_neighbor]
        self.transition = True
        
        # return 
        return self.id_neighbor, self.p_neighbor         
    
    # p_move: following the new schema 
    def p_move(self, configurations,
               configuration_probabilities,
               summary = True):
        _, p_neighbor = self.pid_neighbors(configurations, 
                                                configuration_probabilities)
        prob_moves = 1-(self.p/(self.p + p_neighbor))
        # either return the array or the mean 
        if not summary: 
            return prob_moves 
        else: 
            return np.mean(prob_moves)

    ## the new methods ## 
    # basically probability remain, but foll
    def move(self, configurations, 
             configuration_probabilities,
             n): 
        
        # sample targets 
        targets = random.sample(range(self.len), n)
        # probability move 
        prob_move = self.p_move(configurations,
                                configuration_probabilities,
                                summary = False)
        prob_targets = prob_move[targets]
        # vectorized sample 
        move_bin = prob_targets >= np.array([random.uniform(0, 1) for _ in range(n)])
        
        # if there are no moves just return current
        if any(move_bin) == False:
            return Configuration(self.id, configurations, 
                                 configuration_probabilities)
        # if n == 1 move to the neighbor
        if n == 1: 
            new_id = self.id_neighbor[targets][0]
            return Configuration(new_id, configurations, 
                                 configuration_probabilities)
        # if n > 1 the move is not necessarily to a neighbor 
        else: 
            feature_changes = [x for x, y in zip(targets, move_bin) if y]
            new_configuration = self.flip_indices(feature_changes)
            new_id = np.where(np.all(configurations == new_configuration,
                                    axis = 1))[0][0]
            return Configuration(new_id, configurations,
                                 configuration_probabilities)

    ## functions for investigating stuff ##
    def hamming_distance(self, other): 
        x = self.configuration 
        y = other.configuration
        array_overlap = (x == y)
        h_distance = len(x) - sum(array_overlap)
        return h_distance 
  
    # used by overlap, diverge
    def answer_comparison(self, other, question_reference):  
        answers = pd.DataFrame([(x, y) for x, y in zip(self.configuration, other.configuration)], 
                                columns = [self.id, other.id])
        answers = pd.concat([question_reference, answers], axis = 1)
        return answers
    
    # overlap in answers between two configuration instances 
    def overlap(self, other, question_reference): 
        answers = self.answer_comparison(other, question_reference)
        answers_overlap = answers[answers[self.id] == answers[other.id]]
        return answers_overlap
    
    # difference in answers between two configuration instances 
    def diverge(self, other, question_reference):  
        answers = self.answer_comparison(other, question_reference)
        answers_nonoverlap = answers[answers[self.id] != answers[other.id]]
        return answers_nonoverlap 
 
    # getting more information about neighbor probs 
    def neighbor_probabilities(self, configurations, configuration_probabilities, 
                               question_reference, top_n = False):
        # if enforce move it is simple 
        config_ids, config_probs = self.pid_neighbors(configurations, configuration_probabilities)
        d = pd.DataFrame([(config_id, config_prob) for config_id, config_prob in zip(config_ids, config_probs)],
                        columns = ['config_id', 'config_prob'])
        d = pd.concat([d, question_reference], axis = 1)
        d[self.id] = self.configuration

        # common for both 
        d['transition_prob'] = d['config_prob']/d['config_prob'].sum()
        d = d.sort_values('transition_prob', ascending = False)
        # if we are only interested in the most probable n neighboring probabilities 
        if top_n: 
            d = d.head(top_n)
        return d 
  
    # naive path between two configuration
    def naive_path(other): 
        pass 

    # instantiate civilization class
    def to_civilization(x): 
        pass 

'''
# load documents
entry_configuration_master = pd.read_csv('../data/analysis/entry_configuration_master.csv')
configuration_probabilities = np.loadtxt('../data/analysis/configuration_probabilities.txt')
question_reference = pd.read_csv('../data/analysis/question_reference.csv')

# generate all states
n_nodes = 20
from fun import bin_states 
configurations = bin_states(n_nodes) 

### test some functionality ###
ConfObj = Configuration(769975, 
                        configurations, 
                        configuration_probabilities)

ConfObj.p_move(configurations,
               configuration_probabilities,
               summary = True)

# test the sampling to see 
n = 2
num_move = []
for i in range(1000): 
    nc = ConfObj.move(configurations,
                      configuration_probabilities,
                      n = n)
    if nc.id == ConfObj.id: 
        num_move.append(0)
    else: 
        num_move.append(1)

sum(num_move) # 131 reasonable. 
'''