import numpy as np 
import pandas as pd 
import random 

class Configuration: 
    def __init__(self, id, states, probabilities):
        self.id = id
        self.states = states
        self.probabilities = probabilities
        self.configuration = states[id]
        self.p = probabilities[id]
        self.len = len(self.configuration)

    def to_string(self):
        return "".join([str(x) if x == 1 else str(0) for x in self.configuration])

    @staticmethod
    def flip(x): 
        return -1 if x == 1 else 1 
    
    def flip_at_index(self, index): 
        new_arr = np.copy(self.configuration)
        new_arr[index] = self.flip(new_arr[index])
        return new_arr 
    
    def flip_at_indices(self, indices): 
        new_arr = np.copy(self.configuration)
        for ind in indices: 
            new_arr[ind] = self.flip(new_arr[ind])
        return new_arr 

    def hamming_neighbors(self): 
        hamming_lst = [self.flip_at_index(num) for num, _ in enumerate(self.configuration)]
        hamming_array = np.array(hamming_lst)
        return hamming_array 
 
    def id_and_prob_of_neighbors(self):
        hamming_array = self.hamming_neighbors()
        self.id_neighbor = np.array([np.where((self.states == i).all(1))[0][0] for i in hamming_array])
        self.p_neighbor = self.probabilities[self.id_neighbor]
        return self.id_neighbor, self.p_neighbor
    
    def p_move(self, summary=True):
        self.id_neighbor, self.p_neighbor = self.id_and_prob_of_neighbors()
        prob_moves = 1 - (self.p / (self.p + self.p_neighbor))
        return np.mean(prob_moves) if summary else prob_moves
    
    def move(self, N): 
        targets = random.sample(range(self.len), N)
        prob_move = self.p_move(summary = False)
        prob_targets = prob_move[targets]
        move_bin = prob_targets >= np.array([random.uniform(0, 1) for _ in range(N)])
        
        if not any(move_bin):
            return Configuration(self.id, self.states, self.probabilities)
        if N == 1: 
            new_id = self.id_neighbor[targets][0]
            return Configuration(new_id, self.states, self.probabilities)
        else: 
            feature_changes = [x for x, y in zip(targets, move_bin) if y]
            new_configuration = self.flip_at_indices(feature_changes)
            new_id = np.where(np.all(self.states == new_configuration,
                                    axis = 1))[0][0]
            return Configuration(new_id, self.states, self.probabilities)

    def hamming_distance(self, other): 
        x = self.configuration 
        y = other.configuration
        array_difference = np.not_equal(x, y)
        h_distance = np.sum(array_difference)
        return h_distance

    def answer_comparison(self, other, question_reference):  
        answers = pd.DataFrame({self.id: self.configuration, other.id: other.configuration})
        answers = pd.concat([question_reference, answers], axis = 1)
        return answers

    def overlap(self, other, question_reference):
        answers = self.answer_comparison(other, question_reference)
        return answers[answers[self.id] == answers[other.id]]

    def diverge(self, other, question_reference):  
        answers = self.answer_comparison(other, question_reference)
        answers_diverge = answers[answers[self.id] != answers[other.id]]
        return answers_diverge
    
    def transition_probs_to_neighbors(self, question_reference):
        neighbor_ids, neighbor_probs = self.id_and_prob_of_neighbors()
        df = pd.DataFrame({'config_id': neighbor_ids, 'config_prob': neighbor_probs})
        df = pd.concat([df, question_reference], axis=1)
        df[self.id] = self.configuration
        df['transition_prob'] = df['config_prob'] / df['config_prob'].sum()
        return df.sort_values('transition_prob', ascending=False)
