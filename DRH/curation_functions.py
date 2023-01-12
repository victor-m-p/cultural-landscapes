import pandas as pd 
import numpy as np
import itertools
import os 
from itertools import combinations, product
pd.options.mode.chained_assignment = None

# used when we 

def create_grid(df, c1, c2):
    # create lists for both variables
    l_c1 = df[c1].drop_duplicates().to_list()
    l_c2 = df[c2].drop_duplicates().to_list()
    # create grid and to dataframe 
    l_comb = list(itertools.product(l_c1, l_c2))
    d_comb = pd.DataFrame(l_comb, columns=[c1, c2])
    return d_comb

def fill_grid(df, c1, c2, fill):
    # create dataframe grid 
    d_comb = create_grid(df, c1, c2) 
    # merge outer and fillna with fill value
    df = df.merge(d_comb, on = [c1, c2], how = "outer").fillna(fill)
    return df 

# recodes answers (e.g. "yes" --> 1)
# returns a column 
def recode_answers(d, answer_column, answer_coding, default_value): 
    conditions = []
    choices = []
    # assign conditions and choices
    for condition, choice in answer_coding: 
        conditions.append((d[answer_column] == condition))
        choices.append(choice)
    return np.select(conditions, choices, default = default_value)

# remove questions with a non-binary answer in any record. 
def remove_non_binary(d, answer_column, question_id_column, remove_value):
    # locate questions with an answer of an illegal type 
    d_ = d[d[answer_column] == remove_value][[question_id_column]].drop_duplicates()
    # anti-join with the main dataframe
    d = d.merge(d_, on = question_id_column, how = 'outer', indicator = True)
    d = d[(d._merge=='left_only')].drop('_merge', axis = 1)
    return d 


def grid_and_fractions(d, question_id_column, entry_id_column, answer_column,
                       weight_column, nan_column):

    ## get fractions of Yes/No for answered question/entry combination
    # currently only treating Yes/No as answer (Field Doesn't Know / I Don't Know) is NAN. 
    answered_combinations = d[d[answer_column].isin([1, -1])] 
    # number of times the same answer (e.g. Yes) is present for combination of entry_id and question_id (numerator)
    numerator = answered_combinations.groupby([question_id_column, entry_id_column, answer_column]).size().reset_index(name="count")
    # number of times the same entry_id and question_id are present (denominator)
    denominator = answered_combinations.groupby([question_id_column, entry_id_column]).size().reset_index(name = 'count_total')
    # merge these two and obtain divide to obtain fraction of questions in Yes/No for combination
    weighted_combinations = numerator.merge(denominator, on = [question_id_column, entry_id_column], how = 'inner')
    weighted_combinations[weight_column] = weighted_combinations['count']/weighted_combinations['count_total']
    # select the columns that we need
    weighted_combinations = weighted_combinations[[question_id_column, entry_id_column, answer_column, weight_column]]

    ## fill grid  
    question_entry_combinations = answered_combinations[[question_id_column, entry_id_column]].drop_duplicates()
    question_entry_grid_nan = create_grid(question_entry_combinations, question_id_column, entry_id_column)
    # only the combinations that are not in answered (anti join)
    question_entry_grid_nan = question_entry_grid_nan.merge(question_entry_combinations, on = [question_id_column, entry_id_column], how = 'outer', indicator = True)
    question_entry_grid_nan = question_entry_grid_nan[(question_entry_grid_nan._merge=='left_only')].drop('_merge', axis = 1)

    ## get the full grid 
    # assign column with information about whether combinations are NAN or answered
    question_entry_combinations[nan_column] = 1 # does have Yes/No answer
    question_entry_grid_nan[nan_column] = 0 # does not have Yes/No answer
    # concat dataframes
    question_entry_grid_out = pd.concat([question_entry_grid_nan, question_entry_combinations])

    ## get the fractions
    # rename the dataframe because we need the other one returned
    question_entry_grid_weight = question_entry_grid_nan
    # all weights for NAN combinations are 1 (might be superfluous)
    question_entry_grid_weight[weight_column] = 1.0
    # rename nan_column to answer column
    question_entry_grid_weight = question_entry_grid_weight.rename(columns = {nan_column: answer_column})
    # concat dataframes
    question_fractions = pd.concat([weighted_combinations, question_entry_grid_weight]) 

    # return 
    return question_entry_grid_out, question_fractions

# get the N best records from the grid 
def subset_nbest(d, number_questions, minimize_value, minimize_column, nan_column):
    # get number of nan per entry 
    d_ = d.groupby([minimize_column, nan_column]).size().reset_index(name="count")
    # sort by the number of nan per entry 
    d_ = d_[d_[nan_column] == minimize_value].sort_values("count", ascending=True)  
    # take n best question_id
    d_ = d_[[minimize_column]].head(number_questions)
    # merge inner with the d_grid dataframe
    d = d.merge(d_, on = minimize_column, how = 'inner')
    return d

def max_constraints(best_questions, question_fractions, number_nan, 
                    minimize_value,
                    maximize_column, nan_column, entry_id_column, 
                    question_id_column, answer_column, weight_column): 
    # group by maximize column and nan and sort by this before 
    entry_nan = best_questions.groupby([maximize_column, nan_column]).size().reset_index(name = 'count')
    # fill grid with nan 
    entry_nan = fill_grid(entry_nan, maximize_column, nan_column, minimize_value)
    entry_nan = entry_nan[entry_nan[nan_column] == minimize_value].sort_values('count', ascending = True)
    # get the entry_id of the records with fewer nan than allowed
    entry_tolerated = entry_nan[entry_nan['count'] <= number_nan][[entry_id_column]]
    # merge back in on the selected questions
    entry_questions = best_questions.merge(entry_tolerated, on = entry_id_column, how = 'inner')
    # merge back in on the fraction for answers
    ### here we get the error ###
    entry_questions = entry_questions.merge(question_fractions, 
                                            on = [entry_id_column, question_id_column], 
                                            how = 'inner')
    # select the columns that we need 
    entry_questions = entry_questions[[entry_id_column, question_id_column, answer_column, weight_column]].drop_duplicates()
    # sort ascending (might not be necessary)
    entry_questions = entry_questions.sort_values([entry_id_column, question_id_column],
                                                  ascending = [True, True])
    return entry_questions # entry_nan: temporary

#### figure out what this does 
def entry_question_combinations(d, entry_id_column, 
                                question_id_column, 
                                answer_column, 
                                weight_column, N): 
    d['id'] = d.set_index([entry_id_column, question_id_column]).index.factorize()[0]
    dct = {}
    for _, row in d.iterrows():
        id = int(row['id']) 
        entry_id = int(row[entry_id_column])
        question_id = int(row[question_id_column])
        answer = int(row[answer_column])
        weight = row[weight_column]  
        dct.setdefault(id, []).append((entry_id, question_id, answer, weight))
    l = list(dct.values())
    return [p for c in combinations(l, N) for p in product(*c)]

def apply_combinations(best_entries, entry_id_column, 
                       question_id_column, answer_column, 
                       weight_column, number_questions):

    # run over all entry_id and get the weighted combinations of answers
    # i.e. [(entry_id, question_id, answer, weight), ...] 
    entry_combinations_lst = []
    for entry_id in best_entries[entry_id_column].unique():
        entry_id_uniq = best_entries[best_entries[entry_id_column] == entry_id]
        entry_combinations = entry_question_combinations(entry_id_uniq, entry_id_column, 
                                                         question_id_column, answer_column,
                                                         weight_column, number_questions)
        entry_combinations_lst.extend(entry_combinations)
        
    return entry_combinations_lst 

def combinations_to_dataframe(weighted_combinations):
    ## prepare dataframe
    vals = []
    cols = []
    for x in weighted_combinations: 
        subcols = []
        subvals = []
        w_ = 1
        for y in sorted(x): 
            s, n, a, w = y 
            w_ *= w 
            # values 
            subvals.append(a)
            # columns
            subcols.append(n)
        # values
        subvals.insert(0, s)
        subvals.append(w_)
        vals.append(subvals)
        # columns 
        subcols.insert(0, 'entry_id')
        subcols.append('weight')
        cols.append(subcols)
    ### make sure that format is correct
    if all((cols[i] == cols[i+1]) for i in range(len(cols)-1)):
        cols = cols[0]
    else: 
        print('inconsistent column ordering')
    data_csv = pd.DataFrame(vals, columns = cols)
    data_csv = data_csv.sort_values('entry_id').reset_index(drop=True)
    #data_txt = data_csv.drop(columns = 'entry_id')
    return data_csv

def save_data(best_entries, entry_reference, question_reference, 
             data_csv, data_txt, entry_id_column, question_id_column,
             number_questions, number_nan, reference_path, mpf_path):
    
    # get unique entry_id and question_id in our final data
    entries_unique = data_csv[[entry_id_column]].drop_duplicates()
    questions_unique = best_entries[[question_id_column]].drop_duplicates()
    # merge inner with the original reference to get names 
    entry_reference = entries_unique.merge(entry_reference, 
                                        on = entry_id_column, 
                                        how = 'inner').sort_values(entry_id_column)
    question_reference = questions_unique.merge(question_reference, 
                                                on = question_id_column,
                                                how = 'inner').sort_values(question_id_column)
    # information for writing 
    nrows = len(data_csv) 
    entries_unique = len(entries_unique)

    # outnames 
    identifier = f'questions_{number_questions}_maxna_{number_nan}_nrows_{nrows}_entries_{entries_unique}'
    direct_reference_outname = os.path.join(reference_path, f'direct_reference_{identifier}.csv')
    matrix_outname = os.path.join(mpf_path, f'matrix_{identifier}.txt')
    entry_reference_outname = os.path.join(reference_path, f'entry_reference_{identifier}.csv')
    question_reference_outname = os.path.join(reference_path, f'question_reference_{identifier}.csv')

    # save 
    data_csv.to_csv(direct_reference_outname, index = False)
    data_txt.to_csv(matrix_outname, sep = ' ', header = False, index = False)
    entry_reference.to_csv(entry_reference_outname, index = False)
    question_reference.to_csv(question_reference_outname, index = False)