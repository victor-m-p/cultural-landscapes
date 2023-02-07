'''
functions used in 'curation.py' for preparing our curated subset of the DRH. 
VMP 2023-02-05: added docstrings and refactored with ChatGPT
'''

import pandas as pd 
import numpy as np
import itertools
import os 
from itertools import combinations, product
pd.options.mode.chained_assignment = None

def remove_non_binary(df, answer_column, question_id_column, remove_value):
    """remove all questions with non-binary answers to any question

    Args:
        d (pd.DataFrame): dataframe with columns answer_column, question_id_column
        answer_column (string): column name
        question_id_column (string): column name
        remove_value (int): remove when this value appears in answer_column

    Returns:
        pd.DataFrame: dataframe with questions that have non-binary answers removed
    """
    question_ids = df[df[answer_column] == remove_value][question_id_column].drop_duplicates()
    df = df[~df[question_id_column].isin(question_ids)]
    return df

def grid_and_fractions(df, question_id_column, entry_id_column, 
                       answer_column, weight_column, nan_column):
    """question/entry pairs and whether they have (binary) answers

    Args:
        d (pd.DataFrame): dataframe
        question_id_column (string): column name in d
        entry_id_column (string): column name in d
        answer_column (string): column name in d
        weight_column (string): column name to be created
        nan_column (string): column name to be created 

    Returns:
        pd.DataFrame: two dataframes;
            (1) all question/entry combinations and whether they have (1, -1) answer
            (2) question/entry/answer, and weight (i.e., partial records)
    """
    answered = df[df[answer_column].isin([1, -1])] 
    answered_combinations = answered.groupby([question_id_column, entry_id_column])
    
    numerator = answered_combinations[answer_column].value_counts().reset_index(name="count")
    denominator = answered_combinations.size().reset_index(name='count_total')
    weighted_combinations = numerator.merge(denominator, on=[question_id_column, entry_id_column], how='inner')
    weighted_combinations[weight_column] = weighted_combinations['count'] / weighted_combinations['count_total']
    weighted_combinations = weighted_combinations[[question_id_column, entry_id_column, answer_column, weight_column]]
    
    question_entry_combinations = answered[[question_id_column, entry_id_column]].drop_duplicates()
    
    question_entry_grid = pd.DataFrame(list(itertools.product(question_entry_combinations[question_id_column].drop_duplicates(), 
                                                                  question_entry_combinations[entry_id_column].drop_duplicates())), 
                                           columns=[question_id_column, entry_id_column])
    question_entry_grid = question_entry_grid.merge(question_entry_combinations,
                                                    on = [question_id_column, entry_id_column], 
                                                    how = 'outer', indicator = True).query("_merge == 'left_only'").drop('_merge', axis=1)

    question_entry_combinations[nan_column] = 1
    question_entry_grid[nan_column] = 0
    question_entry_grid_out = pd.concat([question_entry_grid, question_entry_combinations])
    
    question_entry_grid_weight = question_entry_grid.copy()
    question_entry_grid_weight[weight_column] = 1.0
    question_entry_grid_weight = question_entry_grid_weight.rename(columns={nan_column: answer_column})
    question_fractions = pd.concat([weighted_combinations, question_entry_grid_weight]) 
    return question_entry_grid_out, question_fractions

def subset_nbest(d, number_questions, minimize_value, question_id_column, nan_column):
    """subset the N (number_questions) best (most complete) questions

    Args:
        d (pd.DataFrame): ...
        number_questions (int): number of best (most complete) questions to subset
        minimize_value (int): in our case we use 0 for NAN
        question_id_column (string): in our case question_id_drh
        nan_column (string): in our case has_answer (where 0 is NAN)

    Returns:
        pd.DataFrame: dataframe with columns question_id_drh, entry_id, has_answer for 20 best questions only
    """
    d_ = (d.groupby([question_id_column, nan_column])
           .size()
           .reset_index(name="count")
           .query(f"{nan_column} == {minimize_value}")
           .sort_values("count", ascending=True)
           .head(number_questions))
    return d[d[question_id_column].isin(d_[question_id_column])]

def max_constraints(best_questions, question_fractions, number_nan, 
                    minimize_value, entry_id_column, nan_column, 
                    question_id_column, answer_column, weight_column): 
    """subset all entries that have less than number_nan NAN values for the best questions

    Args:
        best_questions (pd.DataFrame): dataframe with entry_id_column, nan_column
        question_fractions (pd.DataFrame): dataframe with entry_id_column, question_id_column
        number_nan (int): number of NAN (in our case 5) allowed
        minimize_value (int): in our case 0 is NAN
        entry_id_column (string): in our case entry_id (which we want to maximize)
        nan_column (string): in our case has_answer (where NAN is coded as 0)
        question_id_column (string): in our case question_id_drh
        answer_column (string): in our case answers
        weight_column (string): in our case weight

    Returns:
        pd.DataFrame: dataframe with entries <= number_nan subsetted 
    """
    entry_nan = best_questions.groupby([entry_id_column, nan_column]).size().reset_index(name='count')
    entry_nan = entry_nan.merge(pd.DataFrame(list(itertools.product(
        entry_nan[entry_id_column].drop_duplicates(), 
        entry_nan[nan_column].drop_duplicates())), 
        columns=[entry_id_column, nan_column]), 
        on=[entry_id_column, nan_column], how="outer").fillna(minimize_value)

    entry_nan = entry_nan[entry_nan[nan_column] == minimize_value].sort_values('count', ascending=True)
    entry_tolerated = entry_nan[entry_nan['count'] <= number_nan][[entry_id_column]]
    entry_questions = best_questions.merge(entry_tolerated, on=entry_id_column, how='inner')
    entry_questions = entry_questions.merge(question_fractions, 
                                            on=[entry_id_column, question_id_column], 
                                            how='inner')

    return entry_questions[[entry_id_column, question_id_column, answer_column, weight_column]]\
        .drop_duplicates().sort_values([entry_id_column, question_id_column], ascending=[True, True])

def apply_combinations(best_entries, entry_id_column, 
                       question_id_column, answer_column, 
                       weight_column, N):
    """get entry/question weighted combinations as list of tuple

    Args:
        best_entries (pd.DataFrame): ...
        entry_id_column (string): column name (in our case entry_id)
        question_id_column (string): column name (in our case question_id_drh)
        answer_column (string): column name (in our case answers)
        weight_column (string): column name (in our case weight)
        N (int): number of questions (in our case 20)

    Returns:
        lst: list of (nested) tuples entry/question/answer/weight
    """
    entry_combinations_lst = []
    for entry_id in best_entries[entry_id_column].unique():
        entry = best_entries[best_entries[entry_id_column] == entry_id]
        entry['id'] = entry.set_index([entry_id_column, question_id_column]).index.factorize()[0]
        groups = entry.groupby('id')[[entry_id_column, question_id_column, answer_column, weight_column]]\
                    .apply(lambda x: x.values.tolist()).tolist()
        entry_combinations = [p for c in combinations(groups, N) for p in product(*c)]
        entry_combinations_lst.extend(entry_combinations)
    return entry_combinations_lst


def to_dataframe(weighted_combinations):
    """list of tuple to dataframe

    Args:
        weighted_combinations (lst): list of weighted entry/question combinations

    Returns:
        pd.DataFrame: expanded (matrix-like) dataframe with questions as columns and entry_id, weight
    """
    vals = []
    cols = []
    for x in weighted_combinations: 
        subcols = []
        subvals = []
        weight_ = 1
        for y in sorted(x): 
            entry, question, answer, weight = y 
            weight_ *= weight 
            subvals.append(int(answer))
            subcols.append(int(question))
        subvals.insert(0, int(entry))
        subvals.append(weight_)
        vals.append(subvals)
        subcols.insert(0, 'entry_id')
        subcols.append('weight')
        cols.append(subcols)
    if all((cols[i] == cols[i+1]) for i in range(len(cols)-1)):
        cols = cols[0]
    else: 
        print('inconsistent column ordering')
    data_csv = pd.DataFrame(vals, columns = cols)
    data_csv = data_csv.sort_values('entry_id').reset_index(drop=True)
    return data_csv

def save_output(best_entries, entry_reference_df, question_reference_df, 
                direct_reference_df, matrix_data, 
                entry_id_column, question_id_column, num_questions, num_nan, 
                reference_folder, mpf_folder):
    """save various outputs

    Args:
        best_entries (pd.DataFrame): ...
        entry_reference_df (pd.DataFrame): ...
        question_reference_df (pd.DataFrame): ...
        direct_reference_df (pd.DataFrame): ...
        matrix_data (pd.DataFrame): ...
        entry_id_column (string): column name
        question_id_column (string): column name
        num_questions (int): number questions (20)
        num_nan (int): number NAN (5)
        reference_folder (string): path
        mpf_folder (string): path
    """
    unique_entries = direct_reference_df[[entry_id_column]].drop_duplicates()
    unique_questions = best_entries[[question_id_column]].drop_duplicates()
    entry_reference_df = unique_entries.merge(entry_reference_df, on=entry_id_column, how='inner') \
                                      .sort_values(entry_id_column)
    question_reference_df = unique_questions.merge(question_reference_df, on=question_id_column, how='inner') \
                                            .sort_values(question_id_column)

    num_rows = len(direct_reference_df)
    num_unique_entries = len(unique_entries)
    identifier = f'questions_{num_questions}_maxna_{num_nan}_rows_{num_rows}_entries_{num_unique_entries}'
    direct_reference_file = os.path.join(reference_folder, f'direct_reference_{identifier}.csv')
    matrix_file = os.path.join(mpf_folder, f'matrix_{identifier}.txt')
    entry_reference_file = os.path.join(reference_folder, f'entry_reference_{identifier}.csv')
    question_reference_file = os.path.join(reference_folder, f'question_reference_{identifier}.csv')

    direct_reference_df.to_csv(direct_reference_file, index=False)
    matrix_data.to_csv(matrix_file, sep=' ', header=False, index=False)
    entry_reference_df.to_csv(entry_reference_file, index=False)
    question_reference_df.to_csv(question_reference_file, index=False)