'''
VMP 2022-12-13: 

curated set used in article: 
python3 curation.py \
    -i ../data/raw/drh_20221019.csv \
    -om ../data/clean/ \
    -or ../data/reference/ \
    -nq 20 \
    -nn 5

can be run for other values of -nq and -nn.
see run_curation.sh for the grid we run over. 
'''

import curation_functions as cf # see here for details
import pandas as pd 
import argparse 

def main(infile, outpath_mpf, outpath_reference, number_questions, number_nan):
    
    # setup
    answer_column = 'answers'
    question_id_column = 'question_id'
    entry_id_column = 'entry_id'
    weight_column = 'weight'
    nan_column = 'has_answer'
    minimize_column = 'question_id'
    maximize_column = 'entry_id'
    minimize_value = 0

    # import & basic cleaning
    infile = '../data/raw/drh_20221019.csv'
    df = pd.read_csv(infile, low_memory=False) 
    ## only 'super' questions (where related_parent_q is NAN)
    d = df[df['related_parent_q'].isna()] 
    ## select the columns that we need 
    d = d[["entry_id", "entry_name", "related_q_id", "related_q", "answers"]] 
    ## rename for clarity 
    d = d.rename(columns = {'related_q_id': 'question_id',
                            'related_q': 'question_name'})
    ## replace with unknown (currently not used - treated as NAN)
    d.replace(
        {'answers': 
            {"Field doesn't know": 'Unknown', 
            "I don't know": 'Unknown'}},
        inplace = True)

    # reference documents (used for saving reference documents)
    entry_reference = d[['entry_id', 'entry_name']].drop_duplicates()
    question_reference = d[['question_id', 'question_name']].drop_duplicates()

    # recoding answer values
    ## coding of questions
    answer_coding = [('Yes', 1), ('No', -1), ('Unknown', 0)]
    ## default value (should be integer, but could be anything)
    default = 100
    ## run the function and recode 
    d[answer_column] = cf.recode_answers(d, answer_column, answer_coding, default)

    # remove questions with non-binary answers
    d = cf.remove_non_binary(d, answer_column, question_id_column, default)

    # create grid of entry_id/question_id
    # create entry_id/question_id weighted by answers for combinations with (Yes/No) answers
    question_entry_grid, question_fractions = cf.grid_and_fractions(d, question_id_column,
                                                                entry_id_column,
                                                                answer_column,
                                                                weight_column,
                                                                nan_column)

    # select the best N (depends on specification) questions
    best_questions = cf.subset_nbest(question_entry_grid, number_questions,
                                minimize_value, minimize_column, nan_column) 

    # take the maximum number of entries that satisy constraints (i.e. number NAN)
    best_entries = cf.max_constraints(best_questions, question_fractions, number_nan, 
                                minimize_value,
                                maximize_column, nan_column, entry_id_column, 
                                question_id_column, answer_column, weight_column)

    # for each entry_id/question_id create tuples of combinations and add to list
    # i.e. [(question_id, entry_id, answer, weight)]
    # if there are more than one answer to entry_id/question_id then we expand and weight
    weighted_combinations_list = cf.apply_combinations(best_entries, entry_id_column, 
                                                    question_id_column, answer_column, 
                                                    weight_column, number_questions)

    # only run below if it is actually possible to satisfy constraints
    if weighted_combinations_list: 
        # convert the list of combinations to dataframe (and .txt) for saving
        data_csv = cf.combinations_to_dataframe(weighted_combinations_list)

        # only those that are from group polls.
        # this would be better to do earlier.
        poll_data = df[['entry_id', 'poll']]
        poll_data = poll_data[poll_data['poll'].str.contains('Group')]
        poll_data = poll_data[['entry_id']].drop_duplicates()

        ## again, additional work because the current structure
        ## is sub-optimal
        data_csv = data_csv.merge(poll_data, on = 'entry_id', how = 'inner')
        data_csv = data_csv.sort_values('entry_id').reset_index(drop=True)
        data_txt = data_csv.drop(columns = 'entry_id')

        # save data 
        cf.save_data(best_entries, entry_reference, question_reference, 
                    data_csv, data_txt, entry_id_column, question_id_column,
                    number_questions, number_nan, outpath_reference, outpath_mpf)
    else: 
        print('No solutions exist for specified constraints')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--infile', required = True, type = str, help = 'path to input file (csv)')
    ap.add_argument('-om', '--outpath_mpf', required = True, type = str, help = 'path to output main folder')
    ap.add_argument('-or', '--outpath_reference', required = True, type = str, help = 'path to output reference folder')
    ap.add_argument('-nq', '--number_questions', required = True, type = int, help = 'number of best questions')
    ap.add_argument('-nn', '--number_nan', required = True, type = int, help = 'number of nan to tolerate per civ')
    args = vars(ap.parse_args())
    main(
        infile = args['infile'],
        outpath_mpf = args['outpath_mpf'],
        outpath_reference = args['outpath_reference'],
        number_questions = args['number_questions'],
        number_nan = args['number_nan'])