'''
VMP 2022-02-07: Produces table A1. 
'''

# read file 
import pandas as pd 
question_reference = pd.read_csv('../data/preprocessing/question_reference.csv')

# save as latex 
question_latex = question_reference.style.hide(axis='index').to_latex()
with open('../tables/table_A1_questions.txt', 'w') as f: 
    f.write(question_latex)