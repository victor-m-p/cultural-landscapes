
<!-- ABOUT THE PROJECT -->
## Files
* ```configuration.py``` is a convenience class for working with configurations. 
* ```fun.py``` contains convenience functions used in both ```/preprocessing``` and ```/analysis```.
* ```fig_3_parameters.py``` creates Figure 3A and figure 3B (put together in ```inkscape```)
* ```fig_A1_dendrogram.py``` creates Figure A1 (appendix) and prepares ```fig_4_landscape.py```.
* ```fig_4_landscape.py``` creates Figure 4A (relies on ```prep_landscape.py```). 
* ```fig_4_methodist.py``` creates Figure 4B.
* ```fig_4_roman.py``` creates Figure 4C.
* ```table_A1_questions.py``` creates Table A1.
* ```table_A2_clades.py``` creates Table A2.
* ```table_A3_and_A4_entries.py``` creates Tables A3, A4.
* ```table_A5_annotations.py``` creates Table A5 (although slightly different).
* ```table_community_weight.py``` creates table of community weight (numbers, not table used in article)

## Regenerating everything
* ```regenerate.py PARAMS_FILE OUTPUT_DIR``` runs the whole pipeline (preprocessing, then all figures and tables) from a parameter file, in a temporary copy, without touching the committed outputs. With the pinned versions in ```requirements.txt``` and the published parameters it reproduces the committed tables exactly.
* ```compare_regenerated.py DIR_A DIR_B``` compares two regenerated analyses (parameters, top configurations, groups, tables, focal landscapes).
* See ```/regenerated``` for the published fit and the revised fit side by side.

## Contact
Victor M. Poulsen 
* Twitter: [@vic_moeller](https://twitter.com/vic_moeller) 
* GitHub: [@victor-m-p](https://github.com/victor-m-p)
* Mail: victormoeller@gmail.com
