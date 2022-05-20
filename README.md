# pitch-execution
Uses machine learning methods to determine the probability of execution for each pitch type.

## Version 0.1 (2022-05-20)

### Added PitchExecution.py, execute.ipynb, sale.ipynb, savant_data.csv, executed.csv 

PitchExecution: class that takes in executed.csv and determines the probability of execution for each pitch.
This is not robust as all of the data is unlabelled to begin and had execution had to be manually determined.

execute.ipynb: Notebook used to clean the data and determine the execution of each pitch. This was accomplished
by writing a function that determined quantitatively whether or not the pitch was executed. Roughly 44% of the
pitches in the dataset were executed which would seem correct.

sale.ipynb: A slightly unrelated past project that used neural networks and SVM classification models to
determine pitch type in the same savant_data.csv dataset. SHAP values were included to determine which features
were the most important predictors

savant_data.csv: The dataset of pitches from Chris Sale's 2019 season taken from BaseballSavant. This is the raw
dataset.

executed.csv: The cleaned dataset with pitch type, count, and whether it was executed. This is what was used for
the PitchExecution class.

The next step will be to generalize the class so that it can take in any BaseballSavant dataset, automatically
clean it up and automatically apply the Executed column. The PitchExecution class also must be tweaked as Sale's
pitches are hard coded into it so it will only work for him (or any pitcher that throws the same pitches).
It will also be used to create models for collegiate pitching that can be used to determine strengths and
weaknesses regarding certain pitches in each counts.
