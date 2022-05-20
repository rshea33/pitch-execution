import pandas as pd

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

import warnings

warnings.filterwarnings("ignore")

# This first version took data off of BaseballSavant and then was manually cleaned.
# A function was created that quantitatively determined the execution of each pitch
# as this was not labelled in the data. The function can be found in execute.ipynb,
# the origional dataset can be found in savant_data.csv, and the cleaned dataset
# can be found in executed.csv.
# Future versions will be able to be run on the BaseballSavant dataset automatically
# with a custom version used for college pitching.

class PitchExecution:
    """
    A class for a pitch execution probability model.

    Attributes
    ----------
    df : Pandas DataFrame with the 'pitch_type', 'count', and 'Executed' columns
    model : The ML model used to predict the probability of a pitch being executed
    x : The predictors of the ML model (count and pitch type)
    y : The target of the ML model (whether the pitch was executed)
    x_train : The training predictors of the ML model
    x_test : The testing predictors of the ML model
    y_train : The training target of the ML model
    y_test : The testing target of the ML model
    predictions : The predictions of the ML model

    Methods
    -------
    score()
        Returns the overall accuracy of the model between 0 and 1

    conf_matrix()
        Returns the confusion matrix of the model predictions

    report()
        Returns the classification report of the model predictions

    predict_prob(pitch_type, count)
        Returns the probability of a pitch being executed given the pitch type and count
    
    predict_all_pitches()
        Prints the probabilities of all different pitches in all counts

    """

    def __init__(self, df, model_type='svm', test_size=0.2):
        """
        Constructs a PitchExecution object.

        Parameters
        ----------
        df : Pandas DataFrame
         - All of the different pitches

        model_type : string (default of 'svm')
         - 'svm' : scikit-learn's SVC Model
         - 'logistic_regression' : scikit-learn's Logistic Regression Model
         - 'tree' : scikit-learn's Decision Tree Classifier
         - 'random_forest' : scikit-learn's Random Forest Classifier
         - 'neural_network' : scikit-learn's Neural Network Classifier

        test_size : float (default of 0.2)
          - used to prevent overfitting in the models during the train 
            test split

         """

        self.df = df

        if model_type == 'svm':
            self.model = SVC(probability=True)
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression()
        elif model_type == 'tree':
            self.model = DecisionTreeClassifier()
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier()
        elif model_type == 'neural_network':
            self.model = MLPClassifier()
        else:
            raise ValueError('model_type must be one of: svm, logistic_regression, tree, random_forest, neural_network')

        # Train-Test Split

        self.x = df.drop('Executed', axis=1)
        self.y = df['Executed']

        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(self.x, self.y, test_size=test_size)

        # Fit the model and save the predictions
        
        self.model.fit(self.x_train, self.y_train)
        self.predictions = self.model.predict(self.x_test)


    def __str__(self):
        """Returns a string representation of the model."""
        return f'Model type: {self.model}'

    
    def score(self):
        """Returns the overall prediction score of the model."""
        return accuracy_score(self.predictions, self.y_test)


    def conf_matrix(self):
        """Returns the confusion matrix of the model predictions."""
        return confusion_matrix(self.predictions, self.y_test)


    def report(self):
        """Returns the classification report of the model predictions."""
        return classification_report(self.predictions, self.y_test)


    def predict_prob(self, pitch_type, count):
        """
        Returns the probability of a certain pitch being executed.

        Parameters
        ----------
        pitch_type : string
            The type of pitch to predict

        count : int
            The number of pitches of that type to predict

        Returns
        -------
        float - The probability of the pitch being executed
        """

        def pitch_idx(pitch_type): # TODO: Make this more general to each pitcher
            """Returns the index of the pitch type to be read by the model."""
            if pitch_type == 'FF':
                return 0
            elif pitch_type == 'SI':
                return 1
            elif pitch_type == 'SL':
                return 2
            elif pitch_type == 'CH':
                return 3
            else:
                raise ValueError('pitch_type must be one of: FF, SI, SL, CH')

        def count_idx(count):
            """Returns the index of the count to be read by the model."""
            if count == '0-0':
                count = 0
                return count
            elif count == '0-1':
                count = 1
                return count
            elif count == '0-2':
                count = 2
                return count
            elif count == '1-0':
                count = 3
                return count
            elif count == '1-1':
                count = 4
                return count
            elif count == '1-2':
                count = 5
                return count
            elif count == '2-0':
                count = 6
                return count
            elif count == '2-1':
                count = 7
                return count
            elif count == '2-2':
                count = 8
                return count
            elif count == '3-0':
                count = 9
                return count
            elif count == '3-1':
                count = 10
                return count
            elif count == '3-2':
                count = 11
                return count
            else:
                raise ValueError('Count must be one of: 0-0, 0-1, 0-2, 1-0, 1-1, 1-2, 2-0, 2-1, 2-2, 3-0, 3-1, 3-2')

        return self.model.predict_proba([[pitch_idx(pitch_type),
                                          count_idx(count)]])[0][1]


    def predict_all_pitches(self):
        """Prints the probabilities of all different pitches in all counts."""

        fb, si, sl, ch = 'FF', 'SI', 'SL', 'CH' # TODO: Change this to be more general

        oh_oh = '0-0'
        oh_one = '0-1'
        oh_two = '0-2'
        one_oh = '1-0'
        one_one = '1-1'
        one_two = '1-2'
        two_oh = '2-0'
        two_one = '2-1'
        two_two = '2-2'
        three_oh = '3-0'
        three_one = '3-1'
        three_two = '3-2'

        pitches = [fb, si, sl, ch]
        counts = [oh_oh, oh_one, oh_two, one_oh, one_one, one_two, two_oh,
                  two_one, two_two, three_oh, three_one, three_two]

        for count in counts:
            print(f'Count: {count}\n')
            average_execution_prob = 0
            for pitch in pitches:

                print(f'{pitch} {count}: {self.predict_prob(pitch, count)}')
                average_execution_prob += self.predict_prob(pitch, count)
            average_execution_prob /= 4
            print(f'Average Execution Probability in {count}: {average_execution_prob}')
            print('\n\n')


    

    
        
        



def main():

    df = pd.read_csv('executed.csv')
    df = df.drop(['Unnamed: 0'], axis=1)

    sale = PitchExecution(df, model_type='neural_network', test_size=0.2)
    print(sale.predict_all_pitches())






if __name__ == '__main__':
    main()
