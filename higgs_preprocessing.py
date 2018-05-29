import numpy as np
import pandas as pd

# Remove columns
def filterCol(dataFram):
    data_cols = dataFram.columns.tolist()
    
    for cols in data_cols:
        
        # Drop ID
        if cols == 'EventId' or cols == 'KaggleSet' or cols=='KaggleWeight' or cols=='Weight':
            dataFram=dataFram.drop([cols], axis = 1)
            continue
        # Drop columns that have missing values 
        test_missing = (dataFram[cols] == -999.000)
        if test_missing.sum() > 0:
            dataFram = dataFram.drop([cols], axis = 1)
            
    return dataFram

#
# Functions from https://github.com/cbracher69/Kaggle-Higgs-Boson-Challenge/blob/master/Higgs%20Linear-Gaussian%20Model%20Archive.ipynb
#
#
    
# Preparation - Turn momenta, weights into logarithms, normalize non-angular data
def logarithmic_momentum(dataFram):
    # Replace momentum data with its logarithm
    # I will add a small offset to avoid trouble with (rare) zero entries

    cols =  cols = list(dataFram.columns)
  
    for column in cols:
        if ((column.count('_phi') + column.count('eta')) == 0):
            # Select momentum features only:
            if not(column=='Label' or column=='Weight'):
                dataFram[column+'_log'] = np.log(dataFram[column] + 1e-8)
                dataFram = dataFram.drop(column, axis = 1)
            
    return dataFram


def logarithmic_weights(dataFram):
    # For training set, split off scoring information, 
    # and add the logarithm of the weight as separate column.
    dataFram_outcome = dataFram['Label'].copy()
    # Remove target information from data set
    dataFram = dataFram.drop(['Label'], axis = 1)
    return pd.concat((dataFram, dataFram_outcome),axis=1)