# DOCUMENTATION ------------------------------------------------
'''
'''





# LOAD LIBRARIES------------------------------------------------

# Python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re



# LOAD DATASET -------------------------------------------------

# Directories
dir_data    = r'/home/ccirelli2/Desktop/GSU/Fall_2019/Project_2/output/lstm_model_11162019'
list_files  = os.listdir(dir_data)


# BUILD VIZUALIZATION ------------------------------------------

def get_list_loss(list_files):
    # Regex
    regex   = re.compile('[0-9][0-9]-[0-9].[0-9]{4}')

    # Create Dataframe
    df  = pd.DataFrame({})

    # List Results
    iteration   = []
    loss        = []
    # Iterate list
    for afile in list_files:
        if 'weights' in afile:
            search  = re.search(regex, afile)
            result  = search.group()
            # Split String (Iteration, Result)
            result_tuple = result.split('-')
            iteration.append(int(result_tuple[0]))
            loss.append(float(result_tuple[1]))

    # Create DataFrame
    df['iteration'] = iteration
    df['loss']      = loss
    df.sort_values(by='iteration', ascending=True, inplace=True)

    # Plot Results
    df_plot = df['loss'].plot(kind='line', use_index=False, grid=True, 
            legend=True, x='Epochs', y='Loss', fontsize=15)
    df_plot.set_title('LSTM MODEL - EPOCHS vs LOSS', fontsize=18)
    df_plot.set_xlabel('Epochs', fontsize=15)
    df_plot.set_ylabel('Loss', fontsize=15)
    plt.show()

    # Return Results

list_results = get_list_loss(list_files)








