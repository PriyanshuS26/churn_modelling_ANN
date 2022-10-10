# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 22:31:50 2022

@author: welcome
"""

# Keras Tuner
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch

#%%
# reading the data
df = pd.read_csv("https://raw.githubusercontent.com/krishnaik06/Keras-Tuner/main/Real_Combine.csv")
print(df.shape)
#%%
X = df.iloc[:,:-1] # Xs
y = df.iloc[:,-1] # Y
#%%
# imp - hyperparameter - 1. how many no. of HL we should have?
                        #2. no. of neurons we should have in HL?
                        #3. Learning Rate
def build_model(hp):
    model = keras.Sequential()
    for i in range(hp.Int('num_layers', 2, 20)):
        model.add(layers.Dense(units=hp.Int('units_' + str(i),
                                            min_value=32,
                                            max_value=512,
                                            step=32),
                               activation='relu'))
    model.add(layers.Dense(1, activation='linear'))  #o/p layer , linear for reg
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss='mean_absolute_error',
        metrics=['mean_absolute_error'])
    return model

#%%
tuner = RandomSearch(
    build_model,
    objective='val_mean_absolute_error',
    max_trials=5,
    executions_per_trial=3,   # 5*3=15 interation
    directory='project',
    project_name='Air Quality Index')

#%%
print(tuner.search_space_summary())
#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
#%%
tuner.search(X_train, y_train,
             epochs=5,
             validation_data=(X_test, y_test))
#%%
print(tuner.results_summary())
#%%
