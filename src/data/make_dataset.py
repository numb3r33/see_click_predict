# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

class Data:
    def __init__(self, train, test):
        self.train = train
        self.test = test
    
    def concat_data(self):
        self.data = pd.concat((self.train, self.test), axis=0)
        return self.data
    
    def round_location(self):
        self.data['latitude'] = self.data.latitude.map(np.round)
        self.data['longitude'] = self.data.longitude.map(np.round)
        
        return self.data
    
    def fill_missing_values(self, feature, value):
        self.data[feature] = self.data[feature].fillna(value)
        
        return self.data
        
    def encode_categorical_variable(self, feature):
        lbl = LabelEncoder()
        
        lbl.fit(self.data[feature])
        self.data[feature] = lbl.transform(self.data[feature])
        
        return self.data
        
    def get_train_test(self):
        mask = self.data.num_votes.notnull()
        
        train = self.data.loc[mask]
        test = self.data.loc[~mask]
        
        return train, test
    
    def one_hot_encode(self, feature):
        ohe = pd.get_dummies(self.data[feature])
        self.data = pd.concat((self.data, ohe), axis=1)
        
        return self.data
    
    def decompose_date(self):
        self.data['month'] = self.data.created_time.dt.month
        self.data['year'] = self.data.created_time.dt.year
        self.data['week'] = self.data.created_time.dt.dayofweek
        
        return self.data