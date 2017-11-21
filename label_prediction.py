# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 20:09:57 2017

@author: shagupta
"""

import pickle
import pandas as pd
import numpy as np
import itertools as it
from sklearn.cross_validation import KFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn import svm

class LabelPrediction:
    def __init__(self):
        self.train_df = pd.read_csv("train.csv")
        self.train_df.drop(['storeId','url'],axis=1,inplace=True)
        self.label = self.train_df.label
        self.length = len(self.train_df)
        self.test_df = pd.read_csv('test.csv')
        self.test_df.drop(['storeId','id','url'],axis=1,inplace=True)
        self.merged_df  = pd.concat([self.train_df,self.test_df])
        self.test_df.drop(self.test_df.iloc[0:0],axis=1,inplace=True)
        
    def process_bdc(self):
        breadcrumbs = self.merged_df.breadcrumbs
        
        #Splitting data by '>'
        breadcrumbs_df = breadcrumbs.str.strip().str.split(" > ",expand = True)
        #breadcrumbs_df = breadcrumbs_df.iloc[:,:7]
        bdc_df = breadcrumbs_df.dropna(how='all',axis=(0,1))
        
        #Merging All columns into one for analysing all possible tags and their counts
        bdc_df = pd.DataFrame(list(it.chain(*bdc_df.values)))
        
        #stripping extra space and converting them in to lower case
        bdc_df[0] = bdc_df[0].map(lambda x:x if type(x)!=str else x.strip().lower())
        
        #print(bdc_df[0].value_counts()[bdc_df[0].value_counts()>5000])
        '''
        54 number has been chosen based on the counts of the unique tags that were found.
        Initially i checked how many were tags are above gien threshold(5000) and then took those
        It comes out that counts was 54 so just took that'''
        bdc_col = list(bdc_df[0].value_counts().index[:54])

        '''
        Here i can not replacing nan's with anything thing.
        Column values that are null helps me in defining their categorial value
        '''
        bdc_df.replace('nan',np.NaN)
        
        '''
        Feature array is just a categorial representation of breadcrumbs of a line in dataset
        '''
        feature_array = np.zeros((self.merged_df.shape[0],len(bdc_col)))
        
        '''
        Taking all original values of breadcrumbs
        Based of these values i will check if the value in present in the top columns.
        If yes then i will set that value to be 1 else 0
        '''
        bdc_value_list = breadcrumbs_df.values.tolist()
        
        '''
        Fills the feature array
        '''
        feature_array = self.fill_feature(feature_array,bdc_value_list,bdc_col,'bdc')
        
        bdc_feat_df = pd.DataFrame(feature_array,columns=bdc_col)
        
        '''
        Droping the DF for memory cleaning
        '''
        bdc_df.drop(bdc_df.iloc[0:0],axis=1,inplace=True)
        breadcrumbs_df.drop(breadcrumbs_df.iloc[0:0],axis=1,inplace=True)
        
        print(bdc_feat_df.shape)
        return bdc_feat_df

    '''
    Same as Above bu the attribute threshold is top 50
    '''
    def process_attr(self):
        attr = self.merged_df.additionalAttributes
        #Splitting Attributes
        attr_df = attr.str.strip().str.split(";",expand = True)

        attr_new_df = attr_df.dropna(how='all',axis=(0,1))
        
        #Merging all column values into one
        attr_new_df = pd.DataFrame(list(it.chain(*attr_new_df.values)))
        attr_new_df[0] = attr_new_df[0].map(lambda x: x if type(x) != str else x.split('=')[0].strip().lower())
        
        #print(attr_new_df[0].value_counts().index[attr_new_df[0].value_counts()>7000])
        #taking top 50 attributes
        attr_col = list(attr_new_df[0].value_counts().index[:50])
        
        #Replacing nan values with np.NAN
        attr_df.replace('nan',np.NaN)

        feature_array = np.zeros((self.merged_df.shape[0],len(attr_col)))
        value_list = attr_df.values.tolist()
        #print(len(value_list))
        feature_array = self.fill_feature(feature_array,value_list,attr_col,'attr')
        
        #print(feature_array.shape)
        attr_feat_df = pd.DataFrame(feature_array,columns=attr_col)
        attr_df.drop(attr_df.iloc[0:0],axis=1,inplace=True)
        attr_new_df.drop(attr_new_df.iloc[0:0],axis=1,inplace=True)

        print(attr_feat_df.shape)
        return attr_feat_df
        
    def fill_feature(self,farray,values,cols,feature):
        for i, attributes in enumerate(values):
            for attr in attributes:
                if(attr == None or pd.isnull(attr)):
                    #values after this will be none or nan
                    break
                elif(feature == 'attr'):
                    new_attr = attr.split('=')[0].lower()
                    if new_attr in cols:
                        farray[i][cols.index(new_attr)] = 1
                        
                elif(feature == 'bdc'):
                    if attr.lower() in cols:
                        farray[i][cols.index(attr)] = 1
        
        return farray
    
    def concat_df(self,df1,df2):
        print("Concatination features...")
        self.merged_df = pd.concat([df1,df2],axis=1)
        '''
        Based on feature importance(Calculated usinng Random Forest) thresold criteria 0.001
        Following were the important features(in decesding order)
        imp_features = ['books', 'publisher', 'movies & tv', 'language', 'movies', 'studio', 'record label', 'title', 'release date', 'release date:', 'label:', 'performer', 'asin', 'movie genre', 'genre', 'upc', 'format', 'software format', 'music', 'average customer review', 'movie studio', 'store item number (dpci)', 'tcin, isbn-13', 'movies', 'music & books', 'upc:', 'cds & vinyl', 'publisher:', 'pages:', 'publication date:', 'books & media', 'publication date', 'edition', 'isbn', 'dimensions', 'drama', 'run time', 'unit weight', 'duration', '1.', 'amazon best sellers rank']
        '''
        
        self.train_df = pd.DataFrame(self.merged_df.head(self.length))
        self.test_df = pd.DataFrame(self.merged_df.iloc[self.length:])
        print(self.merged_df.shape)
        
    def get_features(self,data='Train'):
        self.concat_df(self.process_bdc(),self.process_attr())
        print(self.train_df.shape)
        print(self.test_df.shape)
        return self.train_df, self.test_df, self.label
        
    def get_X_train(self):
        return self.new_df
    
    def get_y_train(self):
        return self.train_df.label
    
    def get_X_test(self):
        return self.test_df
    
    def get_RF_classifier(self, X_train, y_train):
        print("Getting Classifier...")
        clf = RandomForestClassifier()
    
        # Choose some parameter combinations to try
        parameters = {'n_estimators': [4, 6, 9], 
                      'max_features': ['log2', 'sqrt','auto'], 
                      'criterion': ['entropy', 'gini'],
                      'max_depth': [2, 3, 5, 10], 
                      'min_samples_split': [2, 3, 5],
                      'min_samples_leaf': [1,5,8]
                     }
    
        # Type of scoring used to compare parameter combinations
        acc_scorer = make_scorer(accuracy_score)
    
        # Run the grid search
        grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)
        grid_obj = grid_obj.fit(X_train, y_train)
    
        # Set the clf to the best combination of parameters
        clf = grid_obj.best_estimator_
        print(clf)
    
        # Fit the best algorithm to the data. 
        #clf.fit(X_train, y_train)
        return clf

    def get_SVM(self):
        svc = svm.SVC(gamma=0.001, C=100.)
        return svc
    
    def split_data(self,X,Y,test_size):
        X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=test_size,random_state=23)
        return X_train, X_test, y_train, y_test

    def run_kfold(self,clf,X,Y,k=10):
        kf = KFold(X.shape[0],n_folds=k)
        outcome = []
        fold = 0
        
        for train_index, test_index in kf:
            fold += 1
            X_train, X_test = X.values[train_index], X.values[test_index]
            y_train, y_test = Y.values[train_index], Y.values[test_index]
            clf.fit(X_train, y_train)
            new_predictions = clf.predict(X_test)
            accuracy = accuracy_score(y_test, new_predictions)
            outcome.append(accuracy)
            print("Fold {0} accuracy: {1}".format(fold, accuracy))
            
        mean_outcome = np.mean(outcome)
        print("Mean Accuracy: {0}".format(mean_outcome))
        
if __name__ == "__main__":
    '''
    I have used two different classifiers
    1. SVM
    2. Random Forest
    
    Training data was divided in three parts Train(60), Cross Validation(20), Test(20)
    KFold Crossvalidation has been used for measuring accuracy performance(don on 80% of data)
    
    In the end SVM seems to have slightly better results than Random Forest.
    Difference is at second decimal place, so it is quite difficult to say wether
    the classification is linear or not. But it seems that it more linear
    
    IMPORTANT NOTE:-
    URL's has not been considered while training.
    
    To run the program just rum the main funciton. By Default SVM will run.
    If you want to rum RF then comment SVM and uncomment RF
    '''
    lp = LabelPrediction()
    #lp.processAttr()
    X, XT, Y = lp.get_features('Train')
    
    #Split Data in Train and Test
    X_train, X_test, y_train, y_test = lp.split_data(X,Y,0.2)
    
    features_imp = pd.DataFrame()
    features_imp['features'] =  X_train.columns
    
    X_train_new, X_cv, y_train_new, y_cv = lp.split_data(X_train,y_train,0.2)
    
    '''
    SVM Multi class classifier
    '''
    svc = lp.get_SVM()
    #Cross Validation
    lp.run_kfold(svc,X_train,y_train)
    svc.fit(X_train_new,y_train_new)
    svc_dump = pickle.dumps(svc)
    
    #Test Prediction SVM
    prediction_svc = svc.predict(X_test)
    accuracy_svc = accuracy_score(y_test, prediction_svc)
    print("Test Accuracy SVM: {0}".format(accuracy_svc))
    test_predictions = svc.predict(XT)
    test_df = pd.DataFrame(test_predictions)
    test_df.to_csv("test_output_svm.csv")
    
    '''
    Random Forest Classifier
    '''
    '''clf = lp.get_RF_classifier(X_train_new,y_train_new)    
    
    #Kfold run on Original Train Data
    lp.run_kfold(clf, X_train,y_train)
    
    clf.fit(X_train_new,y_train_new)

    #Dumping trained model
    rf_dump = pickle.dumps(clf)
    
    #Getting important features
    features_imp['importance'] = clf.feature_importances_
    features_imp.sort_values(by=['importance'], ascending=True, inplace=True)
    features_imp.set_index('features', inplace=True)
    features_imp.plot(kind='barh', figsize=(20, 20))
    
    #Test Predicting Random Forest
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print("Test Accuracy RF: {0}".format(accuracy))
    test_predictions = clf.predict(XT)
    test_df = pd.DataFrame(test_predictions)
    test_df.to_csv("test_output.csv")'''