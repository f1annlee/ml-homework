import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import RandomizedSearchCV,cross_val_score,train_test_split
from sklearn.linear_model import RandomizedLasso,Ridge,Lasso,LinearRegression
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor,ExtraTreesRegressor,AdaBoostRegressor

def read_train_data():
    X_train=pd.read_csv('train_features.csv')
    y=pd.read_csv('train_salaries.csv')
    x=X_train
    x.index,y.index=x['jobId'],y['jobId']
    x['salary']=y['salary']
    return x,x['salary']

def convert_toDummy(x):
    # convert nominal/categorical data to numerical data
    vec = DictVectorizer(sparse=False, dtype=int)
    vec_x=vec.fit_transform(x.to_dict(orient = 'records'))
    names = vec.get_feature_names()
    df=pd.DataFrame(data=vec_x,columns=names)
    return df

def check_distribution(x):
    # check distribution of salary
    x['salary'].hist(bins=20)
    print('skewness',stats.skew(x['salary']))
    sns.distplot(x['salary'])
    stats.probplot(x['salary'],plot=plt)
    
    # check distribution of other numerical variables
    print('skewness',stats.skew(x['milesFromMetropolis']))
    sns.distplot(x['milesFromMetropolis'])
    stats.probplot(x['milesFromMetropolis'],plot=plt)
    
    print('skewness',stats.skew(x['yearsExperience']))
    sns.distplot(x['yearsExperience'])
    stats.probplot(x['yearsExperience'],plot=plt)
    return True

def preprocess_train_data(x):    
    # filter invalid salary data
    x=x[x['salary']>0]
    x.drop(['jobId'],axis=1,inplace=True)
    
    # check distribution of the numerical features and salary
    check_distribution(x)
    
    # convert categorical/nominal features to numerical dummy features
    df=convert_toDummy(x)
    
    # add a new feature 
    df['newGrad']=df['yearsExperience'].apply(lambda x: 1 if x==1 else 0)  
    g=sns.boxplot(x="jobType", y="salary", hue="newGrad", data=df, palette="Set3")

    return df

def scale_x(data):
    #min-max scaling
    scaler=MinMaxScaler()
    scaler.fit(data)
    x=scaler.transform(data)
    return x

def feature_selection(Xnew,Y):
    train_cols = Xnew.columns.tolist()
    rlasso = RandomizedLasso(alpha=0.005)
    rlasso.fit(Xnew, Y)
    print("features sorted by their socre:")
    featureRanks = sorted(zip(map(lambda x: round(x, 4), rlasso.scores_),
                              train_cols), reverse=True)
    print(featureRanks)
    selectedFeats = [feat[1] for feat in featureRanks if feat[0] > 0.01]
    return selectedFeats,featureRanks

def preprocess_data(df):
    df_numerical=convert_toDummy(df)
    data_x=scale_x(df_numerical)
    cols=df_numerical.columns.values.tolist()
    return data_x,cols

def train_model(data_x,data_y):
    # create training and testing sets
    X_train, X_test, y_train, y_test = \
    train_test_split(data_x,data_y, test_size=0.2, random_state=42)
    
    # model selection & tuning
    clf1 =LinearRegression()
    clf2 = Ridge(alpha = .5)
    treePar = {'splitter':['best', 'random'], 'max_depth':[3, 5, 10], 'min_samples_split':[3, 5, 10, 15, 20]}
    treeModel = DecisionTreeRegressor()
    clf3 = RandomizedSearchCV(treeModel, treePar, verbose=10, n_iter=20)
    clf4 = GradientBoostingRegressor(loss='huber', n_estimators=600, min_samples_split=10)
    ranfPar = {'n_estimators':[10, 20, 500, 100], 'min_samples_split':[3,5,10,20]}
    ranfModel = RandomForestRegressor()
    clf5 = RandomizedSearchCV(ranfModel, ranfPar, verbose=10, n_iter=10)
    clf6 = AdaBoostRegressor()
    clf7 = KNeighborsRegressor()
    clf8 = Ridge()
    clf9 = Lasso()
    clf10 = ExtraTreesRegressor()

    clfs=[clf1,clf2,clf3,clf4,clf5,clf6,clf7,clf8,clf9,clf10]
    for clf in clfs:
        clf.fit(X_train, y_train)
        try:
            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
        except:
            pass
        y_pred_train = clf.predict(X_train)
        
        scores = cross_val_score(regr, X_train, y_train, cv=10)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        
        print("Root Mean squared error of training: %.2f"
              % np.sqrt(mean_squared_error(y_train, y_pred_train)))
        print('Variance score: %.2f' % r2_score(y_train, y_pred_train))
    
        y_pred = clf.predict(X_test)
        print("Root Mean squared error of blind prediction: %.2f"
              % np.sqrt(mean_squared_error(y_test, y_pred)))
        print('Variance score: %.2f' % r2_score(y_test, y_pred))
    return True

def final_model(data_x,data_y):
    X_train, X_test, y_train, y_test = \
    train_test_split(data_x,data_y, test_size=0.2, random_state=42)
    
    regr = RandomForestRegressor(n_estimators=100,min_samples_split=35)
    regr.fit(X_train, y_train)
   
    scores = cross_val_score(regr, X_train, y_train, cv=10)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    y_pred_train=regr.predict(X_train)
    print("Root Mean squared error of training: %.2f"
              % np.sqrt(mean_squared_error(y_train, y_pred_train)))
    print('Variance score: %.2f' % r2_score(y_train, y_pred_train))
    y_pred=regr.predict(X_test)
    print("Root Mean squared error of blind prediction: %.2f"
          % np.sqrt(mean_squared_error(y_test, y_pred)))
    print('Variance score: %.2f' % r2_score(y_test, y_pred))
   
    return regr

def plot_feature_importance(regr,cols):
    importances = regr.feature_importances_
    std = np.std([tree.feature_importances_ for tree in regr.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    
    # Print the feature ranking
    print("Feature ranking:")
    for f in range(X.shape[1]):
        print("%d. feature %s (%f)" % (f + 1, cols[indices[f]], importances[indices[f]]))
    
    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.show()
    
    return True

if __name__ == "__main__":
    #section 1: read train data
    x,y=read_train_data()
    
    '''
    #section 2: explore data (this does not need to run in final prediction)    
    #read & preprocess training data, and select important features
    #commented out because not used in the final model
    
    # preprocess training data
    df_train=preprocess_train_data(x)
    # feature selection
    selectedFeats,featureRanks=feature_selection(df_train.drop('salary',axis=1),df_train['salary']) 
    '''
    
    #section 3: preprocess training data 
    selected_cols=['jobType','degree','industry','yearsExperience','milesFromMetropolis']
    X,cols=preprocess_data(x[selected_cols])
    
    '''
    #section 4: model selection and tuning (this does not need to run in final predction)
    #commented out because not used in the final model
    train_model function does not need to run in final prediction) 
    train_model(X,y.values)
    '''
    
    #section 5: final selected model
    regr=final_model(X,y.values)
    plot_feature_importance(regr,cols)
    '''
    Results:
        Accuracy: 0.74 (+/- 0.00)
        Root Mean squared error of training: 18.14
        Variance score: 0.78
        Root Mean squared error of blind prediction: 19.80
        Variance score: 0.74
    '''
    
    #section 6: predict 
    df_test=pd.read_csv('test_features.csv')
    X,cols=preprocess_data(df_test[selected_cols])
    y_predict=regr.predict(X)
    
    out=df_test[['jobId']]
    out['salary']=y_predict
    out.to_csv('test_salaries.csv',index=False)