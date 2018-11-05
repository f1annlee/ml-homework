from statsmodels.stats.outliers_influence import variance_inflation_factor    
import numpy as np
import pandas as pd

def calculate_vif_(X, thresh=100):
    cols = X.columns
    variables = np.arange(X.shape[1])
    dropped=True
    while dropped:
        dropped=False
        c = X[cols[variables]].values
        vif = [variance_inflation_factor(c, ix) for ix in np.arange(c.shape[1])]
    
        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print('dropping \'' + X[cols[variables]].columns[maxloc] + '\' at index: ' + str(maxloc))
            variables = np.delete(variables, maxloc)
            dropped=True
    
    print('Remaining variables:')
    print(X.columns[variables])
    return X[cols[variables]],variables

def manually_select(df):
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    vif["features"] = df.columns
    x.iloc[0]
    x
    
def main():
    selected_cols=[col for col in df.columns.values.tolist() if col.find('companyId')==-1]
    df_selected=df[selected_cols]
    df_new,var=calculate_vif_(df_selected, thresh=100)
    return df_new,var

if __name__ == "__main__":
    main()