from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json
import pandas as pd

# Import json and make a small helper function to keep the code clean
def load_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
    
def dataset_log(search: GridSearchCV, X_test, y_test, pred_model, labels):
        
    best_model = search.best_estimator_
    best_params = search.best_params_
    
    y_pred = best_model.predict(X_test)
    
    # Use a single formatted string instead of incremental additions
    
    log_model = f"""
    MODEL {pred_model}:
    Best parameters: {best_params}
    Training score: {search.best_score_}
    Test score: {search.score(X_test, y_test)}
    
    Classification report:
    {classification_report(y_test, y_pred, labels=labels)}
    Accuracy: {accuracy_score(y_test, y_pred)}
    
    Confusion matrix:
    {confusion_matrix(y_test, y_pred, labels=labels)}
    
    Labels: {labels}
    """
    return log_model


def visualise_changes(clf, d, encoder=None, method = 'CE-OCL', CEs = None, CEs_ = None,
                      only_changes=False, exp = None, factual = None, scaler = None, F_coh = None,
                      encoding_dict = None):

    if method == 'CARLA':
        if factual is None:
            print('If the method used it DiCE, please specify the factual instance.')
            exit()
        if scaler is None:
            print('If the method used it CARLA, please specify the scaler.')
            exit()

        CEs[d['target']] = clf.predict(CEs)
        CEs = pd.concat([factual, CEs])

        # reverse scaling
        CEs_ = CEs.copy()
        scaled_xdata_inv = scaler.inverse_transform(CEs_[d['numerical']])
        CEs_.loc[:, d['numerical']] = scaled_xdata_inv

    elif method == 'DiCE':
        if exp is None:
            print('If the method used it DiCE, please specify exp.')
            exit()
        if factual is None:
            print('If the method used it DiCE, please specify the factual instance.')
            exit()
        if scaler is None:
            print('If the method used it DiCE, please specify the scaler.')
            exit()

        CEs = exp.cf_examples_list[0].final_cfs_df.iloc[:, :-1].reset_index(drop=True)
        CEs = pd.concat([factual, CEs])

        # reverse scaling
        CEs_ = CEs.copy()
        scaled_xdata_inv = scaler.inverse_transform(CEs_[d['numerical']])
        CEs_.loc[:, d['numerical']] = scaled_xdata_inv
        # CEs_ = scaler.inverse_transform(CEs)
        CEs_[d['target']] = clf.predict(CEs)
        
    elif method == 'CE-OCL':
        # remove scaled_distance from df
        # add target prediction
        # CEs_[d['target']] = clf.predict(CEs.drop('scaled_distance', axis=1))
        CEs_.loc['original', d['target']] = clf.predict(pd.DataFrame(CEs.drop('scaled_distance', axis=1).loc['original']).T)
        CEs_.loc['sol0':, d['target']] = abs(CEs_.loc['original', d['target']] - 1)
        CEs_.loc['sol0':, d['target']] = clf.predict([CEs.drop('scaled_distance', axis=1).iloc[1,:]])[0]
        
        # reverse scaling
        if scaler is not None: 
            CEs_[d['numerical']] = scaler.inverse_transform(CEs_[d['numerical']])


    # if we can specify an encoder, then we're dealing with one of the CARLA datasets
    if encoder is not None:
        df_dummies, df_orig = recreate_orig(CEs_, d, encoder)

        # make sure the index is right
        ix_names = ['original'] + ['sol' + str(i) for i in range(len(CEs.index))]
        ix = {i: ix_names[i] for i in range(len(CEs.index))}
        df_orig = df_orig.reset_index(drop=True).rename(index=ix)

    elif encoder is None:
        if F_coh is None:
            print('Please specify either encoder or F_coh.')
            exit()
            
        df_orig = pd.DataFrame()
        for v in d['numerical']+[d['target']]:
            df_orig[v] = CEs_[v].round(2)

        for v in F_coh:
            df_orig[v] = CEs_[F_coh[v]].apply(lambda row: reverse_dummies(row, CEs_, F_coh, v), axis=1)

        for c in df_orig.columns.difference(d['numerical']+[d['target']]):
            df_orig[c] = df_orig.apply(lambda row: value_names(row, c), axis=1)

        # df_orig = df_orig.round(2)
        if encoding_dict is not None: 
            for column_name in encoding_dict.keys():
                df_orig[column_name] = df_orig[column_name].map({str(idx): val for idx, val in enumerate(encoding_dict[column_name])})

    if only_changes:
        orig = df_orig[:1]
        df_orig = df_orig[1:].copy()
        df1 = pd.DataFrame()
        for c in df_orig.columns:
            df1[c] = df_orig.apply(lambda row: ce_change(row, df1, orig, c), axis=1)

        df = pd.concat([orig, df1])
        df = df.round(2)
        return df
    
    else: return df_orig
        
        
def reverse_dummies(row, CEs_, F_coh, v):
    if sum(row[F_coh[v]]) > 1:
        return 'TWO DUMMIES 1'
    elif sum(row[F_coh[v]]) == 0:
        return 'NO DUMMY 1'
    else:
        for c in F_coh[v]:
            if row[c] == 1:
                return c


def ce_change(row, df, orig, c):
    if row[c] == orig[c].iloc[0]:
        return '-'
    else:
        return row[c]


def value_names(row, c):
    if row[c] == '-':
        return row[c]
    elif row[c] == 'TWO DUMMIES 1':
        return row[c]
    elif row[c] == 'NO DUMMY 1':
        return row[c]
    else:
        return row[c].split('_')[-1]
                    