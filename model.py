import pickle
import pandas as pd
import numpy as np

import gc
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score

from matplotlib import pyplot as plt
import seaborn as sns


import warnings
warnings.filterwarnings('ignore')

df_train = pd.read_csv('dataframe_to_train_model.csv',index_col=0)

# Separating train and test data
proporcao_treino = 0.85
treino, teste = train_test_split(df_train, test_size=1-proporcao_treino)

X = treino.drop(['isFraud'], axis=1)
y = treino['isFraud']  

proporcao_treino = 0.75 
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=1 - proporcao_treino, random_state=42)

# Ajuste o scaler nos dados de treinamento e aplique a padronização aos dados de treinamento e validação
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

pd.value_counts(y_train)

def kfold_train(X, y, params, NFOLDS=3, proporcao_treino=0.8):
    folds = KFold(n_splits=NFOLDS)
    columns = X.columns
    splits = folds.split(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - proporcao_treino, random_state=42)

    y_preds = np.zeros(X_test.shape[0])
    y_oof = np.zeros(X.shape[0])
    score = 0

    feature_importances = pd.DataFrame()
    feature_importances['feature'] = columns

    for fold_n, (train_index, valid_index) in enumerate(splits):
        X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        dtrain = lgb.Dataset(X_train, label=y_train)
        dvalid = lgb.Dataset(X_valid, label=y_valid)

        clf = lgb.train(params, dtrain, 10000, valid_sets=[dtrain, dvalid])

        feature_importances[f'fold_{fold_n + 1}'] = clf.feature_importance()

        y_pred_valid = clf.predict(X_valid)
        y_oof[valid_index] = y_pred_valid
        print(f"Fold {fold_n + 1} | AUC: {roc_auc_score(y_valid, y_pred_valid)}")

        score += roc_auc_score(y_valid, y_pred_valid) / NFOLDS
        y_preds += clf.predict(X_test) / NFOLDS

        del X_train, X_valid, y_train, y_valid
        gc.collect()

    print(f"\nMean AUC = {score}")
    print(f"Out of folds AUC = {roc_auc_score(y, y_oof)}")

    return folds, feature_importances, clf

params = {'num_leaves': 546,
          'min_child_weight': 0.03454472573214212,
          'feature_fraction': 0.1797454081646243,
          'bagging_fraction': 0.2181193142567742,
          'min_data_in_leaf': 106,
          'objective': 'binary',
          'max_depth': -1,
          'learning_rate': 0.005883242363721497,
          "boosting_type": "gbdt",
          "bagging_seed": 11,
          "metric": 'auc',
          "verbosity": -1,
          'reg_alpha': 0.3299927210061127,
          'reg_lambda': 0.3885237330340494,
          'random_state': 42,
}

folds, feature_importances,clf = kfold_train(X, y, params)

train2 = treino[['isFraud','card1', 'TransactionDT', 'addr1', 'TransactionAmt', 'D10','V310', 'card2', 'P_emaildomain',
                    'D15', 'D1', 'V312', 'V315', 'V318', 'V285', 'card4', 'V314', 'card5', 'V313', 'V283',
                    'V130', 'V320', 'ProductCD']]

X = train2.drop(['isFraud'], axis=1)
y = train2['isFraud']  

folds, feature_importances,clf = kfold_train(X, y, params)

#treino, teste
model = pickle.load(open('model.pickle', 'rb'))

teste_df = teste[['isFraud','card1', 'TransactionDT', 'addr1', 'TransactionAmt', 'D10','V310', 'card2', 'P_emaildomain',
                    'D15', 'D1', 'V312', 'V315', 'V318', 'V285', 'card4', 'V314', 'card5', 'V313', 'V283',
                    'V130', 'V320', 'ProductCD']]

clientes = teste_df.drop(['isFraud'], axis=1).head(5)

prediction = model.predict(clientes)
print("prediction users", prediction)
print("prediction first user", round(prediction[1]))

if len(prediction) == len(clientes):
    # Adicione os prediction como uma nova coluna "score" no DataFrame
    clientes['score'] = prediction

    clientes['fraud'] = clientes['score'].apply(lambda x: 1 if x > 0.7 else 0)

clientes.head()    