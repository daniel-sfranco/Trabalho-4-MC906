import os
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def get_raw_data():
    path = os.path.dirname(os.path.abspath(__file__))
    train_df = pd.read_csv(os.path.join(path, 'train.csv'))
    test_df = pd.read_csv(os.path.join(path, 'test.csv'))
    return train_df, test_df


def clean_data(df):
    categoric = ['Chest pain type', 'EKG results', 'Slope of ST', 'Thallium', 'Number of vessels fluro']
    continuum = ['Age', 'BP', 'Cholesterol', 'Max HR', 'ST depression']

    df['Risk Cholesterol'] = 0
    df.loc[df['Cholesterol'] >= 200, 'Risk Cholesterol'] = 1
    df.loc[df['Cholesterol'] >= 240, 'Risk Cholesterol'] = 2

    df['Risk BP'] = 0
    df.loc[df['BP'] >= 120, 'Risk BP'] = 1
    df.loc[df['BP'] >= 140, 'Risk BP'] = 2

    df['Severe exertion syndrome'] = 0
    df.loc[(df['Chest pain type'] >= 3) & (df['Exercise angina'] == 1) & (df['ST depression'] >= 5), 'Severe exertion syndrome'] = 1

    df['Proportional HR'] = df['Max HR'] / (220 - df['Age'])

    categoric += ['Risk Cholesterol', 'Risk BP']
    continuum += ['Proportional HR']

    scaler = StandardScaler()
    df = pd.get_dummies(df, columns=categoric, dtype=int)
    df[continuum] = scaler.fit_transform(df[continuum])
    
    ids = df['id']
    df.drop(columns=['id'], inplace=True)

    if 'Heart Disease' in list(df.columns):
        df['Heart Disease'] = df['Heart Disease'].map({'Presence': 1, 'Absence': 0})

    return df, ids

def logistic_regression(X_train, y_train, X_valid, y_valid, max_iter=1000):
    lr_model = LogisticRegression(max_iter=max_iter)
    lr_model.fit(X_train, y_train)
    y_prob_val = lr_model.predict_proba(X_valid)[:, 1]
    auc_score = roc_auc_score(y_valid, y_prob_val)
    return auc_score


def random_forest(X_train, y_train, X_valid, y_valid, n_estimators=100):
    rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    rf_model.fit(X_train, y_train)
    y_prob_val = rf_model.predict_proba(X_valid)[:, 1]
    auc_score = roc_auc_score(y_valid, y_prob_val)
    return auc_score


def main():
    train_df, test_df = get_raw_data()
    train_df, _ = clean_data(train_df)
    test_df, test_ids = clean_data(test_df)
    X_train = train_df.drop('Heart Disease', axis=1)
    y_train = train_df['Heart Disease']
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    auc_lr_score = logistic_regression(X_train, y_train, X_valid, y_valid)
    print(f"Desempenho do Baseline (Regressão Logística):")
    print(f"ROC-AUC: {auc_lr_score:.4f}")

    auc_rf_score = random_forest(X_train, y_train, X_valid, y_valid)
    print(f"Desempenho do Baseline (Random Forest):")
    print(f"ROC-AUC: {auc_rf_score:.4f}")


if __name__ == "__main__":
    main()
