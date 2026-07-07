import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def get_raw_data():
    path = os.path.dirname(os.path.abspath(__file__))
    train_df = pd.read_csv(os.path.join(path, 'train.csv'))
    test_df = pd.read_csv(os.path.join(path, 'test.csv'))
    return train_df, test_df


def clean_data(df):
    df['Risk Cholesterol'] = 0
    df.loc[df['Cholesterol'] >= 200, 'Risk Cholesterol'] = 1
    df.loc[df['Cholesterol'] >= 240, 'Risk Cholesterol'] = 2

    df['Risk BP'] = 0
    df.loc[df['BP'] >= 120, 'Risk BP'] = 1
    df.loc[df['BP'] >= 140, 'Risk BP'] = 2

    df['Severe exertion syndrome'] = 0
    df.loc[(df['Chest pain type'] >= 3) & (df['Exercise angina'] == 1) & (df['ST depression'] >= 5), 'Severe exertion syndrome'] = 1

    df['Proportional HR'] = df['Max HR'] / (220 - df['Age'])

    categoric = ['Chest pain type', 'EKG results', 'Slope of ST', 'Thallium', 'Number of vessels fluro', 'Risk Cholesterol', 'Risk BP']
    continuum = ['Age', 'BP', 'Cholesterol', 'Max HR', 'ST depression', 'Proportional HR']

    scaler = StandardScaler()
    df = pd.get_dummies(df, columns=categoric, dtype=int)
    df[continuum] = scaler.fit_transform(df[continuum])
    
    ids = df['id']
    df.drop(columns=['id'], inplace=True)

    if 'Heart Disease' in list(df.columns):
        df['Heart Disease'] = df['Heart Disease'].map({'Presence': 1, 'Absence': 0})

    return df, ids


def main():
    train_df, test_df = get_raw_data()
    train_df, _ = clean_data(train_df)
    test_df, test_ids = clean_data(test_df)
    print(train_df.info())
    print(test_df.info())

if __name__ == "__main__":
    main()
