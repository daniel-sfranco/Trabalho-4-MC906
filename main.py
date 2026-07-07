import os
import pandas as pd

def get_data():
    path = os.path.dirname(os.path.abspath(__file__))
    train_df = pd.read_csv(os.path.join(path, 'train.csv'))
    test_df = pd.read_csv(os.path.join(path, 'test.csv'))
    return train_df, test_df


def main():
    train_df, test_df = get_data()
    print(train_df.info())
    print(test_df.info())

if __name__ == "__main__":
    main()