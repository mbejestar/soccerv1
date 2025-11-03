
"""Train a league-agnostic 1X2 classifier and export as model.pkl"""
import argparse, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, log_loss
from sklearn.impute import SimpleImputer

def prepare(df: pd.DataFrame):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    df['Result'] = np.where(df['HomeGoals'] > df['AwayGoals'], 'H',
                     np.where(df['HomeGoals'] < df['AwayGoals'], 'A', 'D'))

    def long_team_frame(df):
        home = df[['Date','HomeTeam','HomeGoals','AwayGoals']].rename(columns={'HomeTeam':'Team','HomeGoals':'GF','AwayGoals':'GA'})
        away = df[['Date','AwayTeam','AwayGoals','HomeGoals']].rename(columns={'AwayTeam':'Team','AwayGoals':'GF','HomeGoals':'GA'})
        long = pd.concat([home.assign(Venue='H'), away.assign(Venue='A')], ignore_index=True)
        long = long.sort_values(['Team','Date']).reset_index(drop=True)
        long['pts'] = np.where(long['GF']>long['GA'],3,np.where(long['GF']==long['GA'],1,0))
        long['gd']  = long['GF']-long['GA']
        long['ppg5'] = long.groupby('Team')['pts'].shift().rolling(5).mean().reset_index(level=0, drop=True)
        long['gd5']  = long.groupby['Team']['gd'].shift().rolling(5).mean().reset_index(level=0, drop=True)
        return long

    long = long_team_frame(df)
    df = df.merge(long[long['Venue']=='H'][['Date','Team','ppg5','gd5']].rename(columns={'Team':'HomeTeam','ppg5':'Home_ppg5','gd5':'Home_gd5'}), on=['Date','HomeTeam'], how='left')
    df = df.merge(long[long['Venue']=='A'][['Date','Team','ppg5','gd5']].rename(columns={'Team':'AwayTeam','ppg5':'Away_ppg5','gd5':'Away_gd5'}), on=['Date','AwayTeam'], how='left')
    df = df.dropna(subset=['Home_ppg5','Away_ppg5']).reset_index(drop=True)
    return df

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', default='model.pkl')
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = prepare(df)

    cutoff = df['Date'].quantile(0.85)
    train = df[df['Date'] <= cutoff]
    test  = df[df['Date']  > cutoff]

    num_cols = ['Home_ppg5','Home_gd5','Away_ppg5','Away_gd5']
    if set(['Odds_H','Odds_D','Odds_A']).issubset(df.columns):
        num_cols += ['Odds_H','Odds_D','Odds_A']
    cat_cols = ['HomeTeam','AwayTeam']

    from joblib import dump
    num_pipe = Pipeline([('imp', SimpleImputer(strategy='median'))])
    pre = ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

    clf = Pipeline([('pre', pre),
                    ('lr', LogisticRegression(max_iter=2000, multi_class='multinomial'))])

    X_train = train[num_cols + cat_cols]; y_train = train['Result']
    X_test  = test[num_cols + cat_cols];  y_test  = test['Result']

    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test); y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred))
    try:
        from sklearn.metrics import log_loss
        print("Log loss:", log_loss(y_test, proba))
    except Exception:
        pass

    dump(clf, args.out)
    print(f"Saved model to {args.out}")

if __name__ == '__main__':
    main()
