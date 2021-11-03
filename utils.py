import pandas as pd


def get_df_data():
    df = pd.read_csv('./data/in-vehicle-coupon-recommendation.csv')

    # "car" has only 108 values
    df.drop(labels=['car'], inplace=True, axis=1)
    df.dropna(inplace=True)

    # "toCoupon_GEQ5min" is fixed to 1
    df.drop(labels=['toCoupon_GEQ5min'], inplace=True, axis=1)

    # "direction_opp" and "direction_same" are opposites
    df.drop(labels=['direction_opp'], inplace=True, axis=1)

    df.replace({"coupon": {'Restaurant(<20)': 'Restaurant(-20)'}}, inplace=True)
    return df

