import pandas as pd
import numpy as np

def preprocess_test_data(test_df, train_columns):
    df = test_df.copy()
    df['Log_LotArea'] = np.log1p(df['LotArea'])
    df['Log_GrLivArea'] = np.log1p(df['GrLivArea'])
    df['Log_LotFrontage'] = np.log1p(df['LotFrontage'])
    cols_to_fill_none = [
        'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
        'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
        'MasVnrType'
    ]
    for col in cols_to_fill_none:
        if col in df.columns:
            df[col] = df[col].fillna("None")
    cols_to_fill_zero = [
        'GarageYrBlt', 'GarageCars', 'GarageArea',
        'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 
        'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'
    ]
    for col in cols_to_fill_zero:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    df['LotFrontage'] = df.groupby("Neighborhood")['LotFrontage'].transform(
        lambda x: x.fillna(x.median())
    )
    df['Log_LotFrontage'] = df.groupby("Neighborhood")['Log_LotFrontage'].transform(
        lambda x: x.fillna(x.median())
    )
    df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])

    for col in df.select_dtypes(include=['object']):
        df[col] = df[col].fillna(df[col].mode()[0])
    for col in df.select_dtypes(include=[np.number]):
        df[col] = df[col].fillna(0)
    cols_to_drop = [
        'Id', 
        'LotArea',      # Replaced by Log_LotArea
        'GrLivArea',    # Replaced by Log_GrLivArea
        'LotFrontage',  # Replaced by Log_LotFrontage
        'SalePrice'     # Target variable (if it accidentally exists in test)
    ]
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    for col in ['MSSubClass', 'MoSold', 'YrSold']:
        if col in df.columns:
            df[col] = df[col].astype(str)

    df_encoded = pd.get_dummies(df, drop_first=True)
    df_final = df_encoded.reindex(columns=train_columns, fill_value=0)

    return df_final