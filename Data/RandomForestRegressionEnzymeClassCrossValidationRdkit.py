import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.utils.class_weight import compute_sample_weight
import statistics
import os

# Load feature importance
importance_df = pd.read_csv('FeatureImportance.csv')

# Load Data
df = pd.read_pickle('FinalDataWithRdkitEC2VECRandom.pkl')
df = df[df.EC_number != '0.0.0.0']

# Add enzyme class (first digit of EC number)
df['enzyme_class'] = df['EC_number'].str.split('.').str[0].astype(int)

# Merge class 5 into 4 and class 7 into class 6
df.loc[df.enzyme_class==5, 'enzyme_class'] = 4
df.loc[df.enzyme_class==7, 'enzyme_class'] = 6

# Select columns
cols = ['SMILES'] + importance_df[importance_df.Importance > 0].Feature.tolist() + \
       [col for col in df.columns if 'ec2vec' in col] + ['pIC50', 'pIC50_random', 'enzyme_class']

df = df[cols].dropna()

# Define GroupKFold
num_fold = 5
gkf = GroupKFold(n_splits=num_fold)

mse_all, mae_all, r2_all = [], [], []
mse_all_random, mae_all_random, r2_all_random = [], [], []

PATH = '/work/gsriva2/BindingAffinityPrediction/BindingDB'
os.makedirs(os.path.join(PATH, 'Trained_model'), exist_ok=True)

fold_cnt = 0

full_class_weights = {1:0.82, 2:0.49, 3:0.7, 4:0.977, 6:0.989}

for train_index, test_index in gkf.split(df, groups=df['enzyme_class']):
    print(f'\nFold {fold_cnt + 1} start:')

    train_set = df.iloc[train_index]
    test_set = df.iloc[test_index]

    feature_cols = [col for col in train_set.columns if col not in ['SMILES', 'pIC50', 'pIC50_random', 'enzyme_class']]

    # Compute sample weights based on enzyme class
    #sample_weights = compute_sample_weight(class_weight='balanced', y=train_set['enzyme_class'])

    # Keep only class weights for classes present in the current fold
    present_classes = train_set['enzyme_class'].unique()
    class_weights = {cls: wt for cls, wt in full_class_weights.items() if cls in present_classes}

    # Compute sample weights
    sample_weights = compute_sample_weight(
        class_weight=class_weights,
        y=train_set['enzyme_class']
    )

    # Define regressors
    rf_regressor = RandomForestRegressor(n_estimators=100, max_depth=30,
                                         min_samples_leaf=4, min_samples_split=5,
                                         random_state=42, n_jobs=-1)

    rf_regressor_random = RandomForestRegressor(n_estimators=100, max_depth=30,
                                                min_samples_leaf=4, min_samples_split=5,
                                                random_state=42, n_jobs=-1)

    # Train models
    rf_regressor.fit(train_set[feature_cols], train_set['pIC50']) #, sample_weight=sample_weights)
    rf_regressor_random.fit(train_set[feature_cols], train_set['pIC50_random']) #, sample_weight=sample_weights)

    # Save models
    joblib.dump(rf_regressor, f'{PATH}/Trained_model/Rdkit_EC_original_regressor_model_fold_{fold_cnt+1}.pkl')
    joblib.dump(rf_regressor_random, f'{PATH}/Trained_model/Rdkit_EC_random_regressor_model_fold_{fold_cnt+1}.pkl')

    # Evaluate
    X_test = test_set[feature_cols]
    y_test = test_set['pIC50']
    y_test_random = test_set['pIC50_random']

    y_pred = rf_regressor.predict(X_test)
    y_pred_random = rf_regressor_random.predict(X_test)

    mse = metrics.mean_squared_error(y_test, y_pred)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    r2 = metrics.r2_score(y_test, y_pred)

    mse_all.append(mse)
    mae_all.append(mae)
    r2_all.append(r2)

    print(f'MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}')

    # Random
    mse_r = metrics.mean_squared_error(y_test_random, y_pred_random)
    mae_r = metrics.mean_absolute_error(y_test_random, y_pred_random)
    r2_r = metrics.r2_score(y_test_random, y_pred_random)

    mse_all_random.append(mse_r)
    mae_all_random.append(mae_r)
    r2_all_random.append(r2_r)

    print(f'MSE_r: {mse_r:.4f}, MAE_r: {mae_r:.4f}, R²_r: {r2_r:.4f}')

    fold_cnt += 1

# Final performance
print('\n########## Overall Regression Performance: Original ##########')
print(f'Mean MSE: {statistics.mean(mse_all):.4f}')
print(f'Mean MAE: {statistics.mean(mae_all):.4f}')
print(f'Mean R² Score: {statistics.mean(r2_all):.4f}')

print('\n########## Overall Regression Performance: Random ##########')
print(f'Mean MSE: {statistics.mean(mse_all_random):.4f}')
print(f'Mean MAE: {statistics.mean(mae_all_random):.4f}')
print(f'Mean R² Score: {statistics.mean(r2_all_random):.4f}')

