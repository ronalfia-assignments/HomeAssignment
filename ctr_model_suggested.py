import string
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, train_test_split
from sklearn import metrics
from sklearn.preprocessing import QuantileTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import xgboost as xgb


full_df = pd.read_csv('listings_dataset.csv', index_col=0)
full_df.head()
full_df.columns

temp = full_df.isnull().mean()
temp[temp > 0].plot(kind='barh')
plt.title('Normalized missing data count per column')
plt.xlim(0, 1)

numeric_columns = ['position', 'gig_price', 'gig_avg_rating', 'gig_rated_orders', 'cnt_helpful_reviews_last_year', 'avg_review_length_last_3_months',
       'avg_position_shown_last_60d', 'previous_order_amount']
dates_columns = ['created_at', 'user_reg_date', 'previous_order_date']
binary_columns = ['is_click', 'is_filtered', 'is_user_buyer','is_seller_onlie']
categorical_columns = list(set(full_df.columns) - set(numeric_columns) - set(dates_columns) - set(binary_columns))
## there are few columns that are not categorical - search_query, gig_id, gig_title, listing_id, user_timezone; free text and time zone should be treated other than categorical. For id, need to be careful

temp = full_df.groupby('gig_sc_id').agg(
    num_clicks=('is_click', 'sum'),
    group_size=('gig_sc_id', 'size')
)
temp['sc_id_ctr'] = temp.num_clicks / temp.group_size

full_df = full_df.join(temp, on='gig_sc_id').drop(['num_clicks','group_size'], axis=1)

def add_time_features(df):
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['user_reg_date'] = pd.to_datetime(df['user_reg_date'])
    df['previous_order_date'] = pd.to_datetime(df['previous_order_date'])

    df['hour'] = (df.created_at.dt.hour + (df.created_at.dt.minute / 60)).round(1)
    df['day'] = df.created_at.dt.day_name().apply(lambda x: x.lower()[:3])
    df['week_in_month'] = (df.created_at.dt.day - 1) // 7 + 1 ## fixed: days are counted from 1
    df['months_since_reg'] = ((df.created_at - df.user_reg_date).dt.days / 30.2).round(1)
    df['hour_diff_from_last_purchase'] = (df.created_at - df.previous_order_date).dt.total_seconds().div(3600).round(1) ## fixed: first substract then extract
    df['days_since_last_purchase'] = (df.created_at - df.previous_order_date).dt.days
    def is_weekend(row): return row.created_at.weekday() >= 5 ## added missing implementation
    df['is_weekend'] = df.apply(is_weekend, axis=1) ## you can do dt['created_at'].dt.weekday >= 5
    df = df.drop(['created_at','previous_order_date','user_reg_date'], axis=1)
    return df

full_df_dt = add_time_features(full_df) ## don't override the original df please

def rare_cats_to_other(ser, thresh=0.05): ## data leakage - apply after splitting
    value_counts = ser.value_counts(normalize=True)
    values_to_replace = value_counts[value_counts < thresh].index
    return ser.replace(values_to_replace, 'other')

full_df_rare = full_df_dt.copy() ## please don't override the df, it's harder to rollback
for col in categorical_columns:
    full_df_rare[col] = rare_cats_to_other(full_df_dt[col])
    full_df_rare[col].fillna('other', inplace=True)

for col in numeric_columns:
    full_df_rare[col].fillna(0, inplace=True)

def compare_sc_id(row):
    if pd.notna(row['gig_sc_id']) and pd.notna(row['previous_order_sc_id']):
        return int(row['gig_sc_id'] == row['previous_order_sc_id'])

def compare_order_amounts(row):
    if pd.notna(row['gig_price']) and pd.notna(row['previous_order_amount']) and row['previous_order_amount'] > 0:
        return row['gig_price'] / row['previous_order_amount']

def add_gig_features(df):
    df['price_change_from_last_purchase'] = df.apply(compare_order_amounts, axis=1)
    df['is_same_sc_as_last_purchase'] = df.apply(compare_sc_id, axis=1)
    df['position_diff_from_avg'] = df.position - df.avg_position_shown_last_60d
    return df

full_df_rare = add_gig_features(full_df_rare)

x_train_full, x_test, y_train_full, y_test = train_test_split(full_df_rare.drop('is_click', axis=1), full_df['is_click'], test_size=0.2)
x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.2)

print(f"Train: {len(x_train.index)}")
print(f"Val: {len(x_val.index)}")
print(f"Test: {len(x_test.index)}")

model_params = dict(
    learning_rate=0.2,
    n_estimators=200,
    max_depth=7,
    colsample_bytree=0.9,
    colsample_bylevel=0.9,
    colsample_bynode=0.9,
    importance_type='cover',
    scale_pos_weight=2,
)

cat_cols = x_train.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
print("Categorical columns in X_train:", cat_cols)

categorial_to_drop = ['listing_id', 'gig_id', 'context', 'search_query', 'gig_sc_id', 'gig_title', 'user_timezone', 'previous_order_sc_id']

x_train_full_d = x_train_full.drop(columns=categorial_to_drop)

remaining_cat = x_train_full_d.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

x_train_full_d = pd.get_dummies(x_train_full_d, columns=remaining_cat, drop_first=True)

n_folds = 6
splitter = KFold(n_splits=n_folds, shuffle=True, random_state=13)
feature_importance = []
cv_metrics = []

best_auc = -np.inf
best_model = None

for i, (train_idx, val_idx) in enumerate(splitter.split(x_train_full_d, y_train_full)):
    print(f'Fold {i+1}/{n_folds}')

    x_tr = x_train_full_d.iloc[train_idx]
    y_tr = y_train_full.iloc[train_idx]
    x_val = x_train_full_d.iloc[val_idx]
    y_val = y_train_full.iloc[val_idx]

    model = xgb.XGBClassifier(eval_metric='logloss', **model_params)
    model.fit(x_tr, y_tr,
              eval_set=[(x_tr, y_tr), (x_val, y_val)], verbose=50)
    y_prob = model.predict_proba(x_val)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)
    auc = metrics.roc_auc_score(y_val, y_prob)
    feature_importance.append(dict(zip(model.feature_names_in_, model.feature_importances_)))
    cv_metrics.append({
        'accuracy': metrics.accuracy_score(y_val, y_pred),
        'balanced_accuracy': metrics.balanced_accuracy_score(y_val, y_pred),
        'f1': metrics.f1_score(y_val, y_pred),
        'auc_roc': auc,
    })
    if auc > best_auc:
        best_auc   = auc
        best_model = model
cv_df = pd.DataFrame(cv_metrics)
print(cv_df)
cv_df.plot()
plt.xticks(range(n_folds), range(1, n_folds+1))
plt.grid()
plt.show();

pd.DataFrame.from_dict(feature_importance).mean().sort_values().plot(kind='barh', figsize=(8, 10))
plt.title('Mean feature importance (cover)')
plt.show();

## we need to look into the promoted gig click prediction task

x_test_copy = x_test.copy().reset_index(drop=True)
y_test_copy = y_test.copy().reset_index(drop=True)

# print("X head:\n", x_test_copy.head(3))
# print("y head:\n", y_test_copy.head(3))

mask = x_test_copy['imp_badge'] == 'promoted_gig'
x_test_pg = x_test_copy[mask]
y_test_pg = y_test_copy[mask]

# print("\nFiltered X (first 3):\n", x_test_pg.head(3))
# print("Filtered y (first 3):\n", y_test_pg.head(3))

drop_cols = [c for c in categorial_to_drop if c in x_test_pg.columns]

x_eval_pg = x_test_pg.drop(columns=drop_cols)

encode_cols = [c for c in remaining_cat if c in x_eval_pg.columns]
x_eval_pg = pd.get_dummies(x_eval_pg, columns=encode_cols, drop_first=True)

x_eval_pg = x_eval_pg.reindex(columns=x_train_full_d.columns, fill_value=0)

test_prob_pg = best_model.predict_proba(x_eval_pg)[:, 1]
test_pred_pg = (test_prob_pg > 0.5).astype(int)

test_metrics_pg = {
    'accuracy':          metrics.accuracy_score(y_test_pg, test_pred_pg),
    'balanced_accuracy': metrics.balanced_accuracy_score(y_test_pg, test_pred_pg),
    'f1':                metrics.f1_score(y_test_pg, test_pred_pg),
    'auc_roc':           metrics.roc_auc_score(y_test_pg, test_prob_pg),
}
test_df_pg  = pd.DataFrame(test_metrics_pg, index=['test_promoted'])
cv_mean_df  = cv_df.mean().to_frame(name='cv_mean').T

print(pd.concat([cv_mean_df, test_df_pg], axis=0))

from xgboost import XGBClassifier

# instantiate a single‐tree model
one_tree = XGBClassifier(
    n_estimators=1,        # just one tree
    max_depth=8,           # reasonably deep
    learning_rate=1.0,     # full step
    subsample=1.0,
    colsample_bytree=1.0,
    use_label_encoder=False,
    eval_metric='auc',
    tree_method='hist'
)

# train on all the training data
x_tr2, x_val2, y_tr2, y_val2 = train_test_split(
    x_train_full_d, y_train_full,
    test_size=0.2,
    stratify=y_train_full,
    random_state=13
)

pos = (y_train_full == 1).sum()
neg = (y_train_full == 0).sum()
balanced_w = neg / pos

model2 = xgb.XGBClassifier(
    n_estimators=5000,
    learning_rate=0.03,

    max_depth=6,
    min_child_weight=20,

    subsample=0.7,
    colsample_bytree=0.7,
    colsample_bynode=0.8,

    reg_lambda=200,
    reg_alpha=20,
    gamma=10,

    use_label_encoder=False,
    eval_metric='auc',
    tree_method='hist',
    scale_pos_weight= balanced_w,
)


model2.fit(
    x_tr2, y_tr2,
    eval_set=[(x_tr2, y_tr2), (x_val2, y_val2)],
    verbose=10
)

test_prob2 = model2.predict_proba(x_eval_pg)[:, 1]
test_pred2 = (test_prob2 > 0.5).astype(int)

metrics2 = {
    'accuracy':          metrics.accuracy_score(y_test_pg, test_pred2),
    'balanced_accuracy': metrics.balanced_accuracy_score(y_test_pg, test_pred2),
    'f1':                metrics.f1_score(y_test_pg, test_pred2),
    'auc_roc':           metrics.roc_auc_score(y_test_pg, test_prob2),
}

print("new tree model on promoted_gig:")
for name, val in metrics2.items():
    print(f"  {name}: {val:.4f}")