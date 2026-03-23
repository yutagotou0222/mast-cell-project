import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from boruta import BorutaPy

# =====================================================================
# 1. Borutaによる特徴量選択
# =====================================================================
def run_boruta_selection(X, y, random_state=42):
    """
    Borutaアルゴリズムを用いて、統計的に有意な特徴量（Confirmed / Tentative）を厳選する。
    
    Parameters:
        X (pd.DataFrame): 特徴量データフレーム
        y (np.array): 正解ラベル (0 or 1)
        
    Returns:
        selected_features (list): 選択された特徴量のカラム名のリスト
        boruta_selector (BorutaPy): 学習済みのBorutaオブジェクト
    """
    print(f"Starting Boruta feature selection with {X.shape[1]} features...")
    
    # Borutaの内部評価器としてRandomForestを使用
    rf_boruta = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5, random_state=random_state)
    
    # BorutaPyの初期化と実行
    boruta_selector = BorutaPy(rf_boruta, n_estimators='auto', verbose=0, random_state=random_state, max_iter=100)
    
    # DataFrameの値をNumPy配列に変換して学習（BorutaPyの仕様対応）
    boruta_selector.fit(X.values, y)
    
    # Confirmed (確実) と Tentative (保留) の両方を採用
    selected_mask = boruta_selector.support_ | boruta_selector.support_weak_
    selected_features = X.columns[selected_mask].tolist()
    
    print(f"Boruta selected {len(selected_features)} features out of {X.shape[1]}.")
    return selected_features, boruta_selector

# =====================================================================
# 2. モデルの学習と交差検証 (Cross Validation)
# =====================================================================
def train_and_evaluate_cv(X, y, n_splits=5, random_state=42):
    """
    層化K分割交差検証（Stratified K-Fold CV）を用いてモデルを評価し、
    特徴量重要度の安定性を記録する。
    """
    print(f"Starting Stratified {n_splits}-Fold Cross Validation...")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    metrics_list = []
    feature_importances = []
    
    # 全データ学習用の最終モデル（後で推論に使用）
    final_model = RandomForestClassifier(n_estimators=300, max_depth=7, class_weight='balanced', random_state=random_state, n_jobs=-1)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # モデルの初期化と学習
        model = RandomForestClassifier(n_estimators=300, max_depth=7, class_weight='balanced', random_state=random_state, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # 予測
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]
        
        # 評価指標の計算
        metrics = {
            'Fold': fold + 1,
            'Accuracy': accuracy_score(y_val, y_pred),
            'Precision': precision_score(y_val, y_pred, zero_division=0),
            'Recall': recall_score(y_val, y_pred, zero_division=0),
            'F1': f1_score(y_val, y_pred, zero_division=0),
            'AUC-ROC': roc_auc_score(y_val, y_prob)
        }
        metrics_list.append(metrics)
        
        # 特徴量の重要度を保存
        for col, imp in zip(X.columns, model.feature_importances_):
            feature_importances.append({'Feature': col, 'Importance': imp, 'Fold': fold + 1})

    df_metrics = pd.DataFrame(metrics_list)
    df_importances = pd.DataFrame(feature_importances)
    
    print("--- CV Results ---")
    print(df_metrics.mean().drop('Fold'))
    
    # 最終モデルの学習（推論用）
    final_model.fit(X, y)
    
    return final_model, df_metrics, df_importances

# =====================================================================
# 3. 評価の可視化 (特徴量重要度の安定性)
# =====================================================================
def plot_feature_importance_stability(df_importances, top_n=20, save_path=None):
    """
    CV各Foldでの特徴量重要度を箱ひげ図（Boxplot）として可視化し、
    重要度が安定しているか（ブレがないか）を確認する。
    """
    # 平均重要度でソート
    mean_imp = df_importances.groupby('Feature')['Importance'].mean().sort_values(ascending=False)
    top_features = mean_imp.head(top_n).index
    
    plot_data = df_importances[df_importances['Feature'].isin(top_features)]
    
    plt.figure(figsize=(10, 8))
    sns.boxplot(data=plot_data, x='Importance', y='Feature', order=top_features, palette='viridis')
    plt.title(f"Feature Importance Stability across Folds (Top {top_n})")
    plt.xlabel("Random Forest Feature Importance")
    plt.ylabel("Features")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
    
    plt.show()

# =====================================================================
# 4. 未知のデータ（GEOの全遺伝子）への推論
# =====================================================================
def predict_novel_targets(model, X_all, selected_features, gene_ids):
    """
    学習済みモデルを用いて、GEOデータで発現が見られた全遺伝子に対して
    脱顆粒因子のスコアリング（確率予測）を行う。
    """
    print(f"Predicting scores for {len(X_all)} genes...")
    
    # 学習時と同じ特徴量（カラム順）を確実に揃える
    X_predict = X_all[selected_features].copy()
    
    # 未知データの確率（スコア）を予測
    probabilities = model.predict_proba(X_predict)[:, 1]
    
    # 結果のデータフレーム作成
    results_df = pd.DataFrame({
        'Gene_ID': gene_ids,
        'Predicted_Score': probabilities
    })
    
    # スコアが高い順にソート
    results_df = results_df.sort_values(by='Predicted_Score', ascending=False).reset_index(drop=True)
    
    return results_df