# Multimodal AI-driven Discovery of Mast Cell Degranulation Factors

## プロジェクトの概要 (Overview)
本プロジェクトは、アレルギーや炎症疾患の鍵となる「マスト細胞の脱顆粒プロセス」に関与する未知の制御因子（新規創薬ターゲット候補）を予測・同定するための機械学習パイプラインです。
トランスクリプトーム、タンパク質間相互作用（PPI）、機能アノテーション、配列情報など、複数の異種データソースを統合したマルチモーダル解析を実施しています。

## 技術スタック (Tech Stack)
- **Languages:** Python 3.x
- **Machine Learning:** scikit-learn, BorutaPy, transformers (ESM-2), networkx, node2vec
- **Data Manipulation:** pandas, numpy, scipy

## データセット (Datasets)
本プロジェクトでは以下の公共データベースを利用しています。
（※データのライセンス保護および一次情報元からの取得を推奨するため、生データはリポジトリに含めていません。以下の公式リンクより取得して `data/raw/` に配置してください。）

- **Transcriptome:** GEO dataset `GSE75526` (BMMC Expression Profiles)
- **PPI Network:** STRING Database (v12.0)
- **Protein Annotation:** UniProt (Keywords, Subcellular location, Sequence etc.)

## 解析パイプライン (Architecture & Pipeline)

### 1. Data Integration & Hard Negative Mining
- BMMCで発現し、脱顆粒への関与が既知の遺伝子を正例（Positive）として定義。
- ハウスキーピング遺伝子に加え、「正例と特徴が似ているが脱顆粒には関与しない遺伝子」を**Hard Negatives**として抽出。これにより、予測モデルの偽陽性を効果的に抑制しています。

### 2. Feature Engineering (特徴量抽出)
- **発現特異性:** 発現データからTau Scoreを算出。
- **配列情報 (ESM-2):** 最先端のタンパク質言語モデル（ESM-2）を用いて配列を高次元ベクトル化し、PCAで次元圧縮。
- **ネットワーク特徴量 (Node2Vec):** STRINGのPPIネットワークからグラフ埋め込みベクトルを取得。
- **自然言語処理 (TF-IDF):** UniProtのキーワードを統合し、TF-IDFによりベクトル化。

### 3. Feature Selection (Borutaによる特徴量選択)
- 膨大なマルチモーダル特徴量に対し、ランダムフォレストベースの**Borutaアルゴリズム**を適用。
- 予測に統計的に有意に寄与する特徴量（Confirmed / Tentative）のみを厳選することで、モデルの解釈性と未知データへの汎化性能を向上させています。

### 4. Machine Learning Model
- 厳選された特徴量を用いて**Random Forest**を学習。
- Stratified 5-Fold CVによりモデルの安定性を評価し、GEOデータ上で発現が見られる全遺伝子に対して新規候補因子のスコアリング（推論）を実施しました。

## 予測結果の生物学的妥当性と考察 (Discussion)
構築したモデルを用いて未知の遺伝子のスコアリングを行った結果、上位20遺伝子にはFceRIシグナル伝達に直結する重要な因子群が濃縮されていました。

具体的には、トップ3以内に特定のTecファミリーキナーゼ（※具体的な遺伝子名は未発表研究のため秘匿）がランクインし、学習データのBTK等と類似の機能を持つアクセル役をAIが正確に見出しました。
また興味深いことに、上位層には既知の強力な抑制性因子（ブレーキ役）も複数含まれていました。これは、本AIモデルがSTRINGとNode2vecのネットワーク構造から「脱顆粒プロセスの中核を担うシグナル複合体（Signalosome）」のハブを空間的に正しく捉えられていることを示唆しています。

## 実行方法 (How to Run)
1. `data/raw/` ディレクトリにGEO (GSE75526) とSTRINGのデータを配置します。
2. `notebooks/main_analysis.ipynb` を上から順に実行することで、データの前処理からモデル学習、候補遺伝子のスコアリングまでを一貫して再現可能です。
