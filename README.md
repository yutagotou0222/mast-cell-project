# Multimodal AI-driven Discovery of Mast Cell Degranulation Factors

## プロジェクトの概要 (Overview)
本プロジェクトは、肺線維症や全身性強皮症などのアレルギー・炎症疾患の進行において重要な役割を果たす「マスト細胞の脱顆粒プロセス」に関与する未知の制御因子（新規創薬ターゲット候補）を予測・同定するための機械学習パイプラインです。
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

## 解決した技術的課題と工夫 (Challenges & Solutions)

### 1. 異種データ（マルチモーダル）の統合と次元の呪いの回避
ゲノミクス分野の機械学習において、発現量、配列、ネットワークといった性質の異なるデータを高次元のまま統合すると、過学習を引き起こす「次元の呪い」が課題となります。この課題に対し、最先端のタンパク質言語モデル（ESM-2）を用いて配列情報の背後にある生化学的特性をベクトル化して次元圧縮を行い、新規タンパク質に対する予測精度を向上させました。さらに、膨大な特徴量空間に対してランダムフォレストベースのBorutaアルゴリズムを適用し、予測に統計的に有意に寄与する特徴量のみを厳選することで、モデルの解釈性と未知データへの汎化性能を確保しています。

### 2. 学習データの質向上（Hard Negative Miningの導入）
単なるランダムサンプリングで負例（Negative）を抽出すると、モデルが単純な特徴しか学習できず偽陽性が増加します。本プロジェクトでは「正例と特徴が似ているが脱顆粒には関与しない遺伝子」を意図的にHard Negativesとして定義・抽出して学習させることで、実運用における識別精度を大幅に向上させました。

### 3. ドライ解析とウェット実験の知見の融合
計算機上のインシリコスクリーニングにとどまらず、実際の実験室で扱ってきたマスト細胞の培養や、フローサイトメトリー（FCM）、Western blotting等による細胞内シグナル解析（Lyn/SykカスケードやIgE受容体のインターナリゼーション評価）の知見を特徴量設計に還元しています。ウェット実験で得た「生物学的な妥当性」の感覚をモデルの評価基準に組み込むことで、単なる数値上の精度向上ではなく、実際の創薬プロセスに繋がる実践的なパイプラインを構築しました。

## 解析パイプライン (Architecture & Pipeline)

### 1. Data Integration & Hard Negative Mining
- BMMCで発現し、脱顆粒への関与が既知の遺伝子を正例（Positive）として定義。
- ハウスキーピング遺伝子に加え、前述のHard Negativesを抽出しモデルの偽陽性を抑制。

### 2. Feature Engineering (特徴量抽出)
- **発現特異性:** 発現データからTau Scoreを算出。
- **配列情報 (ESM-2):** 大規模タンパク質言語モデル（ESM-2）による高次元ベクトル化とPCAによる次元圧縮。
- **ネットワーク特徴量 (Node2Vec):** STRINGのPPIネットワークからグラフ埋め込みベクトルを取得。
- **自然言語処理 (TF-IDF):** UniProtのキーワードを統合し、TF-IDFによりベクトル化。

### 3. Feature Selection (Borutaによる特徴量選択)
- Borutaアルゴリズムを適用し、Confirmed / Tentativeとして判定された特徴量のみを厳選。

### 4. Machine Learning Model
- 厳選された特徴量を用いてRandom Forestを学習。
- Stratified 5-Fold CVによりモデルの安定性を評価し、GEOデータ上で発現が見られる全遺伝子に対して新規候補因子のスコアリングを実施。

## 予測結果の生物学的妥当性と考察 (Discussion)
構築したモデルを用いて未知の遺伝子のスコアリングを行った結果、上位20遺伝子にはFceRIシグナル伝達に直結する重要な因子群が濃縮されていました。

具体的には、トップ3以内に特定のTecファミリーキナーゼ（※具体的な遺伝子名は未発表研究のため秘匿）がランクインし、学習データのBTK等と類似の機能を持つアクセル役をAIが正確に見出しました。また興味深いことに、上位層には既知の強力な抑制性因子（ブレーキ役）も複数含まれていました。これは、本AIモデルがSTRINGとNode2vecのネットワーク構造から「脱顆粒プロセスの中核を担うシグナル複合体（Signalosome）」のハブを空間的に正しく捉えられていることを示唆しています。

## 実行方法 (How to Run)
1. `data/raw/` ディレクトリにGEO (GSE75526) とSTRINGのデータを配置します。
2. `notebooks/main_analysis.ipynb` を上から順に実行することで、データの前処理からモデル学習、候補遺伝子のスコアリングまでを一貫して再現可能です。

## 参考文献 (References)
- Q. Li et al., Clin. Rheumatol. (2026).
- M. W. Libbrecht, W. S. Noble, Nat. Rev. Genet. 16, 321–332 (2015).
- A. Veerappan et al., DNA Cell Biol. 32, 206–218 (2013).
- M. Kulmanov et al., Nat. Mach. Intell. 6, 220–228 (2024).
- H. R. Kilgore et al., Science 387, 1095–1101 (2025) . 
