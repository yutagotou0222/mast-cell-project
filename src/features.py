import os
import tempfile
import numpy as np
import pandas as pd
import networkx as nx
import torch
from tqdm import tqdm

from Bio import Align
from Bio.Align import substitution_matrices
from transformers import EsmTokenizer, EsmModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from pecanpy.pecanpy import SparseOTF

# =====================================================================
# 1. 発現量ベースの特徴量
# =====================================================================
def calculate_Tau_score(row_data):
    """
    各組織/細胞状態間の発現特異性（Tau Score）を計算する
    """
    try:
        max_val = row_data.max()
        if max_val == 0: return 0.0
        return (1 - (row_data/max_val)).sum() / (len(row_data) - 1)
    except Exception as e:
        print(f"Error calculating Tau score: {e}")
        return 0.0

# =====================================================================
# 2. タンパク質配列ベースの特徴量 (BioPython & ESM-2)
# =====================================================================
def setup_aligners():
    """BioPythonのアライメント設定を初期化"""
    aligner_global = Align.PairwiseAligner()
    aligner_global.mode = 'global'
    aligner_global.open_gap_score = -10.0
    aligner_global.extend_gap_score = -0.5

    aligner_local = Align.PairwiseAligner()
    aligner_local.mode = 'local'
    aligner_local.open_gap_score = -10.0
    aligner_local.extend_gap_score = -0.5

    try:
        matrix = substitution_matrices.load("BLOSUM62")
        aligner_global.substitution_matrix = matrix
        aligner_local.substitution_matrix = matrix
    except Exception:
        print("Warning: BLOSUM62 not found. Using default matrix.")
    return aligner_global, aligner_local

# モジュール読み込み時に1度だけアライナーを初期化
aligner_global, aligner_local = setup_aligners()

def preprocess_sequence(seq):
    if not seq or pd.isna(seq): return ""
    table = str.maketrans({'U': 'C', 'O': 'K', 'B': 'D', 'Z': 'E', 'J': 'L'})
    seq = str(seq).strip().upper().translate(table)
    return seq.replace('X', '').replace('*', '')

def calculate_similarity_biopython(target_seq, ref_sequences):
    """ターゲット配列と参照配列（正例）群との類似度を計算"""
    if not target_seq or pd.isna(target_seq):
        return 0.0, 0.0, "None", "None"

    clean_target_seq = preprocess_sequence(target_seq)
    if not clean_target_seq or len(clean_target_seq) < 3:
        return 0.0, 0.0, "None", "None"

    target_self_global = aligner_global.score(clean_target_seq, clean_target_seq)
    target_self_local = aligner_local.score(clean_target_seq, clean_target_seq)

    max_global, max_local = 0.0, 0.0
    best_match_global, best_match_local = "None", "None"

    for ref_name, ref_seq in ref_sequences.items():
        if not ref_seq or pd.isna(ref_seq): continue
        clean_ref_seq = preprocess_sequence(ref_seq)
        if len(ref_seq) < 3: continue
        
        try:
            raw_global_score = aligner_global.score(clean_target_seq, clean_ref_seq)
            norm_global_score = raw_global_score / target_self_global if target_self_global > 0 else 0
            raw_local_score = aligner_local.score(clean_target_seq, clean_ref_seq)
            norm_local_score = raw_local_score / target_self_local if target_self_local > 0 else 0

            norm_global_score = min(max(norm_global_score, 0.0), 1.0)
            norm_local_score = min(max(norm_local_score, 0.0), 1.0)

            if norm_global_score > max_global:
                max_global = norm_global_score
                best_match_global = ref_name

            if norm_local_score > max_local:
                max_local = norm_local_score
                best_match_local = ref_name
        except Exception:
            continue

    return max_global, max_local, best_match_global, best_match_local


class ESM2Embedder:
    """
    ESM-2モデルをクラスとしてラップし、メモリの無駄遣いを防ぐ設計
    """
    def __init__(self, model_name="facebook/esm2_t33_650M_UR50D"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = EsmTokenizer.from_pretrained(model_name)
        self.model = EsmModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
    def get_embedding(self, sequence):
        max_len = 1022
        sequence = sequence[:max_len]
        inputs = self.tokenizer(sequence, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.cpu().numpy()[0]

def extract_esm2_features(df, sequence_col='Sequence', n_components=30):
    """データフレーム内の配列をESM-2でベクトル化し、PCAで圧縮する関数"""
    embedder = ESM2Embedder()
    embeddings_list = []
    for seq in tqdm(df[sequence_col], desc="Extracting ESM2 Embeddings"):
        try:
            vector = embedder.get_embedding(seq)
            embeddings_list.append(vector)
        except Exception:
            embeddings_list.append(np.zeros(1280))
            
    X_esm2_raw = np.array(embeddings_list)
    pca_esm2 = PCA(n_components=n_components, random_state=42)
    X_esm2_compressed = pca_esm2.fit_transform(X_esm2_raw)
    
    esm2_cols = [f'ESM2_PC_{i}' for i in range(n_components)]
    df_esm2 = pd.DataFrame(X_esm2_compressed, columns=esm2_cols)
    return df_esm2, pca_esm2

# =====================================================================
# 3. テキストベースの特徴量 (TF-IDF & ヒューリスティック)
# =====================================================================
def create_full_text(df):
    if 'Full_Text_Info' in df.columns:
        base_text = df['Full_Text_Info'].fillna("").astype(str)
    else:
        base_text = pd.Series([""] * len(df))
        
    if 'Assigned_Module' in df.columns:
        base_text = base_text + " " + df['Assigned_Module'].fillna("").astype(str)
    if 'Summary' in df.columns:
        base_text = base_text + " " + df['Summary'].fillna("").astype(str)
    return base_text

def extract_tfidf_features(df, text_col='Full_Text_Info', n_components=10):
    """テキストデータをTF-IDFでベクトル化し、SVDで圧縮する関数"""
    df[text_col] = create_full_text(df)
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X_tfidf_raw = vectorizer.fit_transform(df[text_col])
    
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_tfidf_compressed = svd.fit_transform(X_tfidf_raw)
    
    svd_cols = [f'Text_SVD_{i}' for i in range(n_components)]
    df_tfidf = pd.DataFrame(X_tfidf_compressed, columns=svd_cols)
    return df_tfidf, vectorizer, svd

def calculate_human_context_score(row):
    """脱顆粒に関連するキーワードによる文脈スコアリング"""
    text = str(row.get('Full_Text_Info', '')).lower()
    keywords = {
        'degranulation': 3, 'mast cell': 3, 'exocytosis': 3, 'secretion': 2, 
        'fusion': 2, 'histamine': 1, 'calcium': 1, 'snare': 2, 'vesicle': 1, 
        'immune response': 2, 'signaling': 1, 'activation': 1, 'kinase': 1, 
        'phosphorylation': 1, 'receptor': 1, 'ige': 2, 'fcer': 3, 'adapter': 1
    }
    score = sum(point for word, point in keywords.items() if word in text)
    return score

# =====================================================================
# 4. ネットワークベースの特徴量 (STRING & Node2Vec)
# =====================================================================
def get_string_interaction(target_genes, raw_data_dir="data/raw/"):
    """STRINGデータベースからのネットワーク構築とPageRankの計算"""
    links_file = os.path.join(raw_data_dir, "10090.protein.links.v12.0.txt.gz")
    info_file = os.path.join(raw_data_dir, "10090.protein.info.v12.0.txt.gz")
    
    df_info = pd.read_csv(info_file, sep="\t", compression='gzip', usecols=[0, 1])
    df_info.columns = [c.replace('#', '') for c in df_info.columns]
    gene_to_id = dict(zip(df_info['preferred_name'].str.upper(), df_info['string_protein_id']))

    target_ids, id_to_user_gene = [], {}
    for g in target_genes:
        g_upper = str(g).upper()
        if g_upper in gene_to_id:
            string_id = gene_to_id[g_upper]
            target_ids.append(string_id)
            id_to_user_gene[string_id] = g
            
    target_id_set = set(target_ids)
    df_links = pd.read_csv(links_file, sep=" ", compression='gzip', dtype={'combined_score': np.uint16})
    
    # 高速フィルタリングとIDの復元
    df_filtered = df_links[
        (df_links['protein1'].isin(target_id_set)) &
        (df_links['protein2'].isin(target_id_set)) &
        (df_links['combined_score'] >= 700)
    ].copy()
    df_filtered['protein1'] = df_filtered['protein1'].map(id_to_user_gene)
    df_filtered['protein2'] = df_filtered['protein2'].map(id_to_user_gene)
    df_filtered['combined_score'] = df_filtered['combined_score'] / 1000.0

    edge_df = df_filtered[['protein1', 'protein2', 'combined_score']]
    G = nx.from_pandas_edgelist(edge_df, 'protein1', 'protein2', ['combined_score'])
    for g in target_genes:
        if g not in G: G.add_node(g)

    deg = nx.degree_centrality(G)
    pr = nx.pagerank(G, weight='combined_score', alpha=0.85)

    centrality_df = pd.DataFrame({
        'Gene_ID': target_genes,
        'Degree_Centrality': [deg.get(g, 0) for g in target_genes],
        'PageRank': [pr.get(g, 0) for g in target_genes]
    })
    return centrality_df, edge_df

def generate_graph_embeddings(string_df, dimension=64):
    """Pecanpy(Node2vec)によるグラフ埋め込みベクトルの生成"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.edgelist') as tmp:
        temp_path = tmp.name
        string_df[['protein1', 'protein2', 'combined_score']].to_csv(temp_path, sep='\t', index=False, header=False)

    try:
        model = SparseOTF(p=1.0, q=1.0, workers=8, verbose=False)
        model.read_edg(temp_path, weighted=True, directed=False)
        embeddings_matrix = model.embed(dim=dimension, num_walks=30, walk_length=20)
        
        df_graph = pd.DataFrame(embeddings_matrix, columns=[f"Graph_Dim_{i+1}" for i in range(dimension)])
        df_graph['Gene_ID'] = model.nodes
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
    return df_graph