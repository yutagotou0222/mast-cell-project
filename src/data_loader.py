import os
import io
import time
import requests
import pandas as pd
import numpy as np
import mygene
from scipy import stats
from tqdm import tqdm

# 特徴量生成モジュールからTauスコア計算関数をインポート
# （Notebookなどからプロジェクト全体を実行する際のパスを想定）
try:
    from src.features import calculate_Tau_score
except ImportError:
    # 単独テスト実行時へのフォールバック
    def calculate_Tau_score(row_data):
        try:
            max_val = row_data.max()
            if max_val == 0: return 0.0
            return (1 - (row_data/max_val)).sum() / (len(row_data) - 1)
        except:
            return 0.0

# =====================================================================
# 1. 教師データ（正例・負例）の設定
# =====================================================================
def get_target_genes():
    """学習に使用する既知の正例(Positives)と負例(Negatives)のリストを返す"""
    positives = [
        "VAMP8", "STX4A", "STX3", "STX11", "STXBP2", "SNAP23",
        "RAB27A", "RAB27B", "RAB3D","RAB37", "EFCAB4B", "RAB44", "UNC13D",
        "SYK", "LYN", "LAT", "RABGEF1", "SCFD1", "FCER1G", "MS4A2", "FCER1A", "FYN", "GAB2",
        "PLCG1", "PLCG2", "BTK", "LCP2", "VAV1", "CDC42", "RAC1", "RAC2",
        "PTPN6", "INPP5D", "SOS1", "GRB2", "MAPK1", "AKT1", "PIK3CD"
    ]
    negatives = [
        "GAPDH", "ACTB", "RPL4", "EEF1A1", "TUBB5", "ALB", "INS", "ACTG1",
        "SYT1", "SNAP25",
        "TRP53", "ATM", "BRCA1", "HK1", "LDHA", "PGK1",
        "COL1A1", "GJB2", "KCNQ1", "SLC2A1", "AQP1", "ATP1A1",
        "TFRC", "CD44", "ITGB1", "BSG", "CANX",
        "RAB1", "RAB2A", "RAB5A","RAB5C","RAB7",
        "STX1A", "STX7", "STX8", "STX12", "STX16", "STX17", "STX18", "STXBP1", "STXBP3",
        "MAPK14", "RAB11A", "MAPK1",
        "RABGEF1", "RABGAP1", "RAB3IP"
    ]
    return positives, negatives

# =====================================================================
# 2. GEOデータセットの読み込みと前処理
# =====================================================================
def load_geo_data(raw_file_path="data/raw/GSE75526_fpkm.txt.gz", cache_path="data/processed/GSE75526_processed.pkl"):
    """
    GEOデータセットを読み込み、発現量、FC、p-value、Tauスコアを計算する。
    処理時間を短縮するため、一度計算したものはpkl形式でキャッシュする。
    """
    if os.path.exists(cache_path):
        print("Loading cached GEO expression data...")
        return pd.read_pickle(cache_path)

    print("Processing raw GEO data...")
    columns_list = [
        'UT_1','UT_2','UT_3',
        'IgE_1','IgE_2','IgE_3',
        'Ag_1','Ag_2','Ag_3'
    ]
    
    try:
        df_geo = pd.read_csv(raw_file_path, compression='gzip', sep='\t', index_col=0)
        df_geo.index = df_geo.index.astype(str).str.upper().str.strip()
        df_geo = df_geo[~df_geo.index.duplicated(keep='first')]

        df_geo = df_geo.fillna(0)
        df_geo_log = np.log2(df_geo + 1)

        ut_cols = [c for c in df_geo_log.columns if "UT_" in c]
        stim_cols = [c for c in df_geo_log.columns if "Ag_" in c]

        df_geo_log['Expression'] = df_geo_log[columns_list].mean(axis=1)
        df_geo_log['ut_Expression'] = df_geo_log[ut_cols].mean(axis=1)
        df_geo_log['stim_Expression'] = df_geo_log[stim_cols].mean(axis=1)

        df_geo_log['log2FC'] = df_geo_log['stim_Expression'] - df_geo_log['ut_Expression']
        df_geo_log['Abs_log2FC'] = df_geo_log['log2FC'].abs()

        p_values = []
        for index, row in tqdm(df_geo_log.iterrows(), total=df_geo_log.shape[0], desc="Calculating p_values"):
            try:
                t_stat, p_val = stats.ttest_ind(row[stim_cols], row[ut_cols])
            except:
                p_val = 1.0
            p_values.append(p_val)

        df_geo_log['p_value'] = p_values
        df_geo_log['-log10_p_value'] = -np.log10(df_geo_log['p_value'] + 1e-300)

        df_geo_log['Tau_Score'] = df_geo_log.apply(calculate_Tau_score, axis=1)

        geo_passed = df_geo_log[df_geo_log['Expression'] > 0.0].copy()
        df_geo_result = geo_passed[['ut_Expression', 'log2FC', 'Abs_log2FC', '-log10_p_value', 'Tau_Score']].copy()
        df_geo_result.index.name = 'Gene_ID'
        df_geo_result = df_geo_result.reset_index()
        
        # 次回以降の高速化のためにprocessedフォルダへキャッシュを保存
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df_geo_result.to_pickle(cache_path)
        
        return df_geo_result

    except Exception as e:
        print(f"Error processing GEO data: {e}")
        return None

# =====================================================================
# 3. UniProtからのアノテーション情報取得
# =====================================================================
def fetch_uniprot_features(gene_ids):
    """遺伝子IDのリストを元に、UniProtから機能ドメインや配列情報を取得する"""
    limit_num = 200000
    if len(gene_ids) > limit_num:
        gene_ids = gene_ids[:limit_num]
        
    print(f"Fetching data from UniProt for {len(gene_ids)} genes...")
    base_url = "https://rest.uniprot.org/uniprotkb/search"
    results = []

    functional_modules = {
        "Module_Trafficking": ["RAB", "MYOSIN", "KINESIN", "DYNEIN", "MICROTUBULE", "ACTIN", "TRANSPORT", "MOTOR"],
        "Module_Fusion": ["SNARE", "VAMP", "STX", "SYNTAXIN", "SNAP23", "MUNC", "SYNAPTOTAGMIN", "FUSION", "DOCKING", "ESYT", "SEC1"],
        "Module_Signaling": ["KINASE", "PHOSPHORYLATION", "SH2", "SH3", "CALCIUM", "PLC", "IP3", "SIGNALING", "BTK", "SYK", "LYN", "ADAPTER"],
        "Module_Maintenance": ["AUTOPHAGY", "LYSOSOME", "PROTEASE", "SQSTM1", "LC3", "UBIQUITIN", "DEGRADATION", "GOLGI"],
        "Module_Reception": ["RECEPTOR", "FCER", "BINDING", "SENSING", "INTEGRIN", "CD74"],
        "Module_Cargo": ["TRYPTASE", "CHYMASE", "HISTAMINE", "SYNTHASE", "HDC", "CMA1", "TPSAB1"]
    }

    batch_size = 100
    for i in tqdm(range(0, len(gene_ids), batch_size), desc="UniProt API"):
        batch = gene_ids[i:i+batch_size]
        batch_map = {g.upper(): g for g in batch}
        query_genes = " OR ".join([f"gene_exact:{g}" for g in batch])
        query = f"({query_genes}) AND organism_id:10090"

        params = {
            "query": query,
            "format": "tsv",
            "fields": "gene_names,accession,keyword,protein_families,sequence,ft_domain,cc_subcellular_location,go",
            "size": 500
        }

        try:
            res = requests.get(base_url, params=params, timeout=20)
            if res.status_code == 200 and res.text.strip():
                df_res = pd.read_csv(io.StringIO(res.text), sep="\t")
                df_res.columns = [c.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('[', '').replace(']', '') for c in df_res.columns]

                for _, row in df_res.iterrows():
                    gene_names_str = str(row.get('gene_names', '') or '').upper()
                    uni_names = gene_names_str.split(' ')
                    matched_keys = set(uni_names) & set(batch_map.keys())

                    if not matched_keys: continue
                    hit_key = list(matched_keys)[0]
                    match_id = batch_map[hit_key]

                    kw_raw = str(row.get('keywords', '') or '')
                    fam_raw = str(row.get('protein_families', '') or '')
                    subcell = str(row.get('subcellular_location_cc', '') or '')
                    dom_ft = " ".join([str(row[c]) for c in df_res.columns if 'domain' in c.lower()])
                    func_raw = str(row.get('function_cc', '') or '')
                    go_raw = str(row.get('go', '') or '')
                    
                    all_info = f"{kw_raw} {fam_raw} {dom_ft} {subcell}".upper()

                    mod_scores = {}
                    for m, kws in functional_modules.items():
                        score = sum([1.0 for k in kws if k in all_info])
                        mod_scores[m] = score

                    has_snare = 1.0 if mod_scores.get("Module_Fusion", 0) > 0 else 0.0
                    has_kinase = 1.0 if mod_scores.get("Module_Signaling", 0) > 0 else 0.0
                    has_transport = 1.0 if mod_scores.get("Module_Trafficking", 0) > 0 else 0.0
                    has_maintenance = 1.0 if mod_scores.get("Module_Maintenance", 0) > 0 else 0.0
                    has_reception = 1.0 if mod_scores.get("Module_Reception", 0) > 0 else 0.0
                    has_cargo = 1.0 if mod_scores.get("Module_Cargo", 0) > 0 else 0.0

                    subcell_upper = subcell.upper()
                    is_secreted = 1.0 if "SECRETED" in subcell_upper or "EXTRACELLULAR" in subcell_upper else 0.0
                    has_tm = 1.0 if "TRANSMEMBRANE" in all_info or "INTEGRAL COMPONENT OF MEMBRANE" in all_info else 0.0

                    res_dict = {
                        'Gene_ID':match_id,
                        'Sequence':(row.get('sequence', '') or ''),
                        'Domain_Score':sum(mod_scores.values()),
                        'Is_Secreted':is_secreted,
                        'Has_Transmembrane':has_tm,
                        'Has_SNARE':has_snare,
                        'Has_Kinase':has_kinase,
                        'Has_Transport':has_transport,
                        'Has_Maintenance':has_maintenance,
                        'Has_Reception':has_reception,
                        'Has_Cargo':has_cargo,
                        'Full_Text_Info':all_info
                    }
                    res_dict.update(mod_scores)
                    results.append(res_dict)
            time.sleep(0.5)

        except Exception as e:
            print(f"Batch {i} error: {e}")
            continue

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results).drop_duplicates('Gene_ID')

# =====================================================================
# 4. MGIからの表現型・致死性データ取得
# =====================================================================
def fetch_mgi_phenotypes(gene_list):
    """MGI APIを使用して、ノックアウト時の表現型スコアと致死性リスクを抽出する"""
    print(f"Fetching MGI Phenotypes for {len(gene_list)} genes...")
    mg = mygene.MyGeneInfo()
    output = []

    pos_terms_weight = {
        "EXOCYTOSIS": 0.4, "SECRETION": 0.4, "DEGRANULATION": 0.5,
        "VESICLE": 0.4, "IMMUNE": 0.05, "MAST CELL": 0.5
    }
    lethal_terms = ["LETHAL", "DEATH", "INVIABLE", "INVIABILITY", "MORTALITY", "DIE", "STILLBORN"]

    try:
        results = mg.querymany(gene_list, scopes='symbol', fields='go,summary', species=10090, verbose=False)

        for item in results:
            gene = str(item.get('query')).upper()
            if not gene: continue

            pos_score = 0.0
            lethal_score = 0.0

            # GO terms check
            go_data = item.get('go', {})
            if isinstance(go_data, dict) and 'BP' in go_data:
                bp_list = go_data['BP'] if isinstance(go_data['BP'], list) else [go_data['BP']]
                for bp in bp_list:
                    term = str(bp.get('term', '')).upper()
                    for kw, weight in pos_terms_weight.items():
                        if kw in term: pos_score += weight

            # Phenotypes check
            phenotypes = item.get('phenotypes', [])
            if isinstance(phenotypes, list):
                for p in phenotypes:
                    p_term = str(p.get('term', '') if isinstance(p, dict) else p).upper()
                    for kw, weight in pos_terms_weight.items():
                        if kw in p_term: pos_score += weight
                    if any(t in p_term for t in lethal_terms): lethal_score = 1.0
            
            output.append({
                'Gene_ID': gene,
                'Phenotype_Score': min(pos_score, 1.0),
                'Lethality_Risk': lethal_score
            })

    except Exception as e:
        print(f"MGI Error: {e}")

    if not output: 
        return pd.DataFrame(columns=['Gene_ID', 'Phenotype_Score', 'Lethality_Risk'])
    return pd.DataFrame(output).drop_duplicates('Gene_ID')