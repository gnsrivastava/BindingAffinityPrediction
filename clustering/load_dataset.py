import os
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem, SaltRemover
from molvs import Standardizer
from joblib import Parallel, delayed
from tqdm import tqdm

RDLogger.DisableLog('rdApp.*')  # Suppress RDKit warnings

# Preload global objects
remover = SaltRemover.SaltRemover()
standardizer = Standardizer()

def process_smile(smi):
    """Preprocess a SMILES string and return fingerprint info."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        mol = remover.StripMol(mol)
        mol = standardizer.standardize(mol)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        return {
            'SMILES': smi,
            'STD_SMILES': Chem.MolToSmiles(mol),
            'FINGERPRINT': list(fp),
            'FP_RDKit': fp
        }
    except:
        return None

def get_mfp_parallel(smiles_list, n_jobs=20):
    """Process SMILES in parallel."""
    print(f"Processing {len(smiles_list)} SMILES with {n_jobs} workers...")
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_smile)(smi) for smi in tqdm(smiles_list)
    )
    results = [r for r in results if r is not None]

    if not results:
        return pd.DataFrame(), []

    df_fp = pd.DataFrame(results)
    col_names = [f'MFP_{i}' for i in range(1024)]
    df_fp[col_names] = pd.DataFrame(df_fp['FINGERPRINT'].tolist(), index=df_fp.index)
    df_fp.drop(columns=['FINGERPRINT'], inplace=True)

    return df_fp.copy(), list(df_fp['FP_RDKit'])

def tanimoto_outlier_filter(fps_rdkit, threshold=0.5):
    """Filter molecules by max Tanimoto similarity to earlier ones."""
    keep = [True]  # First one is always kept
    for i in tqdm(range(1, len(fps_rdkit))):
        sims = DataStructs.BulkTanimotoSimilarity(fps_rdkit[i], fps_rdkit[:i])
        keep.append(np.max(sims) > threshold)
    return np.array(keep)

def smiles_to_fingerprints(smiles_list, remove_outliers=True, n_jobs=20):
    df_fp, fps_rdkit = get_mfp_parallel(smiles_list, n_jobs=n_jobs)
    if remove_outliers and not df_fp.empty:
        print("Removing outliers based on Tanimoto similarity ≤ 0.5")
        keep_mask = tanimoto_outlier_filter(fps_rdkit, threshold=0.5)
        df_fp = df_fp[keep_mask].reset_index(drop=True)
    return df_fp

def check_output_folder(output_directory):
    """
    :param output_directory: folder path
    :return: None
    """
    if not(os.path.isdir(output_directory)):
        os.makedirs(output_directory, exist_ok=True)
    return None


'''
# === Example usage ===
if __name__ == '__main__':
    # Replace with your full list of SMILES
    smiles_df = pd.read_csv("your_300k_smiles_file.csv")  # assumes column 'SMILES'
    smiles_list = smiles_df['SMILES'].dropna().unique().tolist()

    df_fingerprints = smiles_to_fingerprints(smiles_list, remove_outliers=True, n_jobs=8)
    df_fingerprints.to_csv("molecular_fingerprints.tsv", sep='\t', index=False)
    print(df_fingerprints.head())
'''
