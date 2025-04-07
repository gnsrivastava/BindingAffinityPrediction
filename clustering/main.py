from load_dataset import smiles_to_fingerprints, check_output_folder
from umap_clustering import umap_clustering_best
import os
import pandas as pd


def main(smiles_file, dir_file, n_clusters=7, outliers=True):
    print('Best clustering method (SMILES only)')
    
    # Load SMILES
    smiles_path = os.path.join(dir_file, smiles_file)
    '''
    with open(smiles_path, 'r') as f:
        smiles_list = [line.strip() for line in f if line.strip()]
    '''
    smiles_data = pd.read_csv(smiles_path, sep='\t')
    smiles_list = smiles_data.SMILES.unique()
    
    # Get fingerprints and filter outliers if requested
    print('\nProcessing SMILES...')
    df_data = smiles_to_fingerprints(smiles_list, remove_outliers=outliers)
    sample = df_data[[col for col in df_data.columns if col.startswith('MFP_')]].to_numpy()

    # Clustering
    print('\nClustering...')
    df_metrics, df_clustering = umap_clustering_best(sample=sample, df_data=df_data, n_clusters=n_clusters)

    # Save results
    print('\nSaving output...')
    dir_out = os.path.join(dir_file, 'Results', 'non_outliers_molecules' if outliers else 'all_molecules')
    check_output_folder(dir_out)

    file_metrics = os.path.join(dir_out, f'clustering_quality_metrics_k{n_clusters}.csv')
    file_cluster = os.path.join(dir_out, f'clustering_id_k{n_clusters}.csv')

    df_metrics.to_csv(file_metrics, index=False)
    df_clustering.to_csv(file_cluster, index=False)

    print('Done!')

# === Script Entry Point ===
if __name__ == '__main__':
    dir_working = '/work/gsriva2/BindingAffinityPrediction/BindingDB/clustering'
    file_smiles = 'lipinskiProps.tsv'  # e.g., one SMILES per line
    k = 7
    outliers = True

    main(smiles_file=file_smiles, dir_file=dir_working, n_clusters=k, outliers=outliers)

