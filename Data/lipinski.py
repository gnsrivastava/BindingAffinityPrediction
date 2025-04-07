# This is a script to calculate lipinski properties of the molecules in the dataset
from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

# Lipinski calculation
def compute_lipinski(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        'MW': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'HBD': Descriptors.NumHDonors(mol),
        'HBA': Descriptors.NumHAcceptors(mol)
    }

if __name__=='__main__':

    df = pd.read_csv('Final_BindngDb_data.tsv', sep='\t', low_memory=False)
    df = pd.DataFrame(df.SMILES)
    df = df.drop_duplicates()
    
    df[['MW', 'LogP', 'HBD', 'HBA']] = df['SMILES'].apply(compute_lipinski).apply(pd.Series)
    
    df.to_csv('lipinskiProps.tsv', sep='\t', index=False) 
