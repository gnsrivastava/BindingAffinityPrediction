from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
import multiprocessing as mp
import pandas as pd

# The script is to generate input for the Subset program \
# But  can be used to generate morgan fingerprints for any problem

pd.set_option('display.max_columns', 5)
pd.set_option('display.max_rows', 50)


RDLogger.DisableLog('rdApp.*')

def slice_data(data, nprocs):
    aver, res = divmod(len(data), nprocs)
    nums = []
    for proc in range(nprocs):
        if proc < res:
            nums.append(aver + 1)
        else:
            nums.append(aver)
    count = 0
    slices = []
    for proc in range(nprocs):
        slices.append(data[count: count+nums[proc]])
        count += nums[proc]
    return slices

def MorganFingerprint(mol):
    if mol is None:
        return None
    
    fps = AllChem.GetMorganFingerprintAsBitVect(
            mol, 
            useChirality=True, 
            radius=2, 
            nBits=1024
        )
    return fps.ToBitString()

def getMorganFingerprints(subset_input):
    subset_input = subset_input.copy()
    subset_input['MorganFp'] = subset_input['SMILES'].apply(lambda smi: MorganFingerprint(Chem.MolFromSmiles(smi)))
    return subset_input[['molID', 'MorganFp']]

if __name__=="__main__":
    dataPath = '/work/gsriva2/BindingAffinityPrediction/BindingDB'

    bindingdb = pd.read_csv(f'{dataPath}/Final_BindngDb_data.tsv', sep='\t', low_memory=False)
    
    # Vectorized operations to separate SMILES as DataFrame
    bindingdb = (
        bindingdb[['SMILES']]
        .drop_duplicates()
        .reset_index(drop=True)
        .assign(molID=lambda df: 'mol_' + df.index.astype(str))
        [['molID', 'SMILES']]
    )    
    

    # Get number of processors
    nprocs = mp.cpu_count()

    # create a pool of processes
    pool = mp.Pool(processes=nprocs)

    subset_inputs = slice_data(bindingdb, nprocs) 
    
    finalData = pd.DataFrame()
    try:
        multi_result = [pool.apply_async(getMorganFingerprints, (subset_input,)) for subset_input in subset_inputs]
        for result in multi_result:
            finalData = pd.concat([finalData, result.get()], ignore_index=False)
    
    finally:
        # Close and join the pool
        pool.close()
        pool.join()
    
    finalData.to_csv('MorganFP.tab', sep="\t", index=False, header=None)
