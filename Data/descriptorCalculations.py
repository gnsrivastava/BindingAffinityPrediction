from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger
import multiprocessing as mp
import pandas as pd


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

def getMolDescriptors(mol, missingVal=None):
    ''' calculate the full list of descriptors for a molecule
    
        missingVal is used if the descriptor cannot be calculated
    '''
    res = {}
    for nm,fn in Descriptors._descList:
        # some of the descriptor fucntions can throw errors if they fail, catch those here:
        try:
            val = fn(mol)
        except:
            # print the error message:
            import traceback
            traceback.print_exc()
            # and set the descriptor value to whatever missingVal is
            val = missingVal
        res[nm] = val
    return res

def CalculateDescriptor(inp):
    data = dict()
    for mol_dict in inp:
        for mol, rdmol in mol_dict.items():
            if rdmol is None:
                continue
            res = getMolDescriptors(rdmol)
        data[mol] = res
    return data

if __name__=="__main__":
    
    
    dataPath = '/work/gsriva2/BindingAffinityPrediction/BindingDB'

    bindingdb = pd.read_csv(f'{dataPath}/FinalBindingDB.tsv', sep='\t', low_memory=False)
    
    # Get number of processors
    nprocs = mp.cpu_count()
	
    # create a pool of processes
    pool = mp.Pool(processes=nprocs)
    
    mols = [{mol: Chem.MolFromSmiles(mol)} for mol in bindingdb.SMILES.unique().tolist()]

    Data = pd.DataFrame()
    try:
        # Split the data
        inp_lists = slice_data(mols, nprocs)
        multi_result = [pool.apply_async(CalculateDescriptor, (inp,)) for inp in inp_lists]
	
        # Wait for tasks to complete
        for result in multi_result:
            res = pd.DataFrame.from_dict(result.get()).T
            Data = pd.concat([Data, res], ignore_index=False)  # Wait and raise exceptions if any
	    
    finally:
        # Close and join the pool
        pool.close()
        pool.join()
        
    print("All processes completed.")
    Data.drop_duplicates(inplace=True)
    Data.reset_index(inplace=True)
    Data = Data.rename(columns={'index':'SMILES'})
    Data.to_pickle(f'{dataPath}/descriptors.pkl')

