# Download Uniprot sequence
import pandas as pd
import os, sys
import subprocess

bindingdbPath = './Data/'

df = pd.read_csv(f'{bindingdbPath}/BindingDB_pIC50.tsv', sep='\t')

try:
    os.mkdir(f"{bindingdbPath}/TargetSequences")
except FileExistsError:
    pass

print(f'Download is in progress. Please Wait! ✋🏼\n')

for uniprot in df['UniProt (SwissProt) Primary ID of Target Chain'].unique():
    try:
        os.system(f'curl -O -L https://rest.uniprot.org/uniprotkb/{uniprot}.fasta')
        os.system(f'mv {uniprot}.fasta {bindingdbPath}/sequences/TargetSequences/')
    except:
        break

print(f'Download is completed 👍🏼\n')
#https://rest.uniprot.org/uniprotkb/Q9NWZ3.fasta
