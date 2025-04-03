import pandas as pd
df = pd.read_csv('Final_BindngDb_data.tsv', sep='\t')

# Get the subset of the data
df = df[['Sequence','UniProt_ID', 'EC_number']].drop_duplicates().copy() # , 'SMILES']].drop_duplicates().copy()

# Convert the Uniprot ID to string and then group by UniProt ID
df['UniProt_ID'] = df['UniProt_ID'].astype(str)

df['UniProt_ID'] = (df['UniProt_ID'] + 
    df.groupby('UniProt_ID').
        cumcount().
        replace(0, '').
        apply(lambda x: f'.{x}' if x != '' else ''))

#df1 = df[['UniProt_ID', 'Sequence']].drop_duplicates()

with open('./sequnece_cluster_based_split/clustering/Sequences.fasta', 'w') as _file:
    fasta_lines = (
        '>' + df['UniProt_ID'] + 
        '|' + df['EC_number'] +
        '\n' + df['Sequence'] + '\n'
    ).tolist()
    _file.writelines(fasta_lines)

_file.close()
