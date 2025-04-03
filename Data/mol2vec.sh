#! /bin/bash

# This is the script to run mol2vec in parallel


# Export the model and data directory paths

export MODEL_PATH='/work/gsriva2/apps/mol2vec/examples/models/model_300dim.pkl'

mkdir -p /work/gsriva2/BindingAffinityPrediction/BindingDB/temp_mol2vec

export InputFilePath='/work/gsriva2/BindingAffinityPrediction/BindingDB/temp_mol2vec'


cat smiles.csv | parallel -j20 --colsep='\t' ' 
	printf "%s\t%s\n" {1} {2} > $InputFilePath/temp_{2}.smi;
	$mol2vec featurize --in-file $InputFilePath/temp_{2}.smi --out-file $InputFilePath/temp_out_{2}.txt -m $MODEL_PATH -r 1 --uncommon UNK; 
	sed -i 1d $InputFilePath/temp_out_{2}.txt;
	cut -d"," -f4- $InputFilePath/temp_out_{2}.txt > $InputFilePath/temp_out_{2}_1.txt;
	paste -d"," $InputFilePath/temp_{2}.smi $InputFilePath/temp_out_{2}_1.txt > $InputFilePath/temp_out_{2}_2.txt;
	sed -i "s/\t/,/g" $InputFilePath/temp_out_{2}_2.txt;
	cat $InputFilePath/temp_out_{2}_2.txt >> ./OutPutMol2vec.txt;
'

#rm $InputFilePath/temp_*;

cat ./header.txt ./OutPutMol2vec.txt > ./BindingdbMol2vec.csv;

rm ./OutPutMol2vec.txt;

exit 0;

