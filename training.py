import pandas as pd


file_name = './mseg-api/mseg/class_remapping_files/MSeg_master.tsv'
# Read the tsv file with pandas
df = pd.read_csv(file_name, sep='\t')
# See all rows  
pd.set_option("display.max_rows", None)
labels = df["universal"]

print(list(labels))


# # Extract the column 'ade20k-150-relabeled'
# if 'ade20k-150-relabeled' in data.columns:
#     ade20k_column = data['ade20k-150-relabeled']
    
#     # Iterate through the column and print each element
#     for element, idx in enumerate(ade20k_column):
#         print(element," ",idx)
# else:
#     print("'ade20k-150-relabeled' column not found in the TSV file.")
