import pandas as pd 
import ast
df = pd.read_csv('../data/processed/eval.csv', encoding='utf-8') 
cids = df['cid'].apply(ast.literal_eval).tolist()
print(cids[:10])
