import pandas as pd
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('MODEL')

df = pd.read_csv('Source Data File')
df['embedding'] = pd.Series([[]] * len(df))
df['embedding'] = df['Question'].map(lambda x: list(model.encode(x)))
print(df)

df.to_csv("result_data.csv", index=False)