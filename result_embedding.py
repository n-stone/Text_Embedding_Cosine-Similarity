from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from sentence_transformers import SentenceTransformer
import json

model = SentenceTransformer('MODEL')

df = pd.read_csv('Result Data File')
df['embedding'] = df['embedding'].apply(json.loads)

while 1:
    test = input("user > ")
    embedding = model.encode(test)

    df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())

    answer = df.loc[df['distance'].idxmax()]

    print("bot > ", answer['label'])
    print("distance > ", answer['distance'])