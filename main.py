from sentence_transformers import SentenceTransformer
import pandas
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

df = pandas.read_csv('prompts.csv')
df['embedding'] = df['prompt'].apply( lambda x : model.encode(x) )

def calc_score(input):
    input_embed = model.encode(input)
    df['score'] = df['embedding'].apply( lambda x : cosine_similarity([x], [input_embed]).flatten()[0] )
    return df.sort_values(by='score', ascending=False)['act'][0:5]
