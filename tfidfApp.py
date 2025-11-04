import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Leitura do dataset
df = pd.read_csv("steam_reviews_2807960_ptbr.csv")
df.drop_duplicates(subset=['review'], inplace=True)
nltk.download('stopwords')
stops = set(stopwords.words('portuguese'))

def limpar_texto(texto):
    texto = str(texto).lower()
    texto = re.sub(r'[^a-záéíóúãõâêôç\s]', '', texto)
    palavras = [p for p in texto.split() if p not in stops]
    return ' '.join(palavras)

df['clean'] = df['review'].apply(limpar_texto)

# 3. Documento de referência (review negativa)
review_negativa = """Pedi devolução pois enfrentei diversos problemas de perfomance no modo multiplayer, com muito lag, 
rollback e input delay, mesmo tendo especificações acima do mínimo e com drivers atualizados... problemas que não rolaram 
no beta. Ainda vejo potencial no game, mas vou dar um tempo e esperar corrigirem esses problemas (que eu sei que nem todos 
estão enfrentando)."""

review_negativa_clean = limpar_texto(review_negativa)

# 4. Geração dos vetores TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix = vectorizer.fit_transform(list(df['clean']) + [review_negativa_clean])

# O último vetor é da review negativa
tfidf_dataset = tfidf_matrix[:-1]
tfidf_target = tfidf_matrix[-1]

# 5. Similaridade do cosseno
similaridades = cosine_similarity(tfidf_target, tfidf_dataset).flatten()

# Top 10 mais semelhantes
top_indices = similaridades.argsort()[-10:][::-1]

print("===== Reviews mais semelhantes à negativa (sem duplicatas) =====\n")
for i in top_indices:
    print(f"Similaridade: {similaridades[i]:.3f}")
    print(df.iloc[i]['review'])
    print("------")
