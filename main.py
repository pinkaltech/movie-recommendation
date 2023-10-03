from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import pandas as pd

def movie_find(m):
   
    data = pd.read_csv("main_data.csv")
    # cv = CountVectorizer()
    tfd= TfidfVectorizer()
    # count_matrix = cv.fit_transform(data['movie_title'])
    # similarity = cosine_similarity(count_matrix)
    count_matrix = tfd.fit_transform(data["movie_title"])
    user_vector = tfd.transform([m])
    similarity = cosine_similarity(user_vector,count_matrix).flatten()
    print(similarity.dtype)
    print(similarity.argmax())
    a = similarity.argmax()
    
    return data["movie_title"][a]
    




def create_similarity():
    data = pd.read_csv("main_data.csv")
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['comb'])
    similarity  = cosine_similarity(count_matrix)
    return data,similarity

def rcmd(m):
    m = m.lower()
    try:
        data.head()
        similarity.shape()
    except:
        data,similarity = create_similarity()
        if m not in data["movie_title"].unique():
            return ("Sorry! The movie you requested is not in our database. Please check the spelling or try with some other movies")
        else:
            # i = data.loc[data["movie_title"] == m]
            i = data[data["movie_title"] == m].index[0]
            lst = list(enumerate(similarity[i]))
            lst = sorted(lst,key=lambda x:x[1],reverse=True)
            lst = lst[0:11]
            l = []
            # print(lst[1][0])

            for i in range(len(lst)):
                a = lst[i][0]
                l.append(data['movie_title'][a])
            return l


movie = input("Enter movie")
movie = movie.lower()
movie = movie_find(movie)
recommend_movie = rcmd(movie)
print(recommend_movie)

