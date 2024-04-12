import pandas as pd
import numpy as np
import math

ratings = [
    ('u1','i1',2),('u1','i2',3),('u1','i4',1),('u1','i5',3),('u1','i6',8),
    ('u2','i2',3),('u2','i3',1),('u2','i4',4),('u2','i5',6),('u2','i6',7),
    ('u3','i1',3),('u3','i4',3),('u3','i5',4),('u3','i6',6),
    ('u4','i1',9),('u4','i2',5),('u4','i3',1),('u4','i4',5),('u4','i6',7),
    ('u5','i1',3),('u5','i2',4),('u5','i3',6),('u5','i4',7),('u5','i5',9),('u5','i6',9),
    ('u6','i1',4),('u6','i3',1),('u6','i4',4),('u6','i5',8),
    ('u7','i1',2),('u7','i2',4),('u7','i6',8),
]

def construct_df(ratings):
    df = pd.DataFrame(0, columns=items, index=users)
    for user, item, rating in ratings:
        df.at[user, item] = rating
    return df

def cosine_similarity(u1, u2):
    common_items = np.logical_and(u1 > 0, u2 > 0)
    if not np.any(common_items):
        return 0
    u1 = u1[common_items]
    u2 = u2[common_items]
    return np.dot(u1, u2) / (np.linalg.norm(u1) * np.linalg.norm(u2))

def user_filtering(user, item, df, users, k):
    other_users = [u for u in users if u != user]
    similarities = {other_user: cosine_similarity(df.loc[user], df.loc[other_user]) for other_user in other_users}
    top_users = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:k]
    weighted_sum = sum(similarity * df.loc[other_user, item] for other_user, similarity in top_users if df.loc[other_user,item]>0)
    total_similarity = sum(similarity for _, similarity in top_users if df.loc[_,item]>0)
    return weighted_sum / total_similarity if total_similarity > 0 else 0

def item_filtering(user, item, df, items, k):
    other_items = [i for i in items if i != item]
    similarities = {other_item: cosine_similarity(df[item], df[other_item]) for other_item in other_items}
    top_items = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:k]
    weighted_sum = sum(similarity * df.loc[user][other_item] for other_item, similarity in top_items if df.loc[user,other_item]>0)
    total_similarity = sum(similarity for _, similarity in top_items if df.loc[user,_]>0)
    return weighted_sum / total_similarity if total_similarity > 0 else 0


users = sorted(set(user for user, _, _ in ratings))
items = sorted(set(item for _, item, _ in ratings))
df = construct_df(ratings)
print(df)

k = 3
score_user = user_filtering('u1', 'i3', df, users, k)
print("User-User Collaborative Filtering Score:", score_user)

score_item = item_filtering('u1', 'i3', df, items, k)
print("Item-Item Collaborative Filtering Score:", score_item)
