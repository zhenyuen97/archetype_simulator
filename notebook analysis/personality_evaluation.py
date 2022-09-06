def balance_dataset(df, column_name):
    import pandas as pd
    
    sampling_no = df[column_name].value_counts().min()

    count = 0
    for i in df[column_name].unique():
        temp_df = df[df[column_name] == i].sample(n = sampling_no, random_state = 1)
        
        if count == 0:
            full_df = temp_df
            
        else:
            full_df = pd.concat([full_df, temp_df])
            
        count += 1
            
    return full_df[['posts', column_name]]

def build_bag_of_words_features_filtered(words):
    import nltk
    import string
    
    useless_words = nltk.corpus.stopwords.words("english") + list(string.punctuation)
    words = nltk.word_tokenize(words)
    return {
        word:1 for word in words \
        if not word in useless_words}

def feature_creation(df, bag_of_words_features, column_name):
    features = []
    for personality in df[column_name].unique():
        temp = df[df[column_name] == personality]
        temp = temp['posts']
        
        features += [[(bag_of_words_features(i), personality) \
            for i in temp]]  
        
    return features

def train_test(features):
    import numpy as np
    
    split=[]
    for i in range(2):
        split += [len(features[i]) * 0.8]
        
    split = np.array(split,dtype = int)

    train=[]
    for i in range(2):
        train += features[i][:split[i]] 
        
    test=[]
    for i in range(2):
        test += features[i][split[i]:]
        
    return train, test