
def preprocessing(df):
    import nltk
    from nltk import tokenize, word_tokenize
    from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
    from nltk.text import Text
    from nltk.corpus import brown, stopwords
    from nltk.stem.snowball import SnowballStemmer
    
    # =================================
    # Preprocessing
    # =================================
    df['reviews'] = df['pros'] + df['cons']

    # Convert to lowercase
    df['reviews'] = df['reviews'].str.lower()

    # Remove Stop Words
    stop = stopwords.words('english')
    df['reviews'] = df['reviews'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

    # Remove punctuations
    df["reviews"] = df['reviews'].str.replace('[^\w\s]','')

    # Lemmentisation
    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    lemmatizer = nltk.stem.WordNetLemmatizer()

    def lemmatize_text(text):
        return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

    df['reviews'] = df['reviews'].apply(lemmatize_text)
    df['reviews'] = df['reviews'].apply(', '.join)

    return df

def sentiment(df):
    import pandas as pd
    import altair as alt
    import streamlit as st
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    from nltk import tokenize, word_tokenize
    import time
    
    
    charting_df = pd.DataFrame([['positive', 0], ['negative', 0], ['neutral', 0]], columns = ['sentiment', 'count'])
    plot_chart = alt.Chart(charting_df).mark_bar().encode(
        x='count:Q',
        y='sentiment:O'
    ).properties(height = 300)
    
    # Text status                
    progress_bar = st.progress(0)
    positive_status = st.empty()
    neutral_status = st.empty()
    negative_status = st.empty()
    charting = st.altair_chart(plot_chart, use_container_width=True)

    # Sentiment Analyzer
    analyzer = SentimentIntensityAnalyzer()
    sentiments_list = list()
    
    # Create empty variable container
    neutral = 0
    positive = 0
    negative = 0
    count = 1
    for review in df['reviews'].tolist():
        progress_bar.progress(count/len(df['reviews']))
        sentence_list = tokenize.sent_tokenize(review)
        sentiments = {'compound': 0.0, 'neg': 0.0, 'neu': 0.0, 'pos': 0.0}
            
        for sentence in sentence_list:
            vs = analyzer.polarity_scores(sentence)
            sentiments['compound'] += vs['compound']
            sentiments['neg'] += vs['neg']
            sentiments['neu'] += vs['neu']
            sentiments['pos'] += vs['pos']
                
        sentiments['compound'] = sentiments['compound'] / len(sentence_list)
        sentiments['neg'] = sentiments['neg'] / len(sentence_list)
        sentiments['neu'] = sentiments['neu'] / len(sentence_list)
        sentiments['pos'] = sentiments['pos'] / len(sentence_list)
        
        if (sentiments['neg'] > sentiments['neu']) & (sentiments['neg'] > sentiments['pos']):
            sentiment_score = 'negative'
            negative += 1
        elif (sentiments['pos'] > sentiments['neu']) & (sentiments['pos'] > sentiments['neg']):
            sentiment_score = 'positive'
            positive += 1
        else:
            sentiment_score = 'neutral'
            neutral += 1
        
        positive_status.text(
            'The number of positive comments is: ' + str(positive))
        
        negative_status.text(
            'The number of negative comments is: ' + str(negative))
        
        neutral_status.text(
            'The number of neutral comments is: ' + str(neutral))

        append_df = pd.DataFrame([[sentiment_score, 1]], columns = ['sentiment', 'count'])
        charting.add_rows(append_df)
        time.sleep(0.005)
        sentiments_list.append(sentiments)  # add this line
        
        count += 1

    return neutral, negative, positive

def emotions_lexicon(company_data, emotion_lexicon_df):
    from nltk.corpus import brown, stopwords
    from nltk.stem.snowball import SnowballStemmer
    import nltk
    import string
    import pandas as pd
    import streamlit as st
    
    with st.spinner('Transforming dataset...'):
        company_data['reviews'] = company_data['pros'] + company_data['cons']
        company_data['reviews'] = company_data['reviews'].str.lower()
        stop = stopwords.words('english')
        company_data['reviews'] = company_data['reviews'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
        company_data["reviews"] = company_data['reviews'].str.replace('[^\w\s]','')
        
        emotion_dict = emotion_lexicon_df.to_dict('records')
        
        def lemmatize_text(text):    
            stemmer = SnowballStemmer('english')
            w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
            lemmatizer = nltk.stem.WordNetLemmatizer()
            x = [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text) if w_tokenizer.tokenize(text) != ',']
            x = [''.join(c for c in s if c not in string.punctuation) for s in x]
            x = [stemmer.stem(word.lower()) for word in x]
            
            return x
        
        def transform_list(x):
            emotion_list = [next((item for item in emotion_dict if item["Word"] == i), None) for i in x]
            return emotion_list

        def sum_dict_values(dict_list):
            dict_list = [x for x in dict_list if x is not None]
            result = {}
            for d in dict_list:
                for k, v in d.items():
                    if k != 'Word' and isinstance(v, int):
                        result[k] = result.get(k, 0) + v
            return result    
    
    
        company_data['emotions'] = company_data['reviews'].apply(lemmatize_text)
        company_data['emotions'] = company_data['emotions'].apply(transform_list)
        company_data['emotions'] = company_data['emotions'].apply(sum_dict_values)
        company_data = pd.concat([company_data, company_data['emotions'].apply(pd.Series)], axis = 1)
    
    st.success('Transformation completed!!') 
            
    exclusive_list = ['Anger', 'Fear', 'Disgust']
    openness_list = ['Joy', 'Trust', 'Anticipation']
    company_data['is_exclusive'] = company_data[exclusive_list].sum(axis = 1)
    company_data['is_open'] = company_data[openness_list].sum(axis = 1)
    company_data['exclusive_openness'] = company_data.apply(lambda x: 'Exclusive' if (x['is_exclusive'] > x['is_open']) else 'Open', axis = 1)
    try:
        company_data['score'] = company_data['score'].astype(int)
    except KeyError as e:
        pass
    
    return company_data