def emotions_lexicon(company_data, emotion_lexicon_df):
    from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
    from nltk.stem.snowball import SnowballStemmer
    import pandas as pd 
    import streamlit as st
    
    emotions_df = pd.DataFrame(0, index = company_data.index, columns = emotion_lexicon_df.columns)
    stemmer = SnowballStemmer('english')
    # Iterate over reviews
    company_data['reviews'] = company_data['pros'] + company_data['cons']
    progress_bar = st.progress(0)
    status_update = st.empty()
    
    count = 0 
    for row in company_data.itertuples():                       
        # Tokenise reviews
        progress_bar.progress(count/len(company_data['reviews']))
        status_update.text(str(round((count/len(company_data['reviews']))*100)) + '%')
        document = word_tokenize(company_data.loc[row.Index]['reviews'])
        
        # Iterate over words in reviews
        for word in document:
            
            # Stem and convert to lower
            word = stemmer.stem(word.lower())
            
            # Match emotion score with NRC emotions database
            emo_score = emotion_lexicon_df[emotion_lexicon_df['Word'] == word]
            if not emo_score.empty:
                for emotion in list(emotion_lexicon_df.columns.drop("Word")):
                    
                    # Append emotions score
                    emotions_df.at[row.Index, emotion] += emo_score[emotion]
        
        count += 1
                    
    exclusive_list = ['Anger', 'Fear', 'Disgust']
    openness_list = ['Joy', 'Trust', 'Anticipation']
    emotions_df['is_exclusive'] = emotions_df[exclusive_list].sum(axis = 1)
    emotions_df['is_open'] = emotions_df[openness_list].sum(axis = 1)
    emotions_df['exclusive_openness'] = emotions_df.apply(lambda x: 'Exclusive' if (x['is_exclusive'] > x['is_open']) else 'Open', axis = 1)
    company_data = pd.concat([company_data, emotions_df], axis = 1)
    company_data['score'] = company_data['score'].astype(int)
    
    return company_data


    logo = 'SUSS logo.png'
    company_archetypes = 'images/company_archetypes.png'
    emotion_chart = 'images/emotion_chart.png'
    employment_type = 'images/employment_type.png'
    former_employee_description = 'images/former_employee_description.png'
    job_types_avg_score = 'images/job_types_avg_score.png'
    personality_chart = 'images/personality_chart.png'
    personality_type_avg_score = 'images/personality_type_avg_score.png'