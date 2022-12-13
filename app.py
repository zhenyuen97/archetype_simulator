import streamlit as st
import pandas as pd
import numpy as np
import os

from datetime import timedelta
import datetime
import streamlit_authenticator as stauth
import pickle
from pathlib import Path
import altair as alt
from PIL import Image
import plotly.express as px

# Newly import
import string
import numpy as np
from sqlalchemy import create_engine
import pickle

import mysql.connector
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
from nltk.text import Text
from nltk.corpus import brown, stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.classify import NaiveBayesClassifier
from plotly.subplots import make_subplots
import plotly.graph_objs as go

import time
from modules.sentiment_analysis import *
from modules.chart_plotting import *
from modules.word_document import *
image = Image.open('SUSS logo.png')

st.set_page_config(layout="wide")

def main():    
    def app():
    
        # Initialize connection.
        # Uses st.experimental_singleton to only run once.
        @st.experimental_singleton
        def init_connection():
            return mysql.connector.connect(**st.secrets["mysql"])

#         conn = init_connection()

        # Perform query.
        @st.experimental_singleton
        def run_query(query):
            with conn.cursor() as cur:
                cur.execute(query)
                return cur.fetchall()
            
        @st.experimental_singleton
        def convert_dict(df):
            company_dict = {}
            for company in list(df['company'].unique()):
                company_dict[company] = df[df['company'] == company]
            
            return company_dict
        @st.cache(allow_output_mutation=True)
        def read_files(file_path, columns_list):
            df = pd.read_csv(file_path)
            df.columns = columns_list
            return df
        
        @st.cache(allow_output_mutation=True)
        def get_columns(rows, columns_list):
            df = pd.DataFrame(rows)
            df.columns = columns_list
            return df

#         # Get Bytedance Glassdoor Reviews
#         glassdoor_reviews = pd.read_csv('./data/Bytedance_Glassdoor_Reviews.csv')
#         glassdoor_reviews.columns = ['date', 'year exp', 'score','pros', 'cons', 'position']
#         rows = run_query("SELECT * from Bytedance_Glassdoor_Reviews;")
#         glassdoor_reviews = get_columns(rows, ['date', 'year exp', 'score','pros', 'cons', 'position'])

        # Get all company cleaned glassdoor reviews
#         rows = run_query("SELECT * from Company_Glassdoor_Data_Cleaned;")
#         all_company_data = get_columns(rows, ['headline', 'date', 'year exp', 'score', 'pros', 'cons', 'company',
#                             'position', 'raw_reviews', 'Introvert_Extrovert', 'Intuition_Sensing',
#                             'Thinking_Feeling', 'Judging_Perceiving', 'Innovative_Traditional',
#                             'Personality', 'reviews', 'emotions', 'Positive', 'Negative', 'Anger',
#                             'Anticipation', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise',
#                             'Trust', 'is_exclusive', 'is_open', 'exclusive_openness'])
        
        all_company_data = read_files('./data/Company_Glassdoor_Data_Cleaned.csv', ['headline', 'date', 'year exp', 'score', 'pros', 'cons', 'company',
                                    'position', 'raw_reviews', 'Introvert_Extrovert', 'Intuition_Sensing',
                                    'Thinking_Feeling', 'Judging_Perceiving', 'Innovative_Traditional',
                                    'Personality', 'reviews', 'emotions', 'Positive', 'Negative', 'Anger',
                                    'Anticipation', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise',
                                    'Trust', 'is_exclusive', 'is_open', 'exclusive_openness'])

        all_company_data = convert_dict(all_company_data)
        
        # Get emotion lexicon
#         rows = run_query("SELECT * from NRC_Emotion_Lexicon;")
#         emotion_lexicon_df = get_columns(rows, ['Word', 'Positive', 'Negative', 
#                                       'Anger', 'Anticipation', 'Disgust',
#                                       'Fear', 'Joy', 'Sadness', 'Surprise', 'Trust'])
        emotion_lexicon_df = read_files('./data/NRC_Emotion_Lexicon.csv', ['Word', 'Positive', 'Negative', 
                                      'Anger', 'Anticipation', 'Disgust',
                                      'Fear', 'Joy', 'Sadness', 'Surprise', 'Trust'])
        
        # Get personality description
#         rows = run_query("SELECT * from Personality_Type_Description;")
#         personality_type_description = get_columns(rows, ['Personality_Type', 'Name', 'Description'])
        
        personality_type_description = read_files('./data/personality_type_description.csv', ['Personality_Type', 'Name', 'Description'])
        print(personality_type_description)
        
        # models
        innovative_classifier = open('./models/InnovativeTraditionalClassifier.pickle', 'rb')
        InnovativeTraditionalClassifier = pickle.load(innovative_classifier)
        innovative_classifier.close()

        IE_classifier = open('./models/IntroExtroClassifier.pickle', 'rb')
        IntroExtroClassifier = pickle.load(IE_classifier)
        IE_classifier.close()
        
        NS_classifier = open('./models/IntuitionSensing.pickle', 'rb')
        IntuitionSensingClassifier = pickle.load(NS_classifier)
        NS_classifier.close()
        
        
        JP_classifier = open('./models/JudgingPerceiving.pickle', 'rb')
        JudgingPerceivingClassifier = pickle.load(JP_classifier)
        JP_classifier.close()
        
        TF_classifier = open('./models/ThinkingFeeling.pickle', 'rb')
        ThinkingFeelingClassifier = pickle.load(TF_classifier)
        TF_classifier.close()
        
        def build_bag_of_words_features_filtered(words):
            useless_words = nltk.corpus.stopwords.words("english") + list(string.punctuation)
            words = nltk.word_tokenize(words)
            return {
                word:1 for word in words \
                if not word in useless_words}
                
        # CSS to inject contained in a string
        hide_table_row_index = """
                    <style>
                    tbody th {display:none}
                    .blank {display:none}
                    </style>
                    """

        # Inject CSS with Markdown
        st.markdown(hide_table_row_index, unsafe_allow_html=True)

        # Title 
        st.sidebar.image(image, width = 200, use_column_width = 100)
        st.sidebar.title(f"Welcome {name}")
        authenticator.logout("Logout", "sidebar")
        
        st.title("Brand Archetype")
        st.write('This is a web app to look at your brand archetype')
        
        view_selection = st.selectbox('Select View', ("Upload Data", "View Available Resources", "Run Manually"))
        
        def archetype_analysis(company_data):
            company_data = emotions_lexicon(company_data, emotion_lexicon_df)
            
            company_data[['Employment Type', 'Duration']] = company_data['year exp'].str.split(',', expand=True)
            for row in company_data.itertuples():
                input = row.reviews 
                tokenize = build_bag_of_words_features_filtered(input)
                
                # Classify text
                ie = IntroExtroClassifier.classify(tokenize)
                it = InnovativeTraditionalClassifier.classify(tokenize)
                ns = IntuitionSensingClassifier.classify(tokenize)
                jp = JudgingPerceivingClassifier.classify(tokenize)
                tf = ThinkingFeelingClassifier.classify(tokenize)
                
                company_data.loc[row.Index, 'Introvert_Extrovert'] = ie
                company_data.loc[row.Index, 'Innovative_Traditional'] = it
                company_data.loc[row.Index, 'Intuition_Sensing'] = ns
                company_data.loc[row.Index, 'Judging_Perceiving'] = jp
                company_data.loc[row.Index, 'Thinking_Feeling'] = tf
                
            company_data['Personality_IE'] = company_data['Introvert_Extrovert'].apply(lambda x: 'I' if x == 'Introvert' else 'E')
            company_data['Personality_NS'] = company_data['Intuition_Sensing'].apply(lambda x: 'N' if x == 'Intuition' else 'S')
            company_data['Personality_TF'] = company_data['Thinking_Feeling'].apply(lambda x: 'T' if x == 'Thinking' else 'F')
            company_data['Personality_JP'] = company_data['Judging_Perceiving'].apply(lambda x: 'J' if x == 'Judging' else 'P')
            company_data['Personality'] = company_data['Personality_IE'] + company_data['Personality_NS'] + company_data['Personality_TF'] + company_data['Personality_JP']
            company_data.drop(columns = ['Personality_IE', 'Personality_NS', 'Personality_JP', 'Personality_TF'], inplace = True)
            
            return company_data
        
        def plot_charts(company_data, name_of_company):
            emotions_list = ['Anger', 'Anticipation', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise', 'Trust']
            values_list = company_data[emotions_list].sum().tolist()
            data_list = []
            for i in range(len(emotions_list)):
                temp_list = []
                temp_list.append(emotions_list[i])
                temp_list.append(values_list[i])
                data_list.append(temp_list)
                
            company_data[['Positive', 'Negative', 'Anger', 'Anticipation', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise', 'Trust', 'is_exclusive', 'is_open']] = company_data[['Positive', 'Negative', 'Anger', 'Anticipation', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise', 'Trust', 'is_exclusive', 'is_open']].astype('Int64')
            
            columns = ['date', 'Employment Type', 'Duration', 'score', 'pros', 'cons', 'position', 
            'emotions', 'exclusive_openness', 'Innovative_Traditional', 'Personality']
            
            st.code('Number of unique jobs: ' + str(len(company_data['position'].unique())))              
            st.dataframe(company_data[columns])
            
            combined_text_list = []
            col1, col2 = st.columns(2)
            with col1:
                pie_chart = pd.DataFrame(data_list, columns = ['Emotions', 'Count'])
                pie_chart['percent'] = round(((pie_chart['Count'] / pie_chart['Count'].sum()) * 100), 2)
                
                text1_list = pie_chart[['Emotions', 'percent']].sort_values(by = 'percent', ascending=False).head(3).values.tolist()
                text1 = f'{text1_list[0][0]} ({str(text1_list[0][1])}%), {text1_list[1][0]} ({str(text1_list[1][1])}%), and {text1_list[2][0]} ({str(text1_list[2][1])}%)'             
                plot_piechart(pie_chart, 'Emotions Pie Chart', 'Emotions', 'emotion_chart')
                combined_text_list.append(text1)
            
            with col2:
                pie_chart2 = pd.DataFrame(company_data['Personality'].value_counts().reset_index().values.tolist(), columns = ['Type', 'Count'])
                pie_chart2['percent'] = round(((pie_chart2['Count'] / pie_chart2['Count'].sum()) * 100), 2)
                pie_chart2['Type'] = pie_chart2['Type'].astype('str')
                
                
                personality_type_description.rename(columns = {'Personality_Type': 'Type'}, inplace = True)
                personality_type_description['Type'] = personality_type_description['Type'].astype('str')
                
                pie_chart2 = pd.merge(pie_chart2[['Type', 'Count', 'percent']], personality_type_description, on ='Type', how = 'left')
                print("This is pie chart 2")
                print(pie_chart2)
                text2_list = pie_chart2[['Type', 'percent', 'Name', 'Description']].sort_values(by = 'percent', ascending=False).head(3).values.tolist()
                
                text2 = f'{text2_list[0][0]} ({str(text2_list[0][1])}%), {text2_list[1][0]} ({str(text2_list[1][1])}%), and {text2_list[2][0]} ({str(text2_list[2][1])}%)'
                text3 = f'{text2_list[0][0]} are known as {text2_list[0][2]}, where they are {text2_list[0][3]}'
                text4 = f'{text2_list[1][0]} are known as {text2_list[1][2]}, where they are {text2_list[1][3]}'
                text5 = f'{text2_list[2][0]} are known as {text2_list[2][2]}, where they are {text2_list[2][3]}'
                
                plot_piechart(pie_chart2, 'Personality Type', 'Type', 'personality_chart')
                combined_text_list.append(text2)
                combined_text_list.append(text3)
                combined_text_list.append(text4)
                combined_text_list.append(text5)
                
            col3, col4 = st.columns(2)
            with col3:
                pie_chart3 = pd.DataFrame(company_data['Employment Type'].value_counts().reset_index().values.tolist(), columns = ['Type', 'Count'])
                plot_piechart(pie_chart3, 'Employment Type', 'Type', 'employment_type')
                pie_chart3['percent'] = round(((pie_chart3['Count'] / pie_chart3['Count'].sum()) * 100), 2)               
                
                text3_list = pie_chart3[['Type', 'percent']].sort_values(by = 'percent', ascending=False).values.tolist()
                text6 = f'{text3_list[0][0]} ({str(text3_list[0][1])}%) and {text3_list[1][0]} ({str(text3_list[1][1])}%)'
                combined_text_list.append(text6)
            with col4:
                pie_chart4 = pd.DataFrame(company_data.loc[(company_data['Employment Type'] == 'Former Employee') & (company_data['Duration'].notna())]['Duration'].value_counts().reset_index().values.tolist(), columns = ['Type', 'Count'])
                plot_piechart(pie_chart4, 'Former Employee against duration', 'Type', 'former_employee_description')
                pie_chart4['percent'] = round(((pie_chart4['Count'] / pie_chart4['Count'].sum()) * 100), 2)
                
                text4_list = pie_chart4[['Type', 'percent']].sort_values(by = 'percent', ascending=False).head(2).values.tolist()
                text7 = f'{text4_list[0][0]} ({str(text4_list[0][1])}%) and {text4_list[1][0]} ({str(text4_list[1][1])}%)'
                combined_text_list.append(text7)
            
            def archetype_text(data, column_name):
                df = pd.DataFrame(data[column_name].value_counts().reset_index().values.tolist(), columns = ['Type', 'Count'])
                df['percent'] = round(((df['Count'] / df['Count'].sum()) * 100), 2)
                text = df[['Type', 'percent']].sort_values(by = 'percent', ascending=False).head(1).values.tolist()
                text = f'{text[0][0]} ({text[0][1]}%)'
                
                return text
                                
            # Company archetype chart
            fig = make_subplots(rows=1, cols=4,
                                subplot_titles=["Introvert vs Extrovert",
                                                "Intuition vs Sensing",
                                                "Thinking vs Feeling", 
                                                "Judging vs Perceiving"]
                            )
            fig.add_trace(go.Bar(x = plot_subplots(company_data, 'Introvert_Extrovert')['Type'],
                                y = plot_subplots(company_data, 'Introvert_Extrovert')['Count'],
                                text = plot_subplots(company_data, 'Introvert_Extrovert')['Count %']),
                        row=1, col=1)
            
            fig.add_trace(go.Bar(x = plot_subplots(company_data, 'Intuition_Sensing')['Type'],
                                y = plot_subplots(company_data, 'Intuition_Sensing')['Count'],
                                text = plot_subplots(company_data, 'Intuition_Sensing')['Count %']),
                        row=1, col=2)
            
            fig.add_trace(go.Bar(x = plot_subplots(company_data, 'Thinking_Feeling')['Type'],
                                y = plot_subplots(company_data, 'Thinking_Feeling')['Count'],
                                text = plot_subplots(company_data, 'Thinking_Feeling')['Count %']),
                        row=1, col=3)
            
            fig.add_trace(go.Bar(x = plot_subplots(company_data, 'Judging_Perceiving')['Type'],
                                y = plot_subplots(company_data, 'Judging_Perceiving')['Count'],
                                text = plot_subplots(company_data, 'Judging_Perceiving')['Count %']),
                        row=1, col=4)
            
                        # Company archetype chart
            fig.update_layout(
                        title= '<b>MBTI Personalities</b>',
                        font=dict(
                            size=14,
                            color="Black"
                        )
                    )
            
            fig.update(layout_showlegend = False)
            
            st.plotly_chart(fig, use_container_width=True)
            
            fig = make_subplots(rows=1, cols=2,
                                subplot_titles=["Innovative vs Traditional",
                                                "Open vs Exclusive"])
            
            fig.add_trace(go.Bar(x = plot_subplots(company_data, 'Innovative_Traditional')['Type'],
                                y = plot_subplots(company_data, 'Innovative_Traditional')['Count'],
                                text = plot_subplots(company_data, 'Innovative_Traditional')['Count %']),
                        row=1, col=1)
            
            fig.add_trace(go.Bar(x = plot_subplots(company_data, 'exclusive_openness')['Type'],
                                y = plot_subplots(company_data, 'exclusive_openness')['Count'],
                                text = plot_subplots(company_data, 'exclusive_openness')['Count %']), 
                        row=1, col=2)
            
            fig.update_layout(
                        title= '<b>Company Archetypes</b>',
                        font=dict(
                            size=14,
                            color="Black"
                        )
                    )
            
            fig.update(layout_showlegend = False)
            
            
            save_image(fig, 'company_archetypes')
            
            combined_text_list.append(archetype_text(company_data, 'Innovative_Traditional'))
            combined_text_list.append(archetype_text(company_data, 'exclusive_openness'))
            combined_text_list.append(archetype_text(company_data, 'Introvert_Extrovert'))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Plot Personality
            col1, col2 = st.columns(2)
            
            with col1:
                data = plot_personality_bar(5, company_data)
                fig = px.bar(data, x="position", y="count", text="score")
                fig.update_layout(xaxis={'categoryorder':'total descending'},
                                title = '<b>Job Types - Score</b>',
                                font=dict(
                                    size=14,
                                    color="Black"
                                )
                )
                
                text_list = data.sort_values(by = 'count', ascending = False).values.tolist()
                combined_text_list.append(f'{text_list[0][0]} ({text_list[0][2]}) gives an average score of {text_list[0][1]}, \
{text_list[1][0]} ({text_list[1][2]}) an average of {text_list[1][1]},\
{text_list[2][0]} ({text_list[2][2]}) an average of {text_list[2][1]},\
{text_list[3][0]} ({text_list[3][2]}) an average of {text_list[3][1]}, \
and {text_list[4][0]} ({text_list[4][2]}) an average of {text_list[4][1]}')

                save_image(fig, 'job_types_avg_score')
                st.plotly_chart(fig, use_container_width=True)
            
            # Plot personality
            with col2:
                data = company_data.groupby(['Personality'])['score'].mean().round(2).reset_index()
                data = data.sort_values(by = 'score', ascending = False)
                
                fig = px.bar(data, 
                            x="Personality", 
                            y="score", 
                            text="score")
                
                fig.update_layout(xaxis={'categoryorder':'total descending'}, 
                                title = '<b>Personality Type - Score</b>',
                                font=dict(
                                    size=14,
                                    color="Black"
                                )
                )
                
                text_list = data.head(3).values.tolist()
                text_list2 = data.tail(3).values.tolist()
                combined_text_list.append(f'{text_list[0][0]}, {text_list[1][0]} and {text_list[2][0]} are the top 3 personality types that gives an average score of {text_list[0][1]}, {text_list[1][1]} and {text_list[2][1]} respectively')
                combined_text_list.append(f'{text_list2[0][0]}, {text_list2[1][0]} and {text_list2[2][0]} are the bottom 3 personality types that gives an average score of {text_list2[0][1]}, {text_list2[1][1]} and {text_list2[2][1]} respectively. This will indicate that employees of these personalities will require a strong leader to improve the low employee engagement. ')
                
                save_image(fig, 'personality_type_avg_score')
                st.plotly_chart(fig, use_container_width=True)                   
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.code('Positive Comments')
                plot_wordcloud(company_data, "pros")
                
            with col2:
                st.code('Negative Comments')
                plot_wordcloud(company_data, "cons")
            
            # Save to word document
            save_word(combined_text_list, name_of_company)      

            import aspose.words as aw
            doc = aw.Document('./report/Company Report.docx')
            options = aw.ImageWatermarkOptions()
                
            options.scale = 5
            options.is_washout = False
               
            doc.watermark.set_image("./images/whitebackground.jpg", options)
            doc.save('./report/Company Report.pdf')
            with open("./report/Company Report.pdf", "rb") as pdf_file:
                PDFbyte = pdf_file.read()
            
            import streamlit_ext as ste
            st.info('Click below to download report')
            ste.download_button("Download Report",
                                PDFbyte,
                                "./report/Company Report.pdf",
                                'application/octet-stream')

        if view_selection == "Upload Data":
            # Upload excel file
            uploaded_file = st.file_uploader("Choose file here")
            
            if uploaded_file is not None:
                try:
                    df1 = pd.read_excel(uploaded_file)
                    
                except ValueError as e:
                    df1 = pd.read_csv(uploaded_file)

                option = st.selectbox(
                        'What would you liket to see?',
                        ('Sentiment Analysis', 'Archetypes'))
                
                if option == 'Sentiment Analysis':
                    company_data = df1.copy()
                    company_data = preprocessing(company_data)
                    neutral, negative, positive = sentiment(company_data)

                    pie_chart = pd.DataFrame([['Neutral', neutral], ['Negative', negative], ['Positive', positive]], columns = ['Sentiment', 'Count'])
                    fig = px.pie(pie_chart, values='Count', names='Sentiment', title='Sentiment PIE chart')
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    company_data = df1.copy()                    
                    archetype_analysis(company_data)
                    plot_charts(company_data, 'Bytedance')
                    
        elif view_selection == "View Available Resources":

            company_options = list(all_company_data.keys())
            company_options.sort()
            company_opts = st.selectbox('Select your options', options = list(company_options))
            filter_data = all_company_data[company_opts]

            filter_data[['Employment Type', 'Duration']] = filter_data['year exp'].str.split(',', expand=True)
            plot_charts(filter_data, company_opts)

        else:
            comments_text = st.text_area('Text to analyze', '''ABC company is a great place to work, with endless networking opportunities. I adore being around people, and I hope that they can continue to work here forever.''')
            
            def create_probability_chart(type):
                list_df = []
                for label in type.samples():
                    temp_list = []
                    temp_list.append(label)
                    temp_list.append(type.prob(label))
                    list_df.append(temp_list)
                
                source = pd.DataFrame(list_df, columns = ['Type', 'Probability'])
                source['Probability'] = source['Probability'].round(2)
                
                fig = px.pie(source, values='Probability', names='Type')
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(
                        font=dict(
                            size=14,
                            color="Black"
                        )
                    )
                
                return st.plotly_chart(fig, use_container_width=True)
            
            if st.button('Click to submit'):
                tokenize = build_bag_of_words_features_filtered(comments_text)
                
                # Classify text
                ie_prob = IntroExtroClassifier.prob_classify(tokenize)
                it_prob = InnovativeTraditionalClassifier.prob_classify(tokenize)
                ns_prob = IntuitionSensingClassifier.prob_classify(tokenize)
                jp_prob = JudgingPerceivingClassifier.prob_classify(tokenize)
                tf_prob = ThinkingFeelingClassifier.prob_classify(tokenize)
                
                ie = IntroExtroClassifier.classify(tokenize)
                it = InnovativeTraditionalClassifier.classify(tokenize)
                ns = IntuitionSensingClassifier.classify(tokenize)
                jp = JudgingPerceivingClassifier.classify(tokenize)
                tf = ThinkingFeelingClassifier.classify(tokenize)
                
                def get_personality(personality_list):
                    personality_dict = {'Introvert': 'I', 
                                        'Extrovert': 'E', 
                                        'Judging': 'J',
                                        'Percieving': 'P',
                                        'Thinking': 'T', 
                                        'Feeling': 'F',
                                        'Intuition': 'N',
                                        'Sensing': 'S'}
                    personality = ''
                    for i in personality_list:
                        personality += personality_dict[i] 
                        
                    return personality
                
                st.code(f"Personality Type: {get_personality([ie, ns, tf, jp])}")
                
                checkbox = st.checkbox('View Statistics', value = True)
                if checkbox:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.code('Introvert-Extrovert')
                        create_probability_chart(ie_prob)
                        
                    with col2:
                        st.code("Intuition-Sensing")
                        create_probability_chart(ns_prob)
                        
                    with col3:
                        st.code("Judging-Perceiving")
                        create_probability_chart(jp_prob)
                        
                    with col4:
                        st.code("Thinking-Feeling")
                        create_probability_chart(tf_prob)
                
                df = pd.DataFrame({'pros': comments_text, 'cons': ""}, index = [0])
                
                df = emotions_lexicon(df, emotion_lexicon_df)
                df = pd.DataFrame([['Exclusive', df.loc[0, 'is_exclusive']], ['Open', df.loc[0, 'is_open']]], columns = ['Type', 'Count'])
                
                col5, col6 = st.columns(2)
                with col5:
                    st.code('Exclusive-Open')
                    fig = px.pie(df, values='Count', names='Type')
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    fig.update_layout(
                            font=dict(
                                size=14,
                                color="Black"
                            )
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                with col6:
                    st.code('Innovative-Traditional')
                    create_probability_chart(it_prob)
                       

    # --- USER AUTHENTICATION ---
    names = ["Certis Admin", "Zhen Yuen"]
    usernames = ["certis_admin", "zhenyuen97"]

    # Load hashed passwords
    file_path = Path(__file__).parent / "hashed_pw.pkl"
    with file_path.open("rb") as file:
        hashed_passwords = pickle.load(file)

#     authenticator = stauth.Authenticate(names, usernames, hashed_passwords, "sales_dashboard", "abcdef")
#     name, authentication_status, username = authenticator.login("ANL488 Archetype Simulator \n Login", "main")

#     if authentication_status == False:
#         st.error("Username/password is incorrect")

#     if authentication_status == None:
#         st.warning("Please enter your username and password")

#     if authentication_status:
    page_names_to_funcs = {
        "Main Page": app
    }

    selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
    page_names_to_funcs[selected_page]()

if __name__ == "__main__":
    main()
