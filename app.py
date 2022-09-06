import streamlit as st
import pandas as pd
import numpy as np

from datetime import timedelta
import datetime
import streamlit_authenticator as stauth
import pickle
from pathlib import Path
import altair as alt
from PIL import Image
import plotly.express as px

import selenium as sn

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

image = Image.open('SUSS logo.png')

st.set_page_config(layout="wide")

def main():
    def app():
    
        # Initialize connection.
        # Uses st.experimental_singleton to only run once.
        @st.experimental_singleton
        def init_connection():
            return mysql.connector.connect(**st.secrets["mysql"])

        conn = init_connection()

        # Perform query.
        # Uses st.experimental_memo to only rerun when the query changes or after 10 min.
        @st.experimental_memo(ttl=600)
        def run_query(query):
            with conn.cursor() as cur:
                cur.execute(query)
                return cur.fetchall()

        rows = run_query("SELECT * from Bytedance_Glassdoor_Reviews;")
        glassdoor_reviews = pd.DataFrame(rows)
        glassdoor_reviews.columns = ['date', 'year exp', 'score','pros', 'cons', 'position']

        rows = run_query("SELECT * from Innovative_Traditional_Companies;")
        innovative_traditional_posts = pd.DataFrame(rows)
        innovative_traditional_posts.columns = ['posts', 'company', 'type']

        rows = run_query("SELECT * from NRC_Emotion_Lexicon;")
        emotion_lexicon_df = pd.DataFrame(rows)
        emotion_lexicon_df.columns = ['Word', 'Positive', 'Negative', 
                                      'Anger', 'Anticipation', 'Disgust',
                                      'Fear', 'Joy', 'Sadness', 'Surprise', 'Trust']
        
        # Models
        innovative_classifier = open('./Models/InnovativeTraditionalClassifier.pickle', 'rb')
        InnovativeTraditionalClassifier = pickle.load(innovative_classifier)
        innovative_classifier.close()

        IE_classifier = open('./Models/IntroExtroClassifier.pickle', 'rb')
        IntroExtroClassifier = pickle.load(IE_classifier)
        IE_classifier.close()
        
        NS_classifier = open('./Models/IntuitionSensing.pickle', 'rb')
        IntuitionSensingClassifier = pickle.load(NS_classifier)
        NS_classifier.close()
        
        
        JP_classifier = open('./Models/JudgingPerceiving.pickle', 'rb')
        JudgingPerceivingClassifier = pickle.load(JP_classifier)
        JP_classifier.close()
        
        TF_classifier = open('./Models/ThinkingFeeling.pickle', 'rb')
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
                    company_data = emotions_lexicon(company_data, emotion_lexicon_df)
                    
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
                    company_data['Personality_JP'] = company_data['Judging_Perceiving'].apply(lambda x: 'J' if x == 'Judging' else 'P')
                    company_data['Personality_TF'] = company_data['Thinking_Feeling'].apply(lambda x: 'T' if x == 'Thinking' else 'F')
                    company_data['Personality'] = company_data['Personality_IE'] + company_data['Personality_NS'] + company_data['Personality_JP'] + company_data['Personality_TF']
                    company_data.drop(columns = ['Personality_IE', 'Personality_NS', 'Personality_JP', 'Personality_TF'], inplace = True)
                    
                    emotions_list = ['Anger', 'Anticipation', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise', 'Trust']
                    values_list = company_data[emotions_list].sum().tolist()
                    data_list = []
                    for i in range(len(emotions_list)):
                        temp_list = []
                        temp_list.append(emotions_list[i])
                        temp_list.append(values_list[i])
                        data_list.append(temp_list)
                    
                    st.code('Number of unique jobs: ' + str(len(company_data['position'].unique())))              
                    st.dataframe(company_data)
                        
                    col1, col2 = st.columns(2)
                    with col1:
                        pie_chart = pd.DataFrame(data_list, columns = ['Emotions', 'Count'])
                        plot_piechart(pie_chart, 'Emotions Pie Chart', 'Emotions')
                    
                    with col2:
                        pie_chart2 = pd.DataFrame(company_data['Personality'].value_counts().reset_index().values.tolist(), columns = ['Type', 'Count'])
                        plot_piechart(pie_chart2, 'Personality Type', 'Type')
                                      
                    
                    fig = make_subplots(rows=1, cols=3,
                                        subplot_titles=["Innovative vs Traditional",
                                                        "Open vs Exclusive",
                                                        "Extrovert vs Introvert"]
                                    )
                    fig.add_trace(go.Bar(x = plot_subplots(company_data, 'Innovative_Traditional')['Type'],
                                         y = plot_subplots(company_data, 'Innovative_Traditional')['Count'],
                                        text = plot_subplots(company_data, 'Innovative_Traditional')['Count %'],
                                        name="Position"),
                                row=1, col=1)
                    
                    fig.add_trace(go.Bar(x = plot_subplots(company_data, 'exclusive_openness')['Type'],
                                        y = plot_subplots(company_data, 'exclusive_openness')['Count'],
                                        text = plot_subplots(company_data, 'exclusive_openness')['Count %'],
                                        name="Position"), 
                                  row=1, col=2)
                    
                    fig.add_trace(go.Bar(x = plot_subplots(company_data, 'Introvert_Extrovert')['Type'],
                                        y = plot_subplots(company_data, 'Introvert_Extrovert')['Count'],
                                        text = plot_subplots(company_data, 'Introvert_Extrovert')['Count %'],
                                        name="Position"),
                                row=1, col=3)
                    
                    fig.update_layout(
                                title= '<b>Company Archetypes</b>',
                                font=dict(
                                    size=14,
                                    color="Black"
                                )
                            )
                    
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
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Plot personality
                    with col2:
                        fig = px.bar(company_data.groupby(['Personality'])['score'].mean().round(2).reset_index(), 
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
                        st.plotly_chart(fig, use_container_width=True)                   
                    
        elif view_selection == "View Available Resources":
            company_options = list(innovative_traditional_posts['company'].unique())
            company_options.sort()
            company_opts = st.selectbox('Select your options', options = list(company_options))
            filter_data = innovative_traditional_posts[innovative_traditional_posts['company'] == company_opts]

            st.dataframe(filter_data.head())

        else:
            comments_text = st.text_area('Text to analyze', '''Working at ABC company is very tiring, with endless socialising and networking. I really hate being around people and I hope people can just leave me alone''')
                        
            if st.button('Click to submit'):
                tokenize = build_bag_of_words_features_filtered(comments_text)
    
                # Classify text
                prob_dist = IntroExtroClassifier.prob_classify(tokenize)
                list_df = []
                for label in prob_dist.samples():
                    temp_list = []
                    temp_list.append(label)
                    temp_list.append(prob_dist.prob(label))
                    list_df.append(temp_list)
                    # st.code("%s: %f" % (label, prob_dist.prob(label)))
                
                source = pd.DataFrame(list_df, columns = ['Type', 'Probability'])
                source['Probability'] = source['Probability'].round(2)
                
                prob_dist = InnovativeTraditionalClassifier.prob_classify(tokenize)
                list_df = []
                for label in prob_dist.samples():
                    temp_list = []
                    temp_list.append(label)
                    temp_list.append(prob_dist.prob(label))
                    list_df.append(temp_list)
                    # st.code("%s: %f" % (label, prob_dist.prob(label)))
                
                source2 = pd.DataFrame(list_df, columns = ['Type', 'Probability'])
                source2['Probability'] = source2['Probability'].round(2)

                ie = IntroExtroClassifier.classify(tokenize)
                it = InnovativeTraditionalClassifier.classify(tokenize)
                
                # Plot charts
                chart = alt.Chart(source).mark_bar().encode(
                    x='Probability:Q',
                    y='Type:O'
                ).properties(height = 300)
                
                text = chart.mark_text(
                    align='left',
                    baseline='middle',
                    dx=3  # Nudges text to right so it doesn't appear on top of the bar
                ).encode(
                    text='Probability:Q'
                )
                
                chart2 = alt.Chart(source2).mark_bar().encode(
                    x='Probability:Q',
                    y='Type:O'
                ).properties(height = 300)
                
                text2 = chart2.mark_text(
                    align='left',
                    baseline='middle',
                    dx=3  # Nudges text to right so it doesn't appear on top of the bar
                ).encode(
                    text='Probability:Q'
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.code("Comments Type: " + ie)
                    st.altair_chart(chart + text, use_container_width=True)
                    
                with col2:
                    st.code("Post Type: " + it)
                    st.altair_chart(chart2 + text2, use_container_width=True)

    # --- USER AUTHENTICATION ---
    names = ["Certis Admin", "Zhen Yuen"]
    usernames = ["certis_admin", "zhenyuen97"]

    # Load hashed passwords
    file_path = Path(__file__).parent / "hashed_pw.pkl"
    with file_path.open("rb") as file:
        hashed_passwords = pickle.load(file)

    authenticator = stauth.Authenticate(names, usernames, hashed_passwords, "sales_dashboard", "abcdef")
    name, authentication_status, username = authenticator.login("ANL488 Archetype Simulator \n Login", "main")

    if authentication_status == False:
        st.error("Username/password is incorrect")

    if authentication_status == None:
        st.warning("Please enter your username and password")

    if authentication_status:
        page_names_to_funcs = {
            "Main Page": app
        }

        selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
        page_names_to_funcs[selected_page]()

if __name__ == "__main__":
    main()