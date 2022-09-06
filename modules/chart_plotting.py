import plotly.express as px
import streamlit as st
import pandas as pd

def plot_piechart(pie_chart, text, name):
    fig = px.pie(pie_chart, values='Count', names=name)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
            title= f'<b>{text}</b>',
            font=dict(
                size=14,
                color="Black"
            )
        )
    st.plotly_chart(fig) 
    
def plot_subplots(data, column_name):
    data = pd.DataFrame(data[column_name].value_counts(normalize=True).reset_index().values.tolist(), columns = ['Type', 'Count'])
    data['Count %'] = data['Count']*100
    data['Count %'] = data['Count %'].round(2)
    data['Count %'] = data['Count %'].astype(str) + "%"
    return data

def plot_personality_bar(top_n, company_data):
    # Get top 5 roles in a list
    top5_list = company_data['position'].value_counts()[:top_n].index.tolist()
    
    # Filter to list
    filter_dataframe = company_data[company_data['position'].isin(top5_list)]
    
    # Calculate glassdoor score given
    avg_score_df = filter_dataframe.groupby(['position'])['score'].mean().round(2).reset_index()
    
    # Count number of unique position commented
    count_df = filter_dataframe['position'].value_counts().round(2).reset_index()
    count_df.columns = ['position', 'count']
    
    # Combine results
    combined_df = pd.merge(avg_score_df, count_df, on = ['position'], how = 'left')
    
    return combined_df