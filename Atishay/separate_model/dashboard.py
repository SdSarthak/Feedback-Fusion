import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

only_com = pd.read_csv('output_01.csv')

only_com['emoji_count'] = only_com['Emojis'].apply(lambda x: len(x) if isinstance(x, str) else 0)

def analyze_emojis(only_com):
    emoji_list = []
    for emoji_str in only_com['Emojis']:
        if isinstance(emoji_str, str):
            emoji_list.extend(list(emoji_str))
        
    return dict(Counter(emoji_list))


emoji_analysis = analyze_emojis(only_com)


def visualize_emojis(emoji_analysis):
    # Ensure we get keys and values as lists
    emoji_analysis = dict(sorted(emoji_analysis.items(), key=lambda item: item[1], reverse=True))
    emojis = list(emoji_analysis.keys())
    frequencies = list(emoji_analysis.values())
    
    # Get the top 10 emojis and frequencies
    top_emojis = emojis[:10]
    top_frequencies = frequencies[:10]
    
    # Create a DataFrame for Streamlit
    emoji_df = pd.DataFrame({'Emoji': top_emojis, 'Frequency': top_frequencies})
    emoji_df['Frequency'] = emoji_df['Frequency'].sort_values(ascending = False)
    # Show the bar chart in Streamlit
    st.bar_chart(emoji_df.set_index('Emoji'))
    
    
    
st.title("YouTube Comments Analysis")
st.header("Sentiment Analysis Results")

# Display Sentiments
if 'Polarity' in only_com.columns:
    st.bar_chart(only_com['Polarity'])
else:
    st.write("Polarity data not available.")

# Display Emoji Frequencies
st.header("Top 10 Most Frequently Used Emojis")

visualize_emojis(emoji_analysis)