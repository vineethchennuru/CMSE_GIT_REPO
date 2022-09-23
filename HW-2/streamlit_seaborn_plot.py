import streamlit as st
import seaborn as sns

mpg_df = sns.load_dataset('mpg')

fig = sns.pairplot(mpg_df,hue='origin')

st.pyplot(fig)