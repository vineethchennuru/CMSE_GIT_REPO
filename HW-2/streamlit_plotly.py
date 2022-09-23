import streamlit as st
import numpy as np

import plotly.express as px
df = px.data.iris()
sfig = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_width',
                    color='petal_length', symbol='species',title='IRIS data represented in 3D scatter plot')
fig
fig.show()

# Plot!
st.plotly_chart(fig, use_container_width=True)