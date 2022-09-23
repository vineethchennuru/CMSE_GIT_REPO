import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')

# Dropping Id as it is not of any use

df.drop('id',axis=1,inplace=True)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(df.diagnosis)
df.diagnosis = le.transform(df.diagnosis)

from sklearn.preprocessing import StandardScaler
df_features = df.iloc[:,1:]
scaler = StandardScaler()
scaler.fit(df_features)

features_scaled = scaler.transform(df_features)
features_scaled = pd.DataFrame(data=features_scaled,
                               columns=df_features.columns)

df = pd.concat([features_scaled, df['diagnosis']], axis=1)


df_scaled_melt = pd.melt(df, id_vars='diagnosis',
                         var_name='features', value_name='value')

def violin_plot(features, name):
    """
    This function creates violin plots of features given in the argument.
    """
    # Create query
    if features == []:
        plt.figure()
        return
    query = ''
    for x in features:
        query += "features == '" + str(x) + "' or "
    query = query[0:-4]

    # Create data for visualization
    data = df_scaled_melt.query(query)

    # Plot figure
    plt.figure(figsize=(20, 15))
    sns.violinplot(x='features',
                   y='value',
                   hue='diagnosis',
                   data=data,
                   split=True,
                   inner="quart")
    plt.xticks(rotation=45)
    plt.title(name)
    plt.xlabel("Features")
    plt.ylabel("Standardize Value")



st.title('CMSE 830 ICA 5', anchor=None)
str_text = ':Attribute Information:\n        - radius (mean of distances from center to points on the perimeter)\n        - texture (standard deviation of gray-scale values)\n        - perimeter\n        - area\n        - smoothness (local variation in radius lengths)\n        - compactness (perimeter^2 / area - 1.0)\n        - concavity (severity of concave portions of the contour)\n        - concave points (number of concave portions of the contour)\n        - symmetry\n        - fractal dimension ("coastline approximation" - 1)\n\n        The mean, standard error, and "worst" or largest (mean of the three\n        worst/largest values) of these features were computed for each image,\n        resulting in 30 features.  For instance, field 0 is Mean Radius, field\n        10 is Radius SE, field 20 is Worst Radius.\n\n        - class:\n                - WDBC-Malignant\n                - WDBC-Benign\n\n  '
st.text(str_text)
st.write('Breast cancer wisconsin (diagnostic) dataset')


col1, col2 = st.columns(2,gap='large')

with col1:
    options = st.multiselect(
        'Features',df.columns[:-1],default=['radius_mean','texture_mean','perimeter_mean']
        )

    st.set_option('deprecation.showPyplotGlobalUse', False)
    # st.write('You selected:', options)

    fig2 = violin_plot(options, "Violin Plot")
    st.header("Violin Plot")
    st.pyplot(fig2)

with col2:
    column_name =  st.selectbox(
        'For which column would you like to see the distplot',
        df.columns[:-1])

    # st.write('You selected:', column_name)


    plt.figure(figsize=(20, 15))

    fig = sns.displot(df[column_name])

    st.header("Simple Distplot")
    st.pyplot(fig)

st.text("")
st.text("")
st.text("")
st.text("")
st.text("")
st.text("")
st.text("")
st.text("")

col3,col4 = st.columns(2,gap='large')

with col3:
    options = st.multiselect(
        'Features',df.columns[:-1],default=['radius_mean','texture_mean','perimeter_mean'],key=10
        )

    st.set_option('deprecation.showPyplotGlobalUse', False)
    # st.write('You selected:', options)

    fig3 = sns.pairplot(
    pd.concat([df[options],
               df['diagnosis']],
              axis=1),hue='diagnosis')    

    st.header("Simple Pair plot")
    st.pyplot(fig3)

with col4:
    column_name_2 =  st.selectbox(
        'For which column would you like to see the distplot',
        df.columns[:-1],
        key=2
        )

    # st.write('You selected:', column_name)


    plt.figure(figsize=(20, 15))

    fig4 = sns.displot(df, x=column_name_2, hue="diagnosis", kind="kde", multiple="stack")

    st.header("KDE plot with hue as diagnosis feature")
    st.pyplot(fig4)


# st.text("")
# st.text("")
# st.text("")
# st.text("")
# st.text("")
# st.text("")
# st.text("")
# st.text("")

# left, middle, right = st.columns((2, 5, 2))
# with middle:

#     corr_df = df.corr()
#     # Create mask
#     mask = np.zeros_like(corr_df, dtype=np.bool8)
#     mask[np.triu_indices_from(mask, k=1)] = True

#     # Plot heatmap
#     corr_plot = px.imshow(corr_df,width=700, height=700)
#     st.header("Correlation plot between all features")
#     st.plotly_chart(corr_plot)



st.text("")
st.text("")
st.text("")
st.text("")
st.text("")
st.text("")
st.text("")
st.text("")
st.text(' The below is a first 10 rows of the dataset')
st.dataframe(df.head(10))

