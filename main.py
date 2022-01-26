import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

st.markdown(
    """
    <style>
    .main {
        background-color: #F5F5F5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache
def get_data(filename):
    taxi_data = pd.read_csv(filename) if ".gz" not in filename.lower() else pd.read_csv(filename, compression='gzip')
    return taxi_data
with header:
    st.title("Welcome to my data science project")
    st.text('In this project I look into the transactions of taxis in NYC. ...')

with dataset:
    st.header("NYC taxi dataset")
    st.text("I found this dataset on http..blah")
    taxi_data = get_data('data/yellow_tripdata_2021-01.csv.gz')
    st.write(taxi_data.head(5))

    st.subheader('Pick up location ID distribition on the NYC dataset')
    pulocation_dist = pd.DataFrame(taxi_data['PULocationID'].value_counts()).head(50)
    st.bar_chart(pulocation_dist)
with features:
    st.header("The features I created")
    st.markdown('* **first feature:** I created this feature because of this.. I calculated it using logic x')
    st.markdown('* **second feature:** I created this feature because of this.. I calculated it using logic x')

with model_training:
    st.title("Time to train the model")
    st.text('Here you get to choose the hyperparameters of the model and see how the performance changes')

    sel_col, disp_col = st.columns(2)

    max_depth = sel_col.slider('What shoudl be the max_depth of the model?', min_value=10, max_value=100, value=20, step=10)

    n_estimators = sel_col.selectbox('How many trees should there be?', options=[100,200,300,'No Limit'])

    sel_col.text('Here is a list of features in my data')
    sel_col.write(taxi_data.columns)


    input_feature = sel_col.text_input('Which feature should be used as the input feature?', 'PULocationID')

    regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators) if n_estimators != 'No Limit' else  RandomForestRegressor(max_depth=max_depth)

    X = taxi_data[[input_feature]]
    y = taxi_data[['trip_distance']]

    regr.fit(X, y)
    prediction = regr.predict(y)

    disp_col.subheader('Mean Absolute error of the model is:')
    disp_col.write(mean_absolute_error(y, prediction))
    disp_col.subheader('Mean squared error of the model is:')
    disp_col.write(mean_squared_error(y, prediction))
    disp_col.subheader('R Squared error of the model is:')
    disp_col.write(r2_score(y, prediction))


