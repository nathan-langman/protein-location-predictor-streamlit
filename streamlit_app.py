import streamlit as st
import pandas as pd
from pathlib import Path
import config
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os

import seaborn as sns

DATA_FILEPATH = Path(__file__).parent/'data'
print(DATA_FILEPATH)

st.set_page_config(
    page_title='Prot-loc-pred',
    page_icon=':microscope:',
)

@st.cache_data
def get_full_data():
    data_filename = os.path.join(DATA_FILEPATH, 'model_scoring.tsv')
    df = pd.read_csv(data_filename, sep='\t', index_col=0)
    return df


@st.cache_data
def get_data_cleaning_stats():
    data_filename = os.path.join(DATA_FILEPATH, 'data_cleaning_stats.json')
    df = pd.read_json(data_filename, typ='series')
    df.name='Protein counts'
    return df

full_df = get_full_data()
cleaning_stats = get_data_cleaning_stats()

st.title(":microscope: Protein Location Predictor")
st.write("Here we plot charts related to the initial data gathering and statistics for the proteins in our dataset.")

st.subheader("Some example data")
st.dataframe(full_df.head())


def plot_data_cleaning_stats():
    fig, ax = plt.subplots()
    sns.barplot(cleaning_stats/1000,ax=ax)
    plt.xticks(rotation=45, ha='right')
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.1f}k'.format(x) for x in vals])
    for i in ax.containers:
        ax.bar_label(i,fmt='{:,.1f}k')

    plt.close(fig)
    return fig

st.subheader("Data cleaning")
fig = plot_data_cleaning_stats()
st.pyplot(fig)
st.write("We created this dataset by joining Uniref cluster data with sequence and location data for the representative protein of each cluster from uniprot.")


def plot_location_counts():
    fig, ax = plt.subplots()
    mass_cnts = {}
    for c in config.locs:
        mass_cnts[c] = full_df[full_df[c]]['Mass'].count()
    df = pd.Series(mass_cnts, name='Protein counts')
    sns.barplot(df/1000,ax=ax)
    plt.xticks(rotation=45, ha='right')
    ax.set_xlabel('Proteins with this location')
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.1f}k'.format(x) for x in vals])
    for i in ax.containers:
        ax.bar_label(i,fmt='{:,.1f}k')

    plt.close(fig)
    return fig

st.subheader("Location counts")
fig = plot_location_counts()
st.pyplot(fig)


def plot_multiple_locs():
    fig, ax = plt.subplots()
    multi_loc = sum(full_df['locations']>1)
    single_loc = sum(full_df['locations']==1)
    ax.pie([multi_loc, single_loc], labels=['Multiple Locations', 'Single Location'], autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    plt.close(fig)
    return fig


st.subheader("Multiple locations")
fig = plot_multiple_locs()
st.pyplot(fig)
st.write("Fig X: Pie chart showing the proportion of proteins with a single location vs multiple locations to predict.")

st.header("Features")
def amino_acid_plot():
    fig, ax = plt.subplots()
    sns.boxplot(full_df[config.amino_acid_cols],fliersize=2,ax=ax)
    ax.set_xticklabels([a[0] for a in config.amino_acid_cols])
    ax.set_ylabel("Amino acid percent")
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    plt.close(fig)
    return fig

fig = amino_acid_plot()
st.subheader("Amino acid composition")
st.pyplot(fig)
st.write("Fig X: The distribution of amino acid composition across proteins. We can see that some amino acids are more common than others.")

def mass_plot(mass_scale):
    log_scale = mass_scale == 'Log'
    fig, ax = plt.subplots()
    sns.histplot(full_df['Mass'], ax=ax, log_scale=log_scale)
    # ax.set_title("Protein mass distribution")
    ax.set_ylabel("Mass")
    plt.close(fig)
    return fig


st.subheader("Protein mass distribution")
mass_scale = st.selectbox(
    "X axis scale",
    ("Linear", "Log"),
)
fig = mass_plot(mass_scale)
st.pyplot(fig)
st.write("Fig X: A histogram of the mass of each protein in the dataset. Use the selector to swap between log and linear x axis.")


# Diving into specific locations
st.header("Explore location specific data")
location = st.selectbox("Location", config.locs)
metric = st.selectbox("Metric", config.metrics)
loc_chart_scale = st.selectbox(
    key='loc_chart_x_axis_scale',
    label="X axis scale",
    options=("Linear", "Log"),
)
loc_data = full_df[full_df.loc[:,location]][metric].dropna().values
all_data = full_df.loc[:,metric].dropna().values

def correlation_plot(loc_chart_scale):
    log_scale = loc_chart_scale == 'Log'
    fig, ax = plt.subplots()
    sns.histplot(loc_data,ax=ax,alpha=0.5, stat='density', log_scale=log_scale)
    sns.histplot(all_data,ax=ax,alpha=0.5, stat='density', log_scale=log_scale)
    ax.set_xlabel(metric)
    ax.legend([location,'all proteins'])
    plt.close(fig)
    return fig

fig = correlation_plot(loc_chart_scale)

st.subheader(f"{location} vs all proteins distribution for {metric}")
st.pyplot(fig)
st.write("Fig X: The distribution of amino acid composition across proteins. We can see that some amino acids are more common than others.")

st.header("Model performance")
st.write("Here we can see the performance of each model as well as a deep dive into the performance of the best model (the neural network approach), broken down by some factors.")
