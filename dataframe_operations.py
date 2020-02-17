__author__ = "Amaral LAN and Moreira JAG"
__copyright__ = "Copyright 2018, 2019, Amaral LAN and Moreira JAG"
__credits__ = ["Amaral LAN", "Moreira JAG"]
__version__ = "1.0"
__maintainer__ = "Amaral LAN"
__email__ = "amaral@northwestern.edu"
__status__ = "Production"


import json
import pandas as pd

from copy import deepcopy
from os import path, getcwd
from numpy import array, floor, log10



def load_dataset(dataset):
    """
    Loads a json dataset as a pandas DataFrame.
    """
    valid = {'movies_final', 'career_restricted'}
    if dataset not in valid:
        raise ValueError(f"`dataset` must be one of: ({', '.join(valid)})")

    json_dir = path.abspath(path.join(getcwd(), 'Json_files'))
    with open(path.join(json_dir, f"{dataset}.json"), 'r') as js_file:
        data_json = json.load(js_file)

    return pd.DataFrame(data_json)


def get_movies_df(role_key):
    """
    Takes role_key as role in movie and excluded_genres as list of genres to exclude from analysis

    Returns a pandas DataFrame with all movies, and actors gender information.
    """
    movies_df = load_dataset("movies_final")
    print('\nLoaded IMDb movies {}'.format(role_key))
    return movies_df.dropna(subset=[role_key])


def get_movie_budgets_df():
    """
    Takes role_key as role in movie

    Returns a pandas DataFrame with the year, director info, and budget of all
    movies with a single director.
    """
    movies_df = load_dataset("movies_final")
    return movies_df.dropna(subset=["adjusted_budget"])


def get_genre_movies_df(genre, role_key):
    """
    Takes tv_flag is whether we want tv movies or non-tv movies
    Takes genre

    Returns a pandas DataFrame with all movies with the given genre,
    and actors gender information.
    """
    movies_df = load_dataset("movies_final")
    valid_movies_df = movies_df.dropna(subset=[role_key])
    return valid_movies_df[valid_movies_df.genres.apply(lambda x: genre in x)]


def prepare_dataframe(df, concentration_key, CONCENTRATION, GENRES, TIME_DUMMIES, PERIOD):
    """
    Prepare smaller dataframe with all relevant variables for multivariate regression

    :param: df: dataframe with movie information
    :param: concentration_key: string with key
    :param: CONCENTRATION: dictionary of lists with names for concentration measure
    :param: GENRES: list of genres
    :param: TIME_DUMMIES: array with index of time dummies
    :param: PERIOD: int with duration of time dummies

    :return: movies_df: dataframe for further analysis
    """
    movies_df = deepcopy(df)

    json_dir = path.abspath(path.join(getcwd(), 'Json_files'))
    with open(path.join(json_dir, 'employment_interpolated.json'), 'r') as file_in:
        employment = json.load(file_in)
    new_column = array([employment[j - 1910] for j in movies_df['year']])
    movies_df = movies_df.assign(g = new_column)
    movies_df.rename(columns = {'g': 'female_workforce_participation'}, inplace = True)

    with open(path.join(json_dir, 'real_gdp_per_capita_growth_1910-2010.json'), 'r') as file_in:
        gdp_growth = json.load(file_in)
    new_column = array([gdp_growth[j - 1910] for j in movies_df['year']])
    movies_df = movies_df.assign(g = new_column)
    movies_df.rename(columns = {'g': 'gdp_growth'}, inplace = True)

    with open(path.join(json_dir, 'number_movies.json'), 'r') as file_in:
        no_movies = json.load(file_in)
    new_column = array([no_movies[j - 1911] if j > 1910 else 0 for j in movies_df['year']])
    movies_df = movies_df.assign(g = new_column)
    movies_df.rename(columns={'g': 'number_movies'}, inplace = True)

    with open(path.join(json_dir, f'{CONCENTRATION[concentration_key][1]}.json'), 'r') as file_in:
        concentration = json.load(file_in)
    concentration.append(concentration[-1])
    new_column = array([1000*concentration[j - 1911] if j > 1910 else 0 for j in movies_df['year']])
    movies_df = movies_df.assign(g = new_column)
    movies_df.rename(columns={'g': 'concentration'}, inplace = True)

    print(f"There are {len(employment)} values for female workforce participation, "
          f"{len(no_movies)} for number of released movies, and {len(concentration)} for industry concentration.\n")

    # Compile genre strings for formula and add columns to df
    #
    movies_df = add_genre_columns(GENRES, movies_df)

    # Compile decade dummies for formula and add columns to df
    #
    movies_df = add_time_dummies(TIME_DUMMIES, PERIOD, movies_df)

    movies_df['directing_fraction'] = movies_df['directing_gender_percentage'] / 100.
    movies_df['cinematography_fraction'] = movies_df['cinematography_gender_percentage'] / 100.
    movies_df['producing_fraction'] = movies_df['producing_gender_percentage'] / 100.
    movies_df['log10_budget'] = log10(movies_df['adjusted_budget'])

    return movies_df


def add_genre_columns(genres, df):
    for g in genres:
        new_colum = array([1 if g in df.iloc[i].genres else 0 for i in range(len(df))])
        df = df.assign( g = new_colum )
        df.rename(columns = {'g': g.replace('-', '')}, inplace = True)
    return df


def add_time_dummies(dummies, period, df):
    for k, dummy in enumerate(dummies):
        new_column = array([1 if get_period(df.iloc[i].year, period) == k else 0 for i in range(len(df))])
        df = df.assign( g = new_column )
        df.rename(columns={'g': dummy}, inplace = True)
    return df


def get_period(year, period):
    """
    """
    return int( floor((year - 1911)/period) )


def get_decade(year):
    """
    """
    return int( floor((year - 1900)/10) )*10



print(get_decade(1910) )

print(get_period(2010, 2))




