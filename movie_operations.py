__author__ = "Amaral LAN and Moreira JAG"
__copyright__ = "Copyright 2018, 2019, Amaral LAN and Moreira JAG"
__credits__ = ["Amaral LAN", "Moreira JAG"]
__version__ = "1.0"
__maintainer__ = "Amaral LAN"
__email__ = "amaral@northwestern.edu"
__status__ = "Production"

import collections
import json
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf

from random import choice
from My_libraries.my_stats import risk_ratio, significant_digits, star
from matplotlib.patches import Rectangle
from matplotlib.font_manager import FontProperties
from My_libraries.my_stats import window, sem
from numpy import array, nan, std, log2, arange, linspace, log10

from My_libraries.my_stats import bottom_frame, half_frame, place_commas
from dataframe_operations import get_movies_df, get_genre_movies_df, get_decade



def convert_gender(gender):
    """
    Converts gender to a numerical scale:
    0 - male
    1 - female
    2 - undetermined
    """
    if gender == "male":
        return 0
    elif gender == "female":
        return 1
    else:
        return 2


def calc_staff_gender_percent(role):
    """
    Takes role as dataframe with information on specific staff role (director, producer, etc)

    Calculates the gender percent of a movie's role crew.
    The percent only includes staff for which we know the gender.
    """
    females = sum(1 for p in role if p["gender"] == "female")
    males = sum(1 for p in role if p["gender"] == "male")
    try:
        return 100*females/(males + females)
    except ZeroDivisionError:
        return nan


def get_staff_gender_df(movies_df, role):
    """
    Returns a pandas DataFrame with the yearly mean and standard error of the
    role gender fraction for all movies with role information.
    """
    key = role + "_gender_p"

    movies_df[key] = movies_df[role].apply(calc_staff_gender_percent)

    key = movies_df.groupby("year")[key].agg(
        {"mean": "mean", "sem": sem}
    )

    return key


def get_directors_gender_df(directors_df):
    """
    Calculates the percent of movies directed by females over time from the
    input director data.
    """
    gender_raw = []
    for _, d in directors_df.iterrows():
        for m in d["movies_list"]:
            gender_raw.append({
                "year": m["year"],
                "dir_gender": d["gender"],
                "movie": m["movie_id"],
                "dir_id": d["_id"],
            })
    gender_df = pd.DataFrame(gender_raw)

    # Group by year and pivot on gender, i.e.,
    # count occurrences of each gender per year
    gender_pivot = pd.pivot_table(
        gender_df, index='year', columns='dir_gender', aggfunc=len
    )
    # The pivot repeats information in column groups.
    # Any of `dir_id`,`movie` groups will have the same info (these
    # are the columns from `gender_raw` that are not used in created the pivot)
    gender_pivot = gender_pivot["dir_id"]

    # Convert absolute numbers to percent
    return (gender_pivot.female/gender_pivot.sum(axis=1)).dropna()*100


def get_active_gender_percent(elements_df, left, right):
    """
    Calculates the gender percent of active elements (actors, producers, or
    directors) in each year y. Elements are considered active in year y if
    `left` <= y <= `right`.

    Gender percent is relative to females:
        100% means all female;
        0% means all male.
    """
    y_min = int(elements_df[left].min())
    y_max = int(elements_df[right].max())

    active_dir = {"male": {}, "female": {}}
    for gender, gender_active in active_dir.items():
        gender_df = elements_df.loc[elements_df.gender == gender]
        for year in range(y_min, y_max+1):
            gender_active[year] = sum(
                (gender_df[left] <= year) & (gender_df[right] >= year)
            )

    active_df = pd.DataFrame(active_dir)

    return 100*active_df.female/(active_df.male+active_df.female)


def roll_mean_sem_movies(key, df, date_range, w_width, count = False):
    """
    Calculate the rolling mean and stderr for the given movies df.

    Params:
        role       : String with type of role in movie
        df         : Dataframe with 'year' as one of the keys
        date_range : List/array of unique, sorted datetime year values
                     to roll over.
        w_width    : Integer specifying rolling window width.

    Returns (Series with mean, Series with sem)
    """
    series_mean = {}
    series_sem = {}

    # print(df.keys(), len(df))
    if count:
        new_column = array([len(df.iloc[i][key]) if len(df.iloc[i][key]) > 0 else nan for i in range(len(df))])
        df = df.assign(counter=new_column)
        pkey = 'counter'
    else:
        pkey = key

    df.sort_values(by="year", inplace=True)

    for w in window(date_range, w_width):
        rolling_df = df[(df.year >= w[0]) & (df.year <= w[-1])]
        mean = rolling_df[pkey].mean()

        if pd.isnull(mean):
            continue
        series_mean[w[-1]] = mean
        series_sem[w[-1]] = sem(rolling_df[pkey])

    return (pd.Series(series_mean), pd.Series(series_sem))


def calculate_conc_indices(list_companies):

    total = len(list_companies)
    col = collections.Counter(list_companies)
    entropy = 0.
    gini = 0.
    herfindahl = 0.
    cumulative = 0.
    for co, number in reversed(col.most_common()):
        p = number / total
        entropy += -p * log2(p)
        cumulative += p
        gini += cumulative
        herfindahl += p**2

    gini = 1. - 2 * gini / len(col)
    return entropy, gini, herfindahl


def calculate_concentration(list_companies, N_boot):
    """ Take list of producting companies for movies in a time windown and calculates
        both Entropy, Gini coefficient, and Herfindahl-Hirschman index:

        E = Sum_{i=1 to N} si * ln(s_i)

        G = 1 - 1/mu * Integral_{0 to 1} [1 - F(s)]^2 ds

        HHI = Sum_{i=1 to N} s_i^2

        where s_i is market share of company i
        """

    if len(list_companies) == 0:
        return 0, 0, 0, 0

    entropy, gini, herfindahl= calculate_conc_indices(list_companies)

    # Calculate entropy and GC for bootstrapped data
    entropy_boot = []
    gini_boot = []
    herf_boot = []
    for run in range(N_boot):
        bootstrapped_list = []
        for i in range(len(list_companies)):
            bootstrapped_list.append( choice(list_companies) )

        ent, gc, herf = calculate_conc_indices(bootstrapped_list)
        entropy_boot.append(ent)
        gini_boot.append(gc)
        herf_boot.append(herf)

    return entropy, std(entropy_boot), gini, std(gini_boot), herfindahl, std(herf_boot)


def plot_timeline(axes, y_min, y_max, my_font_size):
    """
    Plots the timeline with relevant events in the 20th century.
    """
    # Add the notable events
    notable_events = {
        # line style and size
        ("dashed", 2.5): [
#            ["Eastman Kodak abandons MPPC", 1911],
            # Motion Picture Patent Company
            ["US v. Motion Picture Patent Corp.", 1915],
            ["Hays Code enforced", 1934],
            ["de Havilland v. Warner Bros.", 1944],
            ["US v. Paramount", 1948],
        ],
        ("solid", 4.): [
            ["19th Amendment", 1920],
            ["Equal Pay Act", 1963],
            ["Equal Rights\nAmendment (ERA)", 1972],
            ["ERA revoked", 1982],
        ]
    }

    for (style, size), details in notable_events.items():
        for event, year in details:
            axes.vlines(year, 0, 1.4, zorder=50, linestyles=style, lw=size)
            axes.annotate(
                event,
                xy=(year-2, 1.4),
                size=0.5*size*sns.mpl.rcParams["font.size"],
                rotation=45,
                verticalalignment="bottom"
            )

    # Add the wars
    wars = [
        [1914, 1918],   # WW I
        [1939, 1945],   # WW II
        [1950, 1953],   # Korean War
        [1955, 1975],   # Vietnam War
    ]

    for start, end in wars:
        rec = Rectangle(
            (start, 1.),
            end-start,
            1.2-1.0,
            color=sns.light_palette("red")[3],
            zorder=0
        )
        axes.add_patch(rec)


    # Add Feminism movements
    women = [
             [1914, 1920],   # Suffragette Movement
             [1966, 1976],   # 2nd Feminism wave
    ]

    for start, end in women:
        rec = Rectangle(
            (start, 0.7),
            end-start,
            0.9-0.7,
            color = '#fdcdac',
            zorder = 2
        )
        axes.add_patch(rec)


    # Add the TV adoption percentage
    tv_adoption = [
        # Start, End, Up to X*30% households with TV
        # Data taken from http://www.tvhistory.tv/Annual_TV_Households_50-78.JPG
        # which in turn is reproduced from: `TV Facts` by Cobbett Steinberg (1980)
        [1950, 1951, 1],
        [1951, 1954, 2],
        [1954, 1962, 3],
    ]

    for start, end, alpha in tv_adoption:
        rec = Rectangle(
            (start+0.2, 0.4),
            end-start-0.2,
            0.6-0.4,
            color=sns.light_palette("orange", 4)[alpha],
            zorder=2
        )
        axes.add_patch(rec)


    # Add the movie industry periods
    movie_industry = [
        [1908, 1915],     # MPPC years
        [1922, 1933],     # Studio System consolidation
        [1934, 1945]      # Golden Age Hollywood Studio System
    ]
    k = 1
    for start, end in movie_industry:
        rec = Rectangle(
            (start, 0.1),
            end-start,
            0.3-0.1,
            color = sns.light_palette('b', 15)[k],
            zorder=3
        )
        k += 1
        axes.add_patch(rec)

    bottom_frame(axes, '', font_size = my_font_size)
    axes.set_xlim(y_min, y_max)
    axes.set_xticks( arange(1910, 2011, 10) )


def plot_total_year_movies(axes, all_movies, y_min, y_max, y_scale_max, my_font_size, genre, ax_labels, PALETTE):
    """
    Plots the total number of movies per year over time.
    """
    half_frame(axes, '', '', font_size = my_font_size)
    year_range1 = (all_movies.year >= y_min) & (all_movies.year <= y_max)
    N_movies = len(all_movies[year_range1])
    print('Number of movies in IMDb database is {}'.format(N_movies))

    # Calculate number of movies per year and save to json file
    x_years = arange(y_min, y_max+1, 1)
    hist = [0] * int((y_max - y_min + 1))
    for year in all_movies[year_range1].year:
        i = year - y_min
        hist[i] += 1

    with open('Json_files/number_movies.json', 'w') as file_out:
        json.dump(hist, file_out)

    axes.bar(x_years, hist,
             width = 0.8,
             align = 'center',
             color = PALETTE[0],
            )

    axes.set_xlabel(ax_labels[0])
    axes.set_xlim(y_min-1, y_max+1)
    axes.set_xticks( arange(1910, 2011, 10) )

    axes.set_ylabel(ax_labels[1])
    # May have to adjust parameters for y_ticks
    axes.set_yticks( arange(0, y_scale_max+1, y_scale_max/4) )

    axes.hlines( arange(0, y_scale_max+1, y_scale_max/4), 1910., 2010., color='grey', lw=1, zorder=0 )

    sns.despine(ax=axes, trim=True)

    if genre:
        axes.text(1960, 0.7 * y_scale_max, '\n{} movies'.format(place_commas(N_movies)),
                  horizontalalignment = "center",
                  verticalalignment = "center",
                  size = 1.5 * my_font_size,
                  zorder = 19,
                  bbox={"facecolor": "white", "edgecolor": "white"},
                  )
        font0 = FontProperties()
        font = font0.copy()
        font.set_weight('bold')
        axes.text(1960, 0.9 * y_scale_max, genre.upper(),
                  fontproperties = font,
                  horizontalalignment = "center",
                  verticalalignment = "center",
                  size = 1.7 * my_font_size,
                  zorder = 20,
                  bbox={"facecolor": "white", "edgecolor": "white"},
                  )


def plot_industry_concentration(flag, axes, movies_df, y_min, y_max, my_font_size):
    """
    Plots the mean (+/- stderr) of industry concentration taken over a time window defined by delta
    """
    print('INDUSTRY CONCENTRATION')
    delta = 1
    time_slices = []
    for i in range(2*delta+1):
        time_slices.append([])
    gini_evol = []
    entropy_evol = []
    herfindahl_evol = []
    gini_sem = []
    entropy_sem = []
    herfindahl_sem = []


    for year in range(y_min, y_max):
        if year == y_min:
            k = delta
            for i in range(delta):
                slice = movies_df[movies_df.year == (year + i)]
                for j in range(len(slice)):
                    if type(slice.iloc[j]['production_companies_list']) == list:
                        #time_slices[k + i].extend(slice.iloc[j]['production_companies_list'])
                        time_slices[k + i].append(slice.iloc[j]['production_companies_list'][0])
        if year < y_max - delta:
            slice = movies_df[movies_df.year == (year + delta)]
            for j in range(len(slice)):
                if type(slice.iloc[j]['production_companies_list']) == list:
                    # time_slices[2*delta].extend(slice.iloc[j]['production_companies_list'])
                    time_slices[2 * delta].append(slice.iloc[j]['production_companies_list'][0])

        list_companies = []
        for i in range(2*delta+1):
            list_companies.extend(time_slices[i])

        if year == 1917 or year == 1922 or year == 1940:
            col = collections.Counter(list_companies)
            print(len(list_companies), len(col.most_common()), col.most_common(10))
            print('\n\n')

        entropy, sem_entropy, gini, sem_gini, herfindahl, sem_herfindahl= calculate_concentration(list_companies, 10)

        # for i in range(len(movies_slice)):
        #     if type(movies_slice.iloc[i]['production_companies_list']) == list:
        #         list_companies.extend(movies_slice.iloc[i]['production_companies_list'])
        #     elif movies_slice.iloc[i]['production_companies_list'] is not None:
        #         # print(movies_slice.iloc[i]['_id'], movies_slice.iloc[i]['production_companies_list'])
        #         pass


        gini_evol.append(gini)
        gini_sem.append(sem_gini)
        entropy_evol.append(entropy)
        entropy_sem.append(sem_entropy)
        herfindahl_evol.append(herfindahl)
        herfindahl_sem.append(sem_herfindahl)

        time_slices.pop(0)
        time_slices.append([])

    if flag == 'gini':
        evol = gini_evol
        sem = gini_sem
        y_label = 'Gini\ncoefficient'
        filename = 'Json_files/gini_coefficient.json'
    elif flag == 'herfindahl':
        evol = herfindahl_evol
        sem = herfindahl_sem
        y_label = 'Herfindahl\nindex'
        filename = 'Json_files/herfindahl_hirschman_index.json'

    evol_series = pd.Series(evol, index = list(range(y_min, y_max)))
    sem_series = pd.Series(sem, index = list(range(y_min, y_max)))

    # Save industry concentration to file
    #
    with open(filename, 'w') as file_out:
        json.dump(list(evol_series), file_out)

    line_color = 'red'
    band_color = sns.light_palette(line_color, 3)[1]
    evol_series.plot(ax = axes, label = '', color = line_color, zorder = 2, lw = 2)
    axes.fill_between( evol_series.index,
                       evol_series - 2.*sem_series,
                       evol_series + 2.*sem_series,
                       color = band_color, zorder = 1,
                       )

    # Highlight MPCC role
    axes.fill_between( evol_series.index[1911-y_min:1916-y_min],
                       0, 1,
                       color = sns.light_palette('b', 15)[0], zorder = 0,
                       )

    # Highlight consolidation of Studio System
    axes.fill_between( evol_series.index[1922-y_min:1934-y_min],
                       0, 1,
                       color = sns.light_palette('b', 15)[1], zorder = 0,
                       )
    # Highlight golden age of Studio System
    axes.fill_between( evol_series.index[1934-y_min:1946-y_min],
                       0, 1,
                       color = sns.light_palette('b', 15)[2], zorder = 0,
                       )

    half_frame(axes, 'Year',  y_label, font_size=my_font_size)

    axes.set_xlim(1910, 2010)
    axes.set_xticks( arange(1910, 2011, 10) )
    axes.set_ylim(0, 0.10)
    axes.set_yticks( arange(0, 0.085, 0.02) )
    axes.hlines( arange(0, 0.085, 0.02), 1910., 2010., color='grey', lw=1, zorder=0 )


    axes.hlines([0.25, 0.5, 0.75], 1910., 2010., color='grey', lw=1, zorder=0)

    return evol_series


def plot_gender_percent(axes, role, y_min, y_max, my_font_size, genre, color_id, PALETTE):
    """
    Plots the mean (+/- stderr) percent of female for movie role over time.
    """
    print('\n{} -- {}'.format(role.upper(), genre))
    role_key = role + "_gender_percentage"
    line_color = PALETTE[color_id]

    if genre == 'All':
        movie_df = get_movies_df(role_key)
    else:
        movie_df = get_genre_movies_df(genre, role_key)

    period = [d.year for d in pd.date_range(str(y_min - 10), str(y_max), freq="AS")]
    female_percentage_mean, female_percentage_sem = roll_mean_sem_movies(role_key, movie_df, period, 5)

    # Early industry mean percentage
    y_early = 1918
    y_actual = female_percentage_mean.loc[y_early:1950].idxmax()
    early_gender = female_percentage_mean.loc[y_actual]
    print("Early max of mean female percentage of {:.1f} was achieved in {}".format(early_gender, y_actual))

    # When the percent of females recovers back to the early mean
    # Between 1920 and 1930 the percentage of female actors jumps a lot so we
    # start looking after that period
    y_recover = (female_percentage_mean.loc[1940:] > early_gender).idxmax()
    print("Early maximum reached again in", y_recover)

    # Year of lowest female representation
    y_lowest = female_percentage_mean.loc[1930:].idxmin()
    f_lowest = female_percentage_mean.loc[y_lowest]
    print("Mean female percentage was lowest in {} at a level of {:.1f}".format(y_lowest, f_lowest))

    # Second wave feminism
    y_wave = 1966
    try:
        f_wave = female_percentage_mean.loc[y_wave]
        print("Mean female percentage in {} was {:.1f}".format(y_wave, f_wave))
    except KeyError:
        print('No data for {}'.format(y_wave))

    top_rate = max(female_percentage_mean.loc[y_min:y_max])
    print('Top female rate was: {:.2f}'.format(top_rate))

    # Relevant level lines
    axes.hlines(early_gender, y_early, y_max, color = 'r', linestyle="dashed", lw=2, zorder = 10)
    if f_lowest > 5.:
        axes.hlines(f_lowest, y_early, y_lowest, color = 'r', linestyle = "dashdot", lw=2, zorder = 10)
    # axes.hlines(f_wave, y_wave, y_max , color = 'r', lw=2, zorder = 10)

    # Overall female employment for the nation
    if role == 'acting_credited':
        with open('Json_files/employment_interpolated.json', 'r') as file_in:
            employment = json.load(file_in)

        normalized_employment = array(employment)
        axes.plot(range(1910, 2011), normalized_employment, color = 'olive', lw = 4, zorder = 3)
        top_rate = max(top_rate, max(normalized_employment))

    # Set maximum for graph
    if genre != 'All':
        y_label = '% Female'
        top_rate = 60
    else:
        y_label = ('% Female' + ' \n' + role).replace('_', ' ')
        if top_rate < 10:
            top_rate = max(1.1 * top_rate, 11)

        else:
            top_rate = max(1.1 * top_rate, 16)

    # If want to add parity line uncomment
    # axes.hlines(50, y_min, y_max, linestyles="dashed", color="gray", zorder=0, lw=5)
    # axes.text( 1960, 50, " Gender Parity ",
    #            horizontalalignment="center", verticalalignment="center", size=1.5 * my_font_size,
    #            bbox={"facecolor": "white", "edgecolor": "white"}
    #            )

    if role == 'producing':
        half_frame(axes, 'Year', y_label, font_size=my_font_size)
    else:
        half_frame(axes, '', y_label, font_size=my_font_size)
    band_color = sns.light_palette(line_color, 3)[1]

    # Plot data
    female_percentage_mean.plot(ax = axes, label = genre, color = line_color, zorder = 2, lw = 2)
    axes.fill_between( female_percentage_mean.index,
                       female_percentage_mean - 2.*female_percentage_sem,
                       female_percentage_mean + 2.*female_percentage_sem,
                       color = band_color, zorder = 1
                      )

    axes.set_ylim(0, int(top_rate))
    axes.set_yticks( arange(0, int(top_rate + 1), int(top_rate / 4)) )

    axes.set_xlim(1910, 2010)
    axes.set_xticks( arange(1910, 2011, 10) )

    if genre != 'All':
        font0 = FontProperties()
        font = font0.copy()
        font.set_weight('bold')
        axes.text(1960, 0.9 * top_rate, genre.upper(),
                  fontproperties=font,
                  horizontalalignment="center",
                  verticalalignment="center",
                  size=1.7 * my_font_size,
                  zorder=20,
                  bbox={"facecolor": "white", "edgecolor": "white"},
                  )

    return female_percentage_mean


def calc_decade_budget_pivot(movies_df, key, n_bins):
    """
    Creates a pivot table (for a heatmap) with the relative mean female
    fraction of actors for each (budget decile, decade) group of movies.
    Relative mean: We subtract the mean decade female actor fraction from each
    absolute fraction.
    """
    # We first group movies by decade
    movies_df["decade"] = movies_df.year.apply( get_decade )

    # Then we calculate the budget percentiles for each decade's worth of movies
    for _, group in movies_df.groupby("decade"):
        budget_quintiles = pd.qcut(group["adjusted_budget"], n_bins, labels = range(n_bins))
        movies_df.loc[group.index, "budget_quintile"] = budget_quintiles

    budget_pivot = movies_df.pivot_table( values= key +  "_gender_percentage",
                                          index = 'budget_quintile',
                                          columns = "decade"
                                          )
    budget_pivot.sort_index(ascending=False, inplace=True)

    # Remove data from 2010-2012. Not a complete decade, and from 1910s. Too little data.
    budget_pivot.drop(110, axis=1, inplace=True)
    budget_pivot.drop(10, axis=1, inplace=True)

    return budget_pivot.subtract(budget_pivot.mean(axis=0))


def plot_budget_heatmap(budget_pivot, axes, cbar_axes, key, y_labels, n_bins):
    """
    Plots the budget heatmap in `main_axes` and its colorbar in `cbar_axes`.
    """
    # Prevent small negative values from being displayed as '-0'
    #
    budget_pivot[(budget_pivot.abs() < 0.5)] = 0

    # Plot heatmap
    #
    sns.heatmap(budget_pivot,
                cmap="PuOr_r", center=0., linewidths=0.5, vmin=-10, vmax=10,
                ax=axes, cbar_ax=cbar_axes,
                annot=True, fmt=".0f", annot_kws={"size": 14})

    # Colobar legend and decade labelling
    #
    if key == 'acting_all':
        axes.set_ylabel("Decade", rotation=-90, labelpad=30, fontdict={'fontsize': 18})
        axes.yaxis.set_label_position("right")
        axes.text(n_bins + 3.3, len(y_labels) / 2, 'Decade-normalized female representation',
                  rotation=-90, verticalalignment="center", horizontalalignment="center",
                  fontsize=18)
    else:
        axes.set_ylabel("", fontdict={'fontsize': 14})

    axes.tick_params(length=0)
    axes.set_yticks([i + 0.3 for i in range(0, len(y_labels))])
    axes.set_yticklabels(y_labels, rotation=-90, fontdict={'fontsize': 14})
    axes.yaxis.tick_right()

    axes.set_xticklabels(list(range(1, n_bins + 1)), fontdict={'fontsize': 14})

    # Movie function labelling
    #
    axes.text(n_bins / 2, -0.5, key.replace('_', ' ').capitalize(),
              rotation=0, verticalalignment="center", horizontalalignment="center",
              fontsize=22)

    # Budget labels and explanation
    #
    if key == 'directing':
        axes.set_xlabel("\nBudget Quintile", fontdict={'fontsize': 18})
        axes.text(0.0, len(y_labels) + 0.4, "High",
                  rotation=0, verticalalignment="center", horizontalalignment="center",
                  fontsize=16)
        axes.text(n_bins, len(y_labels) + 0.4, "Low",
                  rotation=0, verticalalignment="center", horizontalalignment="center",
                  fontsize=16)
    else:
        axes.set_xlabel(" ", fontdict={'fontsize': 18})

    return


def plot_median_budgets(movies_df, axes, y_labels):
    """
    Adds a mini heatmap with the median budget for each decade.
    """

    # Median decade budget values
    decade_budget = pd.DataFrame(movies_df.groupby("decade").adjusted_budget.agg("median"))
    # Remove data from 2010-2012. Not a complete decade
    decade_budget.drop(110, axis=0, inplace=True)
    decade_budget.drop(10, axis=0, inplace=True)

    sns.heatmap(decade_budget / 10 ** 6, linewidths=0.5, vmax=50,
                ax=axes, cbar=False, cmap='copper_r',
                xticklabels=[""], yticklabels=y_labels,
                annot=True, fmt=".1f", annot_kws={"size": 14},
                )

    axes.tick_params(length=0)
    axes.set_yticks([i + 0.3 for i in range(0, len(y_labels))])
    axes.set_yticklabels(y_labels, rotation=-90, fontdict={'fontsize': 14})
    axes.yaxis.tick_right()

    axes.yaxis.set_label_position("left")
    axes.set_ylabel("Median budget (inflation-adjusted US$ M)", fontsize=22)

    axes.set_xlabel("")


def plot_count_budgets(movies_df, axes, y_labels):
    """
    Adds a mini heatmap with the count of movies with budget for each decade.
    """

    # Median decade budget values
    decade_count = pd.DataFrame(movies_df.groupby("decade")['cast_all'].count())
    # Remove data from 2010-2012. Not a complete decade
    decade_count.drop(110, axis=0, inplace=True)
    decade_count.drop(10, axis=0, inplace=True)

    sns.heatmap(decade_count, linewidths=0.5, vmax=2000,
                ax=axes, cbar=False, cmap='inferno_r',
                xticklabels=[""], yticklabels=[""],
                annot=True, fmt=".0f", annot_kws={"size": 14},
                )
    axes.set_ylabel("Number of movies", fontsize=22)
    axes.yaxis.set_label_position("left")
    axes.set_xlabel("")


def plot_risk_ratios(y_min, y_max, reduced, movie_roles, logistic_fits, time_dummies_flag, director_flag,
                     base_formula, formula_genre, time_dummies_formula, plot_dummies_flag, N_variables,
                     variable_list, figure, gs1, my_font_size, x_lim, e_value_delta):
    for cred_index, cred in enumerate(movie_roles):
        print('\n', cred.upper(), '----------------', y_min, y_max, '\n')

        # Finalize string for model definition
        #
        if cred in logistic_fits:
            model_string = cred + '_fraction ~' + base_formula + formula_genre
            if cred in logistic_fits:  # Remove Western from model because it affects fit convergence
                model_string = model_string[:-9]
        else:
            model_string = cred + '_gender_percentage ~' + base_formula + formula_genre

        if cred not in 'producing':
            model_string += ' + producing_gender_percentage '

        if cred not in ['producing', 'directing'] and director_flag:
            model_string += ' + directing_gender_percentage '

        if time_dummies_flag:
            model_string += time_dummies_formula

        print(f"\n{model_string}")

        # Fit the model
        #
        if cred in logistic_fits:
            string_evalue = 'logit'
            model = smf.logit(formula=model_string,
                              data=reduced, missing='drop').fit()
            summary_string = (f"$N = ${place_commas(int(model.nobs))}, "
                              f"pseudo-$R^2 = ${model.prsquared :.2f}")
        else:
            string_evalue = 'linear'
            model = smf.ols(formula=model_string,
                            data=reduced, missing='drop').fit()
            summary_string = (f"$N = ${place_commas(int(model.nobs))}, "
                              f"adj-$R^2 = ${model.rsquared_adj :.2f}")

        axes = figure.add_subplot(gs1[2 * (1 + cred_index)])
        axes.set_yticks(arange(0, N_variables))
        axes.set_ylim(-0.1, N_variables - 1.9)
        if cred_index == 0:
            half_frame(axes, '', ' ', font_size=my_font_size)
            axes.set_yticklabels(reversed(variable_list[1:N_variables]))

            font0 = FontProperties()
            font = font0.copy()
            font.set_weight('bold')
            axes.text(-2 * x_lim['acting_all'], N_variables, f'{y_min}-{y_max}',
                      fontproperties=font,
                      horizontalalignment="center",
                      verticalalignment="center",
                      size=1.5 * my_font_size,
                      zorder=20,
                      bbox={"facecolor": "white", "edgecolor": "white"},
                      )

        elif cred_index == int(len(movie_roles) / 2):
            bottom_frame(axes, 'log$_{10}$ Risk ratio', font_size=my_font_size)

        else:
            bottom_frame(axes, '', font_size=my_font_size)

        axes.vlines(0, -0.1, N_variables - 1.9, '0.3', lw=3)

        axes.set_xlim(-x_lim[cred], x_lim[cred])
        axes.set_xticks(linspace(-x_lim[cred], x_lim[cred], 5))

        font0 = FontProperties()
        font = font0.copy()
        font.set_weight('bold')
        axes.text(0, N_variables, cred.replace('_', ' ').upper(),
                  fontproperties=font,
                  horizontalalignment="center",
                  verticalalignment="center",
                  size=1.5 * my_font_size,
                  zorder=20,
                  bbox={"facecolor": "white", "edgecolor": "white"},
                  )

        axes.text(0, N_variables - 1, summary_string,
                  horizontalalignment="center",
                  verticalalignment="center",
                  size=my_font_size,
                  zorder=20,
                  bbox={"facecolor": "white", "edgecolor": "white"},
                  )

        skip = 0
        print(f"\t{'Variable':22} {'beta':8} {'signif'}     {'rr_low':11}"
              f" \t {'  rr':8} \t {'rr_high'} ")
        for j, item in enumerate(variable_list):
            if (((cred == 'directing' or not director_flag) and item == '% Female Directors')
                    or (cred == 'producing' and (item == '% Female Directors' or item == '% Female Producers'))
                    or (cred in logistic_fits and item == 'Western')):
                skip += 1
                print(f"---------{cred} \t {item} -- skip = {skip}  jk = {jk}")
                continue

            # Correct index for excluded variables in certain models
            jk = j - skip

            uncertainty = max(abs(model.params[jk] - model.conf_int()[0][jk]),
                              abs(model.conf_int()[1][jk] - model.params[jk]))
            rr, rr_low, rr_high = risk_ratio(string_evalue, model.params[jk],
                                             [model.conf_int()[0][jk],
                                              model.conf_int()[1][jk]],
                                             e_value_delta(item), model.params[0])
            print(f"{j :>2} - {jk :>2} - {variable_list[j][:15] :20} "
                  f"{significant_digits(model.params[jk], uncertainty)} \t "
                  f"{star(model.pvalues[jk])} \t {log10(rr_low) :7.3f} \t"
                  f"{log10(rr) :7.3f} \t {log10(rr_high) :7.3f}")

            if j <= N_variables or plot_dummies_flag:
                if model.pvalues[jk] > 0.01:
                    calc_color = '0.4'
                else:
                    if rr < 1:
                        calc_color = 'purple'
                    else:
                        calc_color = 'orange'
                axes.hlines(N_variables - j - 1, log10(rr_low), log10(rr_high),
                            color=calc_color, linewidth=2)
                axes.plot(log10(rr), N_variables - j - 1, 'o', color=calc_color,
                          markersize=6)

                if log10(rr) < -x_lim[cred] and log10(rr_high) < -x_lim[cred]:
                    axes.annotate("", xy = (-x_lim[cred]*0.99, N_variables - j - 1),
                                  xytext=(-x_lim[cred]*0.5, N_variables - j - 1),
                                  arrowprops = dict(color = calc_color, arrowstyle="->"))
                if log10(rr) > x_lim[cred] and log10(rr_low) > x_lim[cred]:
                    axes.annotate("", xy = (x_lim[cred]*0.99, N_variables - j - 1),
                                  xytext=(x_lim[cred]*0.5, N_variables - j - 1),
                                  arrowprops = dict(color = calc_color, arrowstyle="->"))


        print(f'\nModel AIC is {model.aic:.1f} and BIC is {model.bic:.1f}')

