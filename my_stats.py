__author__ = "Amaral LAN and Moreira JAG"
__copyright__ = "Copyright 2018, 2019, Amaral LAN and Moreira JAG"
__credits__ = ["Amaral LAN", "Moreira JAG"]
__version__ = "1.0"
__maintainer__ = "Amaral LAN"
__email__ = "amaral@northwestern.edu"
__status__ = "Production"


import pandas as pd

from itertools import islice
from numpy import exp, sqrt, log10, infty, nan, float32
from os import path, getcwd



def window(seq, n=3):
    """
    Returns a sliding window (of width n) over data from the iterable:
       s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...
    From: https://docs.python.org/3.6/library/itertools.html#itertools-recipes
    """
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def sem(data):
    """
    Calculates the standard error of the mean for the given dataset.
    """
    return pd.np.std(data)/pd.np.sqrt(len(data))


def find_largest_consec_region(s_bool, threshold=2):
    """
    Finds the start and end indices of largest consecutive region
    of values from the given boolean Series that are True. Ignores any short
    peaks of `threshold` (default 2) or less values.

    Inspired by: http://stackoverflow.com/a/24433632
    """
    indices = s_bool.index

    regions = pd.DataFrame()

    # First row of consecutive region is a True preceded by a False in tags
    regions["start"] = indices[s_bool & ~s_bool.shift(1).fillna(False)]

    # Last row of consecutive region is a False preceded by a True
    regions["end"] = indices[s_bool & ~s_bool.shift(-1).fillna(False)]

    # How long is each region
    regions["span"] = regions.end - regions.start + 1

    # index of the region with the longest span
    max_idx = regions.span.argmax()
    start_max = regions.start.iloc[max_idx]
    end_max = regions.end.iloc[max_idx]

    # How many years between gaps
    regions["gap"] = regions.start - regions.end.shift(1)

    # Are there any non-spurious gaps separated by `threshold` values or less
    # right after the largest gap?
    small_gaps = (regions.end > end_max)
    small_gaps &= (regions.gap <= threshold)
    small_gaps &= (regions.span > 1)

    # If so, the largest such gap is now the region's end
    if small_gaps.sum() > 0:
        end_max = regions.end[small_gaps].iloc[-1]

    return (start_max, end_max)


def rolling_mean_sem(key, df, date_range, w_width):
    """
    Returns rolling average of counts for each year

    :param key:
    :param df:
    :param w_width:
    :return:
    """
    series_mean = {}
    df.sort_values(by="year", inplace=True)

    for w in window(date_range, w_width):
        rolling_df = df[(df.year >= w[0]) & (df.year <= w[-1])]
        mean = rolling_df[key].count()/ w_width

        if pd.isnull(mean):
            continue
        series_mean[w[-1]] = mean

    return pd.Series(series_mean)


def star(pvalue, thresholds = [0.0001, 0.001, 0.01]):
    """
    Returns number of stars in order to indicate statistical significance

    :param
        pvalue: float
    :return:
        string with 0-3 stars
    """
    if pvalue < thresholds[0]:
        return '***'
    elif pvalue < thresholds[1]:
        return '**'
    elif pvalue < thresholds[2]:
        return '*'

    return ''




def e_value(risk_ratio, p_value):
    """
    Calculates e-value using method described in 'Introducing the E-value' by VanderWeele and Ding, Ann Intern Medicine

    :param risk_ratio: float
    :param p_value: float

    :return: string for printing in latex table
    """

    if p_value > 0.01:
        return ''

    if risk_ratio < 1.:
        risk_ratio = 1/ risk_ratio

    e_value = risk_ratio + sqrt(risk_ratio * (risk_ratio - 1.0))

    if e_value > 1.25:
        return f' \\\\ E-value = {e_value :.2f}'
    else:
        return ''


def significant_digits(estimate, uncertainty):
    """
    Returns strings for estimate with appropriate number of significant digits (revised)

    :param:
        estimate: float
        uncertainty: float
    :return:
        str_estimator: string
    """
    # Get sign and order of magnitude of estimate
    #
    try:
        sign_estimate = estimate / abs(estimate)
        estimate = abs(estimate)
        uncertainty = abs(uncertainty)

        order_magnitude = log10(estimate)
        if order_magnitude < 0:
            order_magnitude = int(order_magnitude) - 1
        else:
            order_magnitude = int(order_magnitude)

        # Split pre-period and post-period digits and add padding zeros in case estimate does not have enough digits
        #
        #   The issue her is that scaled_number[1] can become really long because of digital conversion.
        #   For example, 37 becomes 36.99999999999997
        #
        scaled_number = str(estimate / 10 ** order_magnitude).split('.')
        n_post_period = len(scaled_number[1])

        n_digits = int( log10( estimate / uncertainty ) ) + 1

        if n_post_period < 14:
            n_padding = 14 - n_post_period
            padding_zeros = '0' * n_padding
        else:
            n_padding = 0
            padding_zeros = ''

        digits_post_period = scaled_number[1][:n_post_period+1] + padding_zeros
        digits_pre_period = scaled_number[0]
        sig_digits = float(digits_pre_period) + float(digits_post_period) / 10**(n_post_period + n_padding)

        # print('\n', scaled_number, '--', padding_zeros, '--', n_padding)
        # print(estimate, uncertainty, estimate/uncertainty)
        # print('Estimated # of significant digits is {}'.format(n_digits))
        # print(sig_digits)
        # print('Order of magnitude is {}'.format(order_magnitude))
        # print('Digits post dot are {}'.format(n_post_period))

        # Recreate estimate with appropriate number of significant digits and generate format string
        #
        clean_estimate = sig_digits * 10 ** order_magnitude
        precision = max( 0, n_digits - order_magnitude )
        n_format = '.' + str(precision) + 'f'
        # print(n_format, clean_estimate)

        # return '{0:{1}}'.format(sign_estimate * clean_estimate, n_format)
        return f'{sign_estimate * clean_estimate : {n_format}}'
    except ValueError:
        return nan


def place_commas(n):
    """Takes integer and returns string from printing with commas separating factors of 1000
    """
    tmp = str(n)
    n_digits = len(tmp)

    line = ''
    for i in range(n_digits):
        if not (i) % 3 and i != 0:
            line = tmp[-i - 1] + ',' + line
        else:
            line = tmp[-i - 1] + line

    return line


def to_tex_scientific(numb, sig_digits=2):
    """
    Convert a number to classical scientific notation:
    2.5e+6 -> 2.5 x 10^6

    Outputs number as a string of math tex code (e.g., 2.5 \times 10^{6}), meant
    to be used inside a pre-defined math environment.

    numb: Number to convert
    sig_digits: Number of significant digits to use in the mantissa.
    """
    # If the number is too small python does not convert it to scientific
    # notation
    if abs(numb) <= 1e5:
        return str(numb)
    fmt = "{{:.{}g}}".format(sig_digits)

    numb_str = fmt.format(numb)
    mantissa, exponent = numb_str.split("e")

    return f"{mantissa} \times 10^{{{int(exponent)}}}"


def half_frame(sub, xaxis_label, yaxis_label, font_size = 15, padding = -0.02):
    """Formats frame, axes, and ticks for matplotlib made graphic with half frame."""

    # Format graph frame and tick marks
    sub.yaxis.set_ticks_position('left')
    sub.xaxis.set_ticks_position('bottom')
    sub.tick_params(axis = 'both', which = 'major', length = 7, width = 2, direction = 'out', pad = 10,
                    labelsize = font_size)
    sub.tick_params(axis = 'both', which = 'minor', length = 5, width = 2, direction = 'out', labelsize = 10)
    for axis in ['bottom','left']:
        sub.spines[axis].set_linewidth(2)
        sub.spines[axis].set_position(("axes", padding))
    for axis in ['top','right']:
        sub.spines[axis].set_visible(False)

    # Format axes
    sub.set_xlabel(xaxis_label, fontsize = 1.6 * font_size)
    sub.set_ylabel(yaxis_label, fontsize = 1.6 * font_size)


def bottom_frame(sub, xaxis_label, font_size = 15, padding = -0.02):
    """Formats frame, axes, and ticks for matplotlib made graphic with half frame."""

    # Format graph frame and tick marks
    sub.yaxis.set_ticks_position('none')
    sub.yaxis.set_ticklabels([])
    sub.xaxis.set_ticks_position('bottom')
    sub.tick_params(axis = 'x', which = 'major', length = 7, width = 2, direction = 'out', pad = 10,
                    labelsize = font_size)
    sub.tick_params(axis = 'x', which = 'minor', length = 5, width = 2, direction = 'out', labelsize = 10)
    for axis in ['bottom']:
        sub.spines[axis].set_linewidth(2)
        sub.spines[axis].set_position(("axes", padding))
    for axis in ['top','right', 'left']:
        sub.spines[axis].set_visible(False)

    # Format axes
    sub.set_xlabel(xaxis_label, fontsize = 1.6 * font_size)


def left_frame(axes, xaxis_label, yaxis_label, font_size = 15, padding = -0.02):
    """
    Formats frame, axes, and ticks for matplotlib made graphic with half frame.
    """
    # Format graph frame and tick marks
    axes.yaxis.set_ticks_position('left')
    axes.xaxis.set_ticks_position('none')
    axes.xaxis.set_ticklabels([])
    axes.tick_params(axis = 'y', which = 'major', length = 7, width = 2, direction = 'out', pad = 10,
                    labelsize = font_size)
    axes.tick_params(axis = 'y', which = 'minor', length = 0, width = 0, direction = 'out', labelsize = 0)
    for axis in ['left']:
        axes.spines[axis].set_linewidth(2)
        axes.spines[axis].set_position(("axes", padding))
    for axis in ['bottom','top','right']:
        axes.spines[axis].set_visible(False)

    # Format axes
    axes.set_xlabel(xaxis_label, fontsize = 1.6 * font_size)
    axes.set_ylabel(yaxis_label, fontsize = 1.6 * font_size)


def save_fig(figure, figure_name, extension = 'pdf'):
    """
    Saves a figure with the proper name and properties in the appropriate
    directory.
    """
    figure_path = path.abspath( path.join( getcwd(), 'Figures', figure_name + '.' + extension ) )

    figure.savefig( figure_path, dpi = 300, bbox_inches = "tight" )


def risk_ratio(fit_type, coefficient, confidence_interval, relevant_x, p0):

    if fit_type == 'logit':
        rr = exp(coefficient * relevant_x)
        rr_low = exp(confidence_interval[0] * relevant_x)
        rr_high = exp(confidence_interval[1] * relevant_x)

    elif fit_type == 'linear':
        if p0 > 0:
            if p0 + coefficient * relevant_x > 0:
                rr = (p0 + coefficient * relevant_x) / p0
            else:
                rr = 10**(-5)

            if p0 + confidence_interval[0] * relevant_x > 0:
                rr_low = (p0 + confidence_interval[0] * relevant_x) / p0
            else:
                rr_low = 10**(-5)

            if p0 + confidence_interval[1] * relevant_x > 0:
                rr_high = (p0 + confidence_interval[1] * relevant_x) / p0
            else:
                rr_high = 10**(5)

        else:
            rr = nan
            rr_low = nan
            rr_high = nan

    return rr, rr_low, rr_high










