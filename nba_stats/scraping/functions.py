import numpy as np
import re
import datetime as dt
from bs4 import BeautifulSoup

def is_starter(index, reason=None):
    """Takes index and reason column, returns either S, R or DNP.
    Used on boxscore df. Reason is either Nan or 'Did Not Play'.

    Keyword arguments:
    index -- Used to define if starter or not
    reason -- Used to define if DNP or not
    """
    # Any reason converted to DNP
    if type(reason) == str:
        return 'DNP'
    elif index <= 4:
        return 'S'
    else:
        return 'R'
    
def to_int(x, to_float=False):
    """Wrapper for numpy isnan to deal with non-float types.
    If nan, returns unaltered values.
    """
    if x == '' or (type(x) == float and np.isnan(x)):
        return ''
    else:
        if to_float:
            return float(x)
        else:
            return int(x)
    
def convert_mp(mp):
    """Converts minutes played to a float."""
    if type(mp) == str:
        if ':' in mp:
            split_mp = mp.split(':')
            try:
                 return float(split_mp[0]) + float(split_mp[1])/60
            except Exception as e:
                raise Exception("%s. split_mp: %s" % (e, split_mp))
        else:
            return np.nan
    else:
        return mp

def convert_feet(feet_str, delim='-'):
    """Convert feet_inches measurement to cms measurement."""
    if type(feet_str) == str and delim in feet_str:
        feet_inches = [int(x) for x in feet_str.split(delim)]
        cms = 2.54 * (int(feet_inches[0]) * 12 + int(feet_inches[1]))
        return cms
    else:
        return np.nan
    
def split_first_last(name, ref):
    """Converts name into first name, last name.
    Uses the bref username to define which name is the last name when more than 2 names given.

    Keyword arguments:
    name -- The full name of a player. Can be any length of words
    ref -- The bref username. The first two characters are used to define which name is last name
    """
    try:
        split_name = name.replace('*','').split(' ')
    except:
        print(name)
        return np.nan, np.nan
    if len(split_name) == 2:
        return split_name[0], split_name[1]
    elif len(split_name) == 1:
        return np.nan, split_name[0]
    else:
        check_str = ref[:2].lower()
        matches = [n for n in split_name if check_str == n[:2].lower()]
        if len(matches) == 1:
            idx = split_name.index(matches[0])
            return ' '.join(split_name[:idx]), ' '.join(split_name[idx:])
        else:
            return np.nan, name

def include_comments(soup):
    '''Removes any comment characters in the soup object so the commented text will not be ignored.
    
    Keyword arguments:
    soup -- the soup to be altered
    '''
    string = soup.decode_contents().replace('<!--','').replace('-->','')
    return BeautifulSoup(string, 'html.parser')

def get_split(string, delim, pos):
    '''Splits the string and then gets the element at pos.
    Returns None if not enough elements.
    '''
    split_str = string.split(delim)
    if pos + 1 > len(split_str):
        return np.nan
    else:
        return split_str[pos].strip()

def combine_columns(col_a, col_b):
    '''Combines two columns. Returns col_b if col_a is nan.
    If col_b is nan, returns col_a by default.
    '''
    if type(col_a) == str:
        return col_a
    elif type(col_a) == float and np.isnan(col_a):
        if type(col_b) == str:
            return col_b
        else:
            return col_a
    else:
        return col_a

def column_time(time_str):
    '''Apply function to convert to time object'''
    if time_str == '':
        return time_str
    else:
        time_str = time_str if re.match('.*[mM]', time_str) else time_str +'M'
        return dt.datetime.strptime(time_str.replace(' ',''), '%I:%M%p').time().strftime('%H:%M:%S')