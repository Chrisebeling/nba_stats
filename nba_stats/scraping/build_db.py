import sys
from bs4 import BeautifulSoup
import urllib
import re
import time
import calendar
import datetime as dt
import pandas as pd
import numpy as np
import mysql.connector as sql
import progressbar
import logging
from IPython.display import clear_output

from nba_stats.scraping.base_functions import get_soup, get_bref_soup, get_bref_tables
from nba_stats.scraping.functions import split_first_last, get_split, convert_feet, combine_columns, is_starter, to_int, convert_mp, include_comments, column_time
from nba_stats.read_write.db_insert import SqlDataframes
from nba_stats.read_write.functions import export_txt, create_schema_str

CURRENT_YEAR = dt.datetime.now().year
CURRENT_SEASON = CURRENT_YEAR + 1 if dt.datetime.now().month > 7 else CURRENT_YEAR
BREF_HTML = 'https://www.basketball-reference.com'
CRAWL_DELAY = 3
SEASON_TEAMS = {1977: 22,
                1981: 23,
                1989: 25,
                1990: 27,
                1996: 29,
                2005: 30}
PLAYOFF_TEAMS = {1954: 6,
                1967: 8,
                1975: 10,
                1977: 12,
                1984: 16}

stats_db = SqlDataframes(_host="nba-stats-inst.clmw4mwgj0eg.ap-southeast-2.rds.amazonaws.com", _password="23cHcGN9PNxxUKtAzGp28kJ7u")

logger_build = logging.getLogger(__name__)
# handler = logging.StreamHandler()
# file_handler = logging.FileHandler("logging\\%s.log" % dt.datetime.today().strftime('%Y%m%d'))
# formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-10s %(message)s')
# for a_handler in [handler, file_handler]:
#     a_handler.setFormatter(formatter)
# logger_build.addHandler(handler)
# logger_build.addHandler(file_handler)
# logger_build.setLevel(logging.INFO)

def get_players_urls(players_url=None):
    '''Returns soup objects of bref player pages (a-z)

    Keyword arguments:
    players_url - The url used to scrape the soups (default None)
    '''
    players_soups = []
    if players_url == None:
        players_url = BREF_HTML + '/players/'
    letters = [chr(n) for n in range(97,123)]
    
    success_count, http_error_count = 0, 0
    start_time = time.time()
    for letter in letters:
        players_soup = get_soup(players_url + letter)
        if players_soup != None:
            players_soups.append(players_soup)
            success_count += 1
        else:
            http_error_count += 1
    end_time = time.time()
    
    logger_build.info('Per run: ', (end_time - start_time)/(success_count+http_error_count), ', Successes: ', success_count, ', Failures: ', http_error_count)
    return players_soups

def get_all_players(players_soups):
    '''Takes soups of bref players and returns a df containing info of all players

    Keywork arguments:
    players_soups - A list of all the soups to be processed.
    '''
    players_dfs = []
    for p_soup in players_soups:
        players_df = get_bref_tables(p_soup, ['all_players'])
        players_dfs.append(players_df['all_players'])
    
    players = pd.concat(players_dfs)
    for i in [1,2]:
        players.loc[:,'pos' + str(i)] = players.pos.apply(lambda x: get_split(x, '-', i-1))
        players.loc[:,'college' + str(i)] = players.colleges.apply(lambda x: get_split(x, ',', i-1))
    players.birth_date = players.birth_date.apply(lambda x: '' if x == '' else dt.datetime.strptime(x, '%B %d, %Y').date().strftime('%Y-%m-%d'))
    for column in ['year_max', 'year_min', 'weight']:
        players[column] = players[column].apply(lambda x: to_int(x))
    players.year_max = players.year_max.apply(lambda x: np.nan if x == CURRENT_SEASON else x)
    for column, idx in zip(['first_name','last_name'],[0,1]):
        players.loc[:,column] = players.apply(lambda x: split_first_last(x['player'],x['bref'])[idx], axis=1)
    players.height = players.height.apply(lambda x: convert_feet(x))
        
    drop_columns = ['colleges', 'player', 'pos']
    players = players.drop(drop_columns, axis=1).reset_index(drop=True)
    
    return players

def get_colleges(players_table):
    '''Takes full players table and returns df of unique set of colleges

    Keyword arguments:
    players_table - The df containing players info
    '''
    colleges = pd.DataFrame(list(set(list(players_table.college1)+ list(players_table.college2))), columns=['college']).dropna()
    return colleges[colleges.college != ''].reset_index(drop=True)

def get_teams(url=None, headings=None):
    '''Returns a df containing the abbreviation and team name of all teams from bref page.

    Keywork arguments:
    url - the url to scrape, bref team page if none given (default None)
    headings - the headings to use when scraping, if none given uses default behaviour of get_soup (default None)
    '''
    if url == None:
        url = BREF_HTML + '/teams/'
    team_soup = get_soup(url, headings)
    
    tables= get_bref_tables(team_soup,['all_teams_active','all_teams_defunct'],'franch_name')
    
    for key in tables.keys():
        tables[key].loc[:,'team'] = tables[key].apply(lambda row: combine_columns(row['franch_name'], row['team_name']), axis=1)
    teams = pd.concat(tables).reset_index(drop=True)
    teams = teams.drop_duplicates('team').reset_index(drop=True)
    teams.loc[:,'abbreviation'] = teams.bref.apply(lambda x: re.findall('(?<=/teams/)[A-Z]{3}',x)[0] if type(x) == str else x)
    
    return teams[['abbreviation', 'team']]

def get_boxscore_htmls_month(year, month, headers=None, url_template=None):
    '''Returns a df containing info for all games in the given month.
    
    Keyword arguments:
    year -- the year the season ends in
    month -- the month as an integer
    headers -- override headers to use for the soup object (default None)
    url_template -- override template to use for url (default None)
    '''
    assert type(year) == int and type(month) == int, 'Year and month must be int'
    assert year <= CURRENT_YEAR + 1, 'Year must be before %s' % (CURRENT_YEAR + 1)
    assert month >= 1 and month <= 12, 'Month must be between 1 and 12'
    
    if url_template == None:
        url_template = "https://www.basketball-reference.com/leagues/NBA_%year%_games-%month%.html"
    month_url = url_template.replace('%year%',str(year)).replace('%month%', calendar.month_name[month].lower())
    soup = get_soup(month_url, headers)
    
    if soup:
        try:
            boxscores_month = get_bref_tables(soup,['all_schedule'],'box_score_text')['all_schedule']
        except KeyError as e:
            logger_build.info("Games table does not exist. Year: %s, month: %s." % (year, month))
            return None
        except:
            raise

        drop_columns = ['attendance', 'box_score_text', 'game_remarks', 'overtimes']
        boxscores_month.drop(drop_columns, inplace=True, axis=1)
        boxscores_month.rename(columns={'game_start_time':'start_time', 'home_team_name':'home_team', 'visitor_team_name':'visitor_team'}, inplace=True)
        boxscores_month.date_game = boxscores_month.date_game.apply(lambda x: dt.datetime.strptime(x, '%a, %b %d, %Y').date().strftime('%Y-%m-%d'))
        if 'start_time' in boxscores_month.columns:
            boxscores_month.start_time = boxscores_month.start_time.apply(lambda x: column_time(x))

        # keep only games that have been played
        boxscores_month = boxscores_month[boxscores_month.loc[:,'home_pts'] != '']
        
        for home_visitor in ['home','visitor']:
            boxscores_month[home_visitor+'_pts'] = boxscores_month[home_visitor+'_pts'].astype(int)

        return boxscores_month

def get_boxscore_htmls_year(year, regular_length=True, crawl_sleep=True, season_teams=SEASON_TEAMS, playoff_teams=PLAYOFF_TEAMS):
    '''Returns the html links for games of a given season.
    Season year is year season finishes. 
    
    Keyword arguments:
    year -- The year the desired season finishes. i.e. 2017/18, year=2018
    regular_length -- If False, will not apply assertions on game numbers (default True)
    '''
    season_boxscore_htmls = []
    months = list(range(8,13))+list(range(1,8))
    for month in months:
        if crawl_sleep:
            time.sleep(CRAWL_DELAY)
        try:
            month_games = get_boxscore_htmls_month(year, month)
        except:
            logger_build.info('Error on Season: {}, Month {}'.format(year, month))
        
        if isinstance(month_games, pd.DataFrame) and len(month_games) > 0:
            first_game_date = dt.datetime.strptime(re.findall('[0-9]{8}', month_games['bref'][0])[0],"%Y%m%d")
            assert first_game_date.month == month, 'Month error. First game: %s, Expected: %s' % (first_game_date.month, month)
            # for games in year before 'season year' actual year of game will be year before
            expected_year = year - 1 if month >= 8 else year
            assert first_game_date.year == expected_year, 'Year error. First game %s, Expected: %s' % (first_game_date.year, expected_year)
        
            season_boxscore_htmls.append(month_games)
    season_table_df = pd.concat(season_boxscore_htmls, sort=False).reset_index(drop=True)
    season_table_df.loc[:, 'season'] = year
    
    if regular_length:  
        no_playoff_teams = [y for x,y in playoff_teams.items() if x <= year][-1]

        actual_playoff_teams = sum(season_table_df.groupby('home_team').count().date_game > 41)
        assert actual_playoff_teams == no_playoff_teams, 'Nooooo. %s teams made playoffs. Should be 16. Year: %s' % (actual_playoff_teams, year)
        # min = 4 game playoff series, max = 7 game playoff series. Ideally want to compare against another source for exact number
        
        no_teams = [y for x,y in season_teams.items() if x <= year][-1]
           
        regular_games = no_teams * 82 / 2
        min_games = regular_games + 15 * 4
        max_games = regular_games + 15 * 7
        assert len(season_table_df) >= min_games and len(season_table_df) <= max_games, \
            'Impossible games number: %s, Min: %s, Max: %s. Year: %s' % \
            (len(season_table_df), min_games, max_games, year)

    return season_table_df

def get_season_gameno(games_df, regular_length=True):
    """Adds game numbers for each team.
    The relevant game number is which game the current game is in that team's season.
    
    Keyword arguments:
    season_boxscore_html -- the df of a seasons games
    """
    season_games_df = games_df.copy()
    teams = set(season_games_df.home_team_id)
    home_team_gameno = None
    visitor_team_gameno = None
    for team in teams:
        team_gameno = (np.cumsum(season_games_df.home_team_id==team)+np.cumsum(season_games_df.visitor_team_id==team))
        if isinstance(home_team_gameno, pd.Series):
            home_team_gameno += team_gameno * (season_games_df.home_team_id==team)
        else:
            home_team_gameno = team_gameno * (season_games_df.home_team_id==team)
        if isinstance(visitor_team_gameno, pd.Series):
            visitor_team_gameno += team_gameno * (season_games_df.visitor_team_id==team)
        else:
            visitor_team_gameno = team_gameno * (season_games_df.visitor_team_id==team)

    season_games_df.loc[:,'home_gameno'] = home_team_gameno
    season_games_df.loc[:,'visitor_gameno'] = visitor_team_gameno
    
    if regular_length:
        no_fullseasons = sum(season_games_df.home_gameno == 82)+sum(season_games_df.visitor_gameno == 82)
        no_playoffs = sum(season_games_df.home_gameno == 83)+sum(season_games_df.visitor_gameno == 83)
        assert no_fullseasons == 30, 'Expected 30 82nd games, got %s' % no_fullseasons
        assert no_playoffs == 16, 'Expected 16 playoff teams, got %s' % no_playoffs

    return season_games_df[['bref','game_id', 'home_gameno', 'visitor_gameno']]

def get_game_soups(games_table, check_tables=['boxscores', 'fourfactors'], limit=500, crawl_sleep=True, max_errors=3):
    """Returns a list containing the game id, bref and soup of all games in games table not already in check tables.
    Will only add game to the list if it has not been already added to each check table.
    
    Keyword arguments:
    games_table -- the df of games, includes game id and bref
    check_tables -- the tables to check if the game has already been added (default ['boxscores', 'linescores'])
    limit -- the max number of new entries in list (default 500)
    """
    current_ids = []
    start_time=time.time()

    for table in check_tables:
        game_ids = list((stats_db.read_table(get_str='SELECT DISTINCT game_id from {};'.format(table)).game_id))
        # game_ids = list((stats_db.read_table(table,['game_id'], distinct_only=True).game_id))

        if table == check_tables[0]:
            # for first iteration
            current_ids = game_ids
        else:
            current_ids = [game_id for game_id in current_ids if game_id in game_ids]
    
    new_games = games_table[~games_table.game_id.isin(current_ids)]
    if len(new_games) == 0:
        logger_build.info("No new games to add to database")
        return []
    id_bref = zip(new_games.game_id, new_games.bref)
    id_bref_soup = []
    count = 0

    logger_build.info('Finished prep: {:.1f} seconds since start'.format(time.time()-start_time))
    
    start_time = time.time()
    pbar = progressbar.ProgressBar(max_value=limit,
                                   widgets=[
                                    ' [', progressbar.Timer(), '] ',
                                    progressbar.Percentage(),
                                    progressbar.Bar(),
                                    ' (', progressbar.ETA(), ') ',
                                ])
    handshake_errors = 0
    pbar.start()
    for game_id, bref in id_bref:
        if count < limit:
            try:
                soup = get_soup(BREF_HTML + bref, timeout=10)
            except Exception as e:
                if handshake_errors < max_errors:
                    logger_build.error("Scraping soup error: ", e)
                    handshake_errors += 1
                    time.sleep(60)
                    continue
                else:
                    logger_build.error("Exceeded handshake error limit. Errors: %s, Max: %s" % (handshake_errors, max_errors))
                    raise

            id_bref_soup.append((game_id, bref, soup))
            count += 1
            if crawl_sleep:
                time.sleep(CRAWL_DELAY)
            pbar.update(count)
    pbar.finish()
    end_time = time.time()
    
    logger_build.info('Average run time excluding sleep: %s' % ((end_time - start_time-count*CRAWL_DELAY)/count))
    
    return id_bref_soup

def add_basic_gamestats(id_bref_soup, commit_changes=True):
    """Adds boxscores and linecores to db, taking list of game_ids, brefs and boxscore soups as input.
    Applies mappings to convert players, teams, etc to relevant id
    
    Keyword arguments:
    id_bref_soup -- A list of tuples, contains game id, bref, boxscore soup"""
    boxscores, adv_boxscores, linescores, fourfactors = [], [], [], []
    
    length = len(id_bref_soup)
    pbar = progressbar.ProgressBar(max_value=length,
                                   widgets=[
                                    ' [', progressbar.Timer(), '] ',
                                    progressbar.Percentage(),
                                    progressbar.Bar(),
                                    ' (', progressbar.ETA(), ') ',
                                ])
    pbar.start()
    start_time = time.time()
    count=0
    
    for game_id, bref, soup in id_bref_soup:

        if soup == None:
            continue
        boxscore = get_boxscore(soup)
        adv_boxscore = get_boxscore(soup, True)
        try:
            linescore = get_linescore(soup)
            fourfactor = get_linescore(soup, table_type='four_factors')
        except Exception as e:
            logger_build.error(game_id, bref)
            raise
        
        for df in [boxscore, adv_boxscore]:
            if df.empty:
                year_regex = re.findall("(?<=boxscores.)[0-9]{4}", bref)
                if year_regex:
                    if int(year_regex[0]) > 1982:
                        logger_build.error("Missing a boxscore. game_id: %s, bref: %s" % (game_id, bref))
                else:
                    logger_build.error("Missing a boxscore. Bref format strange. game_id: %s, bref: %s" % (game_id, bref))
                # df.loc[0,'game_id'] = game_id
            # else:
            #     df.loc[:,'game_id'] = game_id
        if linescore.empty or fourfactor.empty:
            logger_build.error("Missing a linescore. Linescore empty: %s, FourFactors empty: %s. game_id: %s, bref: %s" % 
                (linescore.empty, fourfactor.empty, game_id, bref))
            # df.loc[0,'game_id'] = game_id
        # else:
        #     df.loc[:,'game_id'] = game_id
        for df in [boxscore, adv_boxscore, linescore, fourfactor]:
            if len(df.index) > 0:
                df.loc[:,'game_id'] = game_id
            else:
                df.loc[0,'game_id'] = game_id
        
        boxscores.append(boxscore)
        adv_boxscores.append(adv_boxscore)
        linescores.append(linescore)
        fourfactors.append(fourfactor)
        
        count+=1
        pbar.update(count)
        
    end_time = time.time()

    boxscores_df = pd.concat(boxscores, sort=False).reset_index(drop=True)
    adv_boxscores_df = pd.concat(adv_boxscores, sort=False).reset_index(drop=True)
    linescores_df = pd.concat(linescores, sort=False).reset_index(drop=True)
    fourfactors_df = pd.concat(fourfactors, sort=False).reset_index(drop=True)
    
    pbar.finish()

    boxscores_df = stats_db.apply_mappings(boxscores_df, 'starters', ['starter'])
    boxscores_df = stats_db.apply_mappings(boxscores_df, 'players', ['player'], 'bref')
    boxscores_df = stats_db.apply_mappings(boxscores_df, 'teams', ['team'])
    if 'player' in adv_boxscores_df.columns:
        adv_boxscores_df = stats_db.apply_mappings(adv_boxscores_df, 'players', ['player'], 'bref')
    linescores_df = stats_db.apply_mappings(linescores_df, 'teams', ['team'])
    fourfactors_df = stats_db.apply_mappings(fourfactors_df, 'teams', ['team'])

    if commit_changes:
        stats_db.add_to_db(boxscores_df, 'boxscores', 'game_id')
        stats_db.add_to_db(adv_boxscores_df, 'adv_boxscores', 'game_id')
        stats_db.add_to_db(linescores_df, 'linescores', 'game_id')
        stats_db.add_to_db(fourfactors_df, 'fourfactors', 'game_id')
    
    logger_build.info('Average run time of soup extraction: %s' % ((end_time - start_time)/length))

def get_boxscore(boxscore_soup, advanced=False):
    '''Returns a df containing boxscore data for both teams, given the soup of the boxscore url.
    pct fields are removed as these can be inferred from data.
    Advanced box score option is in development stage. Will return df but formatting not refined.
    
    Keyword arguments:
    boxscore_soup -- A soup object of the boxscore url
    advanced -- If True, returns the advanced box score (Default False)
    '''
    # start_time = time.time()
    table_dict = {}
    re_match = 'all_box-[A-Z]{3}-game-advanced' if advanced else 'all_box-[A-Z]{3}-game-basic'
    re_compile = re.compile(re_match)
    find_team_regex = '(?<=all_box_)[a-z]{3}(?=_advanced)' if advanced else '(?<=all_box_)[a-z]{3}(?=_basic)'
    
    tables = get_bref_tables(boxscore_soup, [re_compile])
    teams = get_away_home_teams(boxscore_soup)
    
    for key in tables.keys():
        if 'reason' in tables[key].keys():
            tables[key].loc[:,'starter'] = tables[key].apply(lambda row: is_starter(row.name, row.reason), axis=1)
        else:
            tables[key].loc[:,'starter'] = tables[key].apply(lambda row: is_starter(row.name), axis=1)
    #     team_abb = re.findall(find_team_regex, key)[0].upper()
    #     tables[key].loc[:,'team'] = team_abb
        tables[key].loc[:,'team'] = teams[0]
        teams.pop(0)
    
    try:
        boxscore = pd.concat([tables[key] for key in tables.keys()], sort=False).reset_index(drop=True)
    except ValueError as e:
        return pd.DataFrame()
    except:
        raise
    boxscore = boxscore[boxscore.player != 'Reserves'] 
    
    if advanced:
        column_drops = ['reason', 'player', 'efg_pct', 'ts_pct', 'fg3a_per_fga_pct', 'fta_per_fga_pct', 'starter', 'team', 'mp']
    else:
        column_drops = ['reason', 'player'] + [header for header in boxscore.keys() if 'pct' in header]
        boxscore['mp'] = boxscore['mp'].apply(lambda x: convert_mp(x))
        
    column_drops = [x for x in column_drops if x in boxscore.keys()]
    non_number = ['mp', 'player', 'starter', 'team']
    boxscore.drop(column_drops, axis=1, inplace=True)
    boxscore.rename(columns={'bref':'player'}, inplace=True)
    for column in boxscore.columns:
        if column not in non_number:
            boxscore[column] = boxscore[column].apply(lambda x: to_int(x, 'pct' in column))
            
    # end_time = time.time()
    # export_txt(str(end_time - start_time) + '\n', 'boxscore_times_%label%.csv'.replace('%label%', test_csv_name))
    
    return boxscore

def get_linescore(boxscore_soup, table_type='line_score'):
    '''Returns a df of the basic linescore from the boxscore soup.
    Will contain any number of OTs and the total score.
    
    Keyword arguments:
    boxscore_soup -- the soup object of the boxscore url
    '''
    teams = pd.Series(get_away_home_teams(boxscore_soup))
    table = boxscore_soup.find('div',{'id':'all_' + table_type})
    if not table:
        return pd.DataFrame()
    full_table = include_comments(table)
    rows = full_table.find_all('tr')
    table_rows = []
    
    if table_type == 'line_score':
        possible_headers = [str(x) for x in [1,2,3,4,'T']]
    else:
        possible_headers = [str(x) for x in ['Pace', 'eFG%', 'TOV%', 'ORB%', 'FT/FGA', 'ORtg']]
        
    for row in rows:
        if not row.find_all('td'):
            if 'class' in row.attrs:
                raw_header = [x.text for x in row.find_all()]
                header = ['team'] + [x for x in raw_header if x in possible_headers or 'OT' in x]
        else:
            table_rows.append([x.text for x in row.find_all(['td','th'])])
    
    if table_type == 'line_score':
        header = ['Q' + column if re.match("^[1-4]$", column) else column for column in header]
    else:
        header = ['ft_rate' if column == 'FT/FGA' else column.lower().replace('%','') for column in header]
    boxscore_df = pd.DataFrame(table_rows, columns = header)
    boxscore_df.team = teams

    return boxscore_df.set_index('team').apply(pd.to_numeric).reset_index()

def get_series_games(soup):
    '''Returns a df containing each game of the series given the bref soup of the series.
    Returns game no, game_id, series_id

    Keyword Arguments:
    soup -- the soup of the bref series url
    '''
    series_games = {}

    for div in soup.find_all('div',{'class':'game_summary expanded nohover'}):
        for row in div.find_all('tr',{'class':'date'}):
            game_str = row.text.split(',')[0]
            if re.match('^Game [1-9]$',game_str):
                game_no = int(game_str[5])
            else:
                raise('Game no str not readable: {}'.format(row))

        for td in div.find_all('td',{'class':'right gamelink'}):
            bref = td.find('a')['href']

        series_games[game_no] = bref
        
    return pd.DataFrame({'bref':series_games}).reset_index().rename(columns={'index':'game_no'})

def get_playoff_games(season_range):
    '''Returns a df containing details of all playoff games in the desired seasons.
    Converts bref and series name to the relevant ids.
    Columns returned: game_no, game_id, series_id.
    
    Keyword Arguments:
    season_range -- A tuple, only seasons in this range will be returned.
    '''
    assert type(season_range) == tuple, 'season_range must be a tuple'
    assert len(season_range) == 2, 'season_range must contain 2 elements'
    assert season_range[0] <= season_range[1], 'first season must be before second season in range'
    
    playoffs_soup = get_bref_soup('/playoffs/series.html')
    playoffs_table = get_bref_tables(playoffs_soup, ['div_playoffs_series'], 'series')['div_playoffs_series'].loc[:,['bref','series','season']].dropna()
    playoffs_table.loc[:,'season'] = playoffs_table.loc[:,'season'].astype(int)
    
    count = 0
    all_series = []
    
    playoffs_table_restricted = playoffs_table[(playoffs_table.loc[:,'season'] >= season_range[0]) & (playoffs_table.loc[:,'season'] <= season_range[1])]
    
    pbar = progressbar.ProgressBar(max_value=len(playoffs_table_restricted),
                                   widgets=[
                                    ' [', progressbar.Timer(), '] ',
                                    progressbar.Percentage(),
                                    progressbar.Bar(),
                                    ' (', progressbar.ETA(), ') ',
                                ])
    pbar.start()
    for idx, row in playoffs_table_restricted.iterrows():
        series_soup = get_bref_soup(row[0])
        series_games = get_series_games(series_soup)

        series_games.loc[:,'series_name'] = row[1]

        all_series.append(series_games)
        count+=1
        pbar.update(count)
    
    pbar.finish()
    
    series_df = pd.concat(all_series)

    series_ids = stats_db.apply_mappings(series_df, 'games', ['bref'], 'bref').rename(columns={'bref_id':'game_id'})
    series_ids = stats_db.apply_mappings(series_ids, 'series', ['series_name'], 'series_name').rename(columns={'series_name_id':'series_id'})

    scraped_ids = list(series_ids.loc[:,'game_id'])
    table_ids = list((stats_db.read_table('games',['game_id'], distinct_only=True).game_id))
    to_add = [game_id for game_id in scraped_ids if game_id in table_ids]
    series_ids = series_ids[series_ids.loc[:,'game_id'].isin(to_add)]
    
    return series_ids

def get_away_home_teams(soup):
    teams = []
    for item in soup.find_all('div',{'class':'scorebox'}):
        for link in item.find_all('a',{'itemprop':'name'}):
            teams.append(link.text)
    
    return teams                