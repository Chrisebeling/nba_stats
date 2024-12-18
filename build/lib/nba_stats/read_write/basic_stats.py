import pandas as pd
import numpy as np
import datetime as dt
import re
import time
import matplotlib.pyplot as plt
import seaborn as sns
import mysql.connector as sql

from nba_stats.read_write.config import get_dbconfig

cfg = get_dbconfig()
HOST = cfg['host']
PORT = int(cfg['port'])
USER = cfg['user']
PASSWORD = cfg['password']
DB = cfg['db']

def isfloat(number):
    try:
        float(number)
        return True
    except (ValueError, TypeError):
        return False

def add_where(original, add):
    if original == '':
        return 'WHERE ' + add
    else:
        return original + ' AND ' + add

class ReadDatabase(object):
    def __init__(self, _password=PASSWORD, _user=USER, _db=DB, _host=HOST, _port=PORT,_unix=None):
        self.establish_connection()
        self.summary = {}
        self.summary_cats = {}
        self.current_summary = 'default'

    def establish_connection(self, _password=PASSWORD, _user=USER, _db=DB, _host=HOST, _port=PORT,_unix=None):
        if _unix:
            self.conn = sql.connect(
                unix_socket=_unix,
                user=_user,
                password=_password,
                database=_db)
        elif _port and _host:
            self.conn = sql.connect(
                host=_host,
                port=_port,
                user=_user,
                password=_password,
                database=_db)
        else:
            raise("unix socket or port/host must be provided.")
        self.conn.autocommit = True

    def establish_cursor(self):
        try:
            cursor = self.conn.cursor()
        except sql.errors.OperationalError:
            self.establish_connection()
            cursor = self.conn.cursor()

        return cursor

    def read_table(self, table_name=None, columns=None, get_str=None, write=False):
        '''Returns the given table, all columns if none given.
        Pass a get_str to customise request. In this case, table name is not required.
        Columns where the first entry is a float will be converted to floats.

        Keyword Arguments:
        table_name - The name of the table in the database to read (default None)
		columns - The columns to read from the table (default None)
		get_str - A sql query, will execute on the current db (default None)
		write - If set, will not return any values and commit the query made. (default False)
        '''
        assert type(columns) == list or columns == None, 'Columns must be given in list, not %s' % type(columns)

        if get_str:
            sql_string = get_str
        else:
            columns_str = "*" if columns == None else ', '.join(columns)
            sql_string = "SELECT %s FROM %s;" % (columns_str, table_name)

        cursor = self.establish_cursor()
        try:
            cursor.execute(sql_string)
        except sql.errors.ProgrammingError:
            print(sql_string)
            raise

        if write:
            pass
            self.conn.commit()
        else:
            table_columns = cursor.column_names
            table_lines = cursor.fetchall()

            df = pd.DataFrame(table_lines, columns = table_columns)

            if not df.empty:
                float_columns = [header for header, item in zip(table_columns, table_lines[0]) if isfloat(item)]

                # apply on each column individually so only failed do not convert
                for float_column in float_columns:
                    try:
                        df[[float_column]] = df[[float_column]].apply(pd.to_numeric)
                    except Exception as e:
                        print('Could not convert float columns to numeric: {}'.format(e))

            return df

    def set_summary(self, summary_name, reset_cats=False):
        '''Set the current summary which is to be worked on. The relevant cat_summaries and summary df will now be used.

        Keyword Arguments:
        summary_name - The name the summary will be assigned to in each dict. Must be a string.
        '''
        assert type(summary_name) == str, 'summary_name must be str, {} given'.format(type(summary_name))
        # alter current summary to given summary_name
        self.current_summary = summary_name
        # if summary name is new, create dict entry in summary_cats
        if self.current_summary not in self.summary_cats.keys() or reset_cats:
            self.summary_cats[self.current_summary] = []

    def reset_summary(self, summary_name=None):
        '''Reset the df and list of cat summaries assigned to the given summary name.

        Keyword Arguments:
        summary_name - The name the summary is assigned to. Must be a string.
        '''
        assert type(summary_name) == str or summary_name == None, 'summary_name must be str, {} given'.format(type(summary_name))
        if summary_name == None:
            summary_name = self.current_summary
        self.summary_cats[summary_name] = []
        self.summary[summary_name] = None

    def get_summary(self, summary_name=None):
        '''Returns current summary dataframe'''
        if summary_name == None:
            return self.summary[self.current_summary]
        else:
            return self.summary[summary_name]

    def basic_summary(self, player=None, categories=[], adv_categories=[], aggregator='AVG', groupby='season', groupby_team=False,
        return_teams=False, lastteam_only=True,convert_ids=True, summary_name=None, modern_only=True, playoffs='all',
        adv_stats=False, player_fields=True, years=None, min_date=None, max_date=None, suppress_query=True):
        '''Reads boxscore data and returns a summary dataframe grouped over desired category.
        Writes the df to the desired summary key so multiple summaries can be stored.

        player - The player to return stats for, must be a tuple/list in the form of LastName, FirstName (default None)
		categories - THe fields to read from the boxscore table (default [])
        categories - THe fields to read from the adv_boxscore table (default [])
        aggregator - The aggregator to summarise boxscore stats (default 'AVG')
		groupby - The field to groupby stats over (default 'season')
		groupby_team - Boolean, set to true if team field required per line (default False)
        return_teams - Boolean, if set to true will return the teams but not use as groupby (default False)
        lastteam_only - Boolean, if set to true and return teams is activated, only return the last team the player was on (default True)
		convert_ids - Boolean, set to True to convert ids to values (default True)
		summary_name - The summary to write the df to, if none given will write to current summary (default None)
		modern_only - Boolean, if true only returns seasons after 1983 (default True)
        playoffs - 'all': all games are returned, 'playoffs': only playoff games returned, 'regular': only regular season games returned (default 'all')
        adv_stats - Boolean, if true will also return adv boxscore stats (default False)
        years - Only return seasons within the years provided. If tuple or list of len 2 provided,
        will take these as inclusive bounds. If int provided, will only return seasons of the exact value provided. (default None)
        min_date - Only return games on or after this date (default None)
        max_date - Only return games on or before this date (default None)
        suppress_query - If True, will print the query submitted to SQL. (default True)
        '''
        extra_group, stat_str, adv_str, adv_table, where_clause, extra_groupby = '', '', '', '', '', ''

        if summary_name:
            self.set_summary(summary_name, reset_cats=True)
        else:
            self.reset_summary()

        if player:
            assert type(player) == tuple or type(player) == list, 'Player must be a tuple or list. {} given'.format(type(player))
            assert len(player) == 2, 'Player must be in format (Last_Name, FirstName). Length is {}'. format(len(player))

            player_ids = list(self.read_table(get_str='SELECT player_id FROM players WHERE last_name = "{}" AND first_name="{}"'.format(player[0], player[1])).loc[:,'player_id'])
            assert len(player_ids) == 1, 'Wrong number of player matches. Returned {} matches.'.format(len(player_ids))

            where_clause = add_where(where_clause, 'b.player_id = {}'.format(player_ids[0]))

        if years:
            if type(years) == tuple or type(years) == list:
                assert len(years) == 2, '2 years must be provided in tuple or list form'
                where_clause = add_where(where_clause, 'g.season >= {0} AND g.season <= {1}'.format(years[0], years[1]))

            else:
                where_clause = add_where(where_clause, 'g.season = {0}'.format(years))
        elif modern_only:
            where_clause = add_where(where_clause, 'g.season > 1983')

        # add min and max date where clauses if they are provided
        if min_date:
            where_clause = add_where(where_clause, "g.date_game >= '{}'".format(min_date.strftime('%Y-%m-%d')))
        if max_date:
            where_clause = add_where(where_clause, "g.date_game <= '{}'".format(max_date.strftime('%Y-%m-%d')))

        # check where clause exists, then add playoffs filter to it
        # assert(where_clause != '', 'where clause not defined, must be defined at this point')
        if playoffs == 'playoffs':
            where_clause = add_where(where_clause, 'p.series_id <> 23 and p.series_id IS NOT NULL')
            playoff_table = 'LEFT JOIN playoffgames p on b.game_id = p.game_id'
        elif playoffs == 'regular':
            where_clause = add_where(where_clause, '(p.series_id = 23 or p.series_id IS NULL)')
            playoff_table = 'LEFT JOIN playoffgames p on b.game_id = p.game_id'
        else:
            playoff_table = ''
        # if neither add nothing to clause, want to return all games

        boxscore_columns = list(self.read_table(get_str='SHOW COLUMNS FROM boxscores').loc[:,'Field'])
        des_categories = [x for x in boxscore_columns if '_id' not in x] if categories==[] else categories
        stat_str = ', '. join(['{0}(b.{1}) AS {1}'.format(aggregator, cat) for cat in des_categories])

        if adv_stats:
            adv_columns = list(self.read_table(get_str='SHOW COLUMNS FROM adv_boxscores').loc[:,'Field'])
            des_adv = [x for x in adv_columns if '_id' not in x] if adv_categories==[] else adv_categories
            adv_agg = aggregator if aggregator == '' else 'AVG'
            adv_str = ','+', '. join(['{0}(a.{1}) AS {1}'.format(adv_agg, cat) for cat in des_adv])

            adv_table = 'LEFT JOIN adv_boxscores a ON b.game_id = a.game_id and b.player_id = a.player_id'

        pct_agg = aggregator if aggregator == '' else 'SUM'
        for shot_type in ['fg','fg3','ft']:
            if shot_type in des_categories and shot_type + 'a' in des_categories:
                shot_str = ', {0}(b.{1})/{0}(b.{1}a) AS {1}_pct'.format(pct_agg, shot_type)
                stat_str += shot_str
        # add fg2 stats if fg and fg3 stats are in data
        if categories == []:
            for attempt in ['', 'a']:
                stat_str += ', {0}(b.fg{1}-b.fg3{1}) AS fg2{1}'.format(aggregator, attempt)
            stat_str += ', {0}(b.fg-b.fg3)/{0}(b.fga-b.fg3a) AS fg2_pct'.format(pct_agg)

        if 'ast' in des_categories and 'tov' in des_categories:
            stat_str += ', {0}(b.ast)/{0}(b.tov) AS assist_tov'.format(pct_agg)
        if sum([(x in des_categories) for x in ['pts','fga','fta']]) == 3:
            stat_str += ', {0}(b.pts)/(2*({0}(b.fga)+0.44*{0}(b.fta))) as ts_pct'.format(pct_agg)

        if groupby == 'game_id':
            extra_group = 'g.home_pts, g.home_team_id, g.visitor_pts, g.visitor_team_id, g.date_game, '
        elif player == None and player_fields:
            extra_group = 'b.player_id, '
            extra_groupby = 'b.player_id, '
        if groupby_team:
            extra_groupby = 'b.team_id, ' + extra_groupby
            extra_group = 'b.team_id, ' + extra_group
        elif return_teams:
            extra_group = 'GROUP_CONCAT(DISTINCT b.team_id ORDER BY g.date_game ASC) AS team_ids, ' + extra_group
        no_groupby = (groupby == '' or groupby == None)
        if no_groupby:
            return_season = 'MIN(g.season) AS min_season, MAX(g.season) AS max_season, '
        else:
            return_season = 'g.season, '

        if aggregator == '':
            full_str = '''SELECT {0}g.season, g.date_game, {1}{4}
                        FROM boxscores b
                        {5}
                        LEFT JOIN games g ON b.game_id = g.game_id
                        {2}
                        {3}
                        ORDER BY g.date_game ASC'''.format(extra_group, stat_str, playoff_table, where_clause,
                                                            adv_str, adv_table)
        else:
            full_str = '''SELECT {0}{5} COUNT(b.pts) AS game_count, {1}{6}
                        FROM boxscores b
                        {7}
                        LEFT JOIN games g ON b.game_id = g.game_id
                        {2}
                        {3}
                        GROUP BY {4}'''.format(extra_group, stat_str, playoff_table, where_clause,
                                                extra_groupby, return_season, adv_str, adv_table)
            if not no_groupby:
                full_str += '''g.{0}
                        ORDER BY g.{0} ASC'''.format(groupby)

        to_add = ['min_season','max_season'] if no_groupby else ['season']
        self.summary_cats[self.current_summary] += to_add

        if not suppress_query:
            print(full_str)
        start_time = time.time()
        self.summary[self.current_summary] = self.read_table(get_str=full_str)
        print('{} SQL query takes {:.1f} seconds.'.format(self.current_summary, time.time() - start_time))

        if self.summary[self.current_summary].empty:
            print('{} dataframe is empty.'.format(self.current_summary))
        else:
            if return_teams and lastteam_only:
                self.summary[self.current_summary].loc[:,'team_id'] = self.summary[self.current_summary]['team_ids'].str.split(',').str[-1].astype(int)
                self.summary[self.current_summary] = self.summary[self.current_summary].drop(columns='team_ids', axis=1)

            if convert_ids:
                self.convert_ids('player', ['last_name','first_name','height','weight'])
                self.convert_ids('team', ['abbreviation'], column_convert={'abbreviation':'team'})
                if groupby == 'game_id':
                    for home_visitor in ['home', 'visitor']:
                        self.convert_ids('team', ['abbreviation'], header_override= home_visitor+ '_team_id', column_convert={'abbreviation': home_visitor + '_team'})
                    self.clean_games(game_total=~team)

    def convert_ids(self, id_type, des_columns, column_convert=None, header_override=None, keep_id=False):
        '''Use to merge on id column of a table in the db. i.e. convert player_id to player name.
        Takes current summary df and replaces the id column with the desired columns from the given table in the database.

        Keyword Arguments:
        id_type - the id name, must refer to a table name and the id name of the header
        des_columns - the columns from the id table to attach to the summary df
        column_convert - a dict where keys are original header names and values are desired values. Converts any relevant header names to desired name (default None)
        keep_id - If true, will keep the id column (default None)
        '''
        id_header = id_type + '_id'
        if id_header in self.summary[self.current_summary].columns or header_override != None:
            id_df = self.read_table(id_type+'s', [id_header] + des_columns)
            left_header = id_header if header_override == None else header_override
            return_df = self.summary[self.current_summary].merge(id_df, how='left', left_on=left_header, right_on=id_header)
            if not keep_id:
                for header in [left_header, id_header]:
                    if header in return_df.columns:
                        return_df = return_df.drop(columns=[header])

            self.summary_cats[self.current_summary] += des_columns
            self.summary_cats[self.current_summary] = list(set(self.summary_cats[self.current_summary]))
            if column_convert:
                self.summary_cats[self.current_summary] = [column_convert[x] if x in column_convert.keys() else x for x in self.summary_cats[self.current_summary]]
                return_df = return_df.rename(columns=column_convert)
            columns = [x for x in return_df.columns if x not in self.summary_cats[self.current_summary]]
            self.summary[self.current_summary] = return_df[self.summary_cats[self.current_summary] + columns]

    def df_ids(self, df, id_type, des_columns, column_convert=None, header_override=None, keep_id=False):
        '''Use to merge on id column of a table in the db. i.e. convert player_id to player name.
        Takes df provided and replaces the id column with the desired columns from the given table in the database.

        Keyword Arguments:
        id_type - the id name, must refer to a table name and the id name of the header
        des_columns - the columns from the id table to attach to the summary df
        column_convert - a dict where keys are original header names and values are desired values. Converts any relevant header names to desired name (default None)
        keep_id - If true, will keep the id column (default None)
        '''
        id_header = id_type + '_id'
        assert id_header in df.columns or header_override != None, 'id_type not in df columns.'
        table_name = id_type+'s' if id_type[-1] != 's' else id_type
        id_df = self.read_table(table_name, [id_header] + des_columns)
        left_header = id_header if header_override == None else header_override
        return_df = df.merge(id_df, how='left', left_on=left_header, right_on=id_header)
        if not keep_id:
            for header in [left_header, id_header]:
                if header in return_df.columns:
                    return_df = return_df.drop(columns=[header])

        if column_convert:
            return_df = return_df.rename(columns=column_convert)

        return return_df

    def apply_qualifiers(self, qualifiers, return_subset=True, sort_on=None, sort_asc=False, all_columns=False):
        '''Applies a dict of qualifiers to the summary df, outputing the df with only items matching the qualifiers.
        Each key should be given in a string '>10' or '<10' form.

        Keyword Arguments:
        qualifier - dict of qualifier column name and string qualifier
        return_subset - if True, only return the items matching qualifier, otherwise returns full df with qualifier column (default True)
        sort_on - will return the df sorted by this column, if not set will sort on the qualifier column (default None)
        sort_asc - if true, sort ascending (default False)
        all_columns - by default will only return the summary cats and the qualifier columns (default False)
        '''
        return_df = self.summary[self.current_summary].copy()
        condition_series = pd.Series()
        for column, threshold in qualifiers.items():
            if sort_on == None:
                sort_on = column
            if re.match('<|>.*', threshold):
                more_than = threshold[0] == '>'
                threshold_float = float(threshold[1:])
            else:
                more_than = False
                theshold_float = float(threshold)

            if more_than:
                current_series = return_df.loc[:,column] >= threshold_float
            else:
                current_series = return_df.loc[:,column] <= threshold_float
            if condition_series.empty:
                condition_series = current_series
            else:
                condition_series = condition_series & current_series

        current_cats = self.summary_cats[self.current_summary]
        desired_columns = current_cats + [x for x in list(qualifiers.keys()) if x not in current_cats]
        if sort_on:
            desired_columns.append(sort_on)

        if return_subset:
            return_df = return_df[condition_series].sort_values(sort_on, ascending=sort_asc)
        else:
            return_df.loc[:,'qualifier'] = condition_series
            desired_columns.append('qualifier')

        if all_columns:
            non_columns = [column for column in return_df.columns if column not in desired_columns]
            return return_df[desired_columns + non_columns]
        else:
            return return_df[desired_columns]

    def find_streaks(self, qualifier_dict, summary_name=None, groupby='player'):
        '''Return a df of the top streaks matching the qualifiers given

        Keyword Arguments:
        qualifier_dict - the dict of qualifiers to apply
        summary_name - the summary df to apply the qualifiers to, if not set use current summary df (default None)'''
        assert groupby in ['player'] or 'team' in groupby, 'Groupby must be player or team.'
        groupby_column = ['first_name', 'last_name'] if groupby=='player' else groupby

        if summary_name:
            self.set_summary(summary_name)
        streak_df = self.apply_qualifiers(qualifier_dict, return_subset=False, sort_on='date_game', sort_asc=True, all_columns=False).copy()
        if 'first_name' in streak_df.columns:
            streak_df.loc[:,'first_name'] = streak_df.loc[:,'first_name'].fillna('')
        streak_df = streak_df.dropna(subset=[groupby_column] if type(groupby_column) != list else groupby_column)

        streak_df.loc[:,'cumsum'] = streak_df.groupby(groupby_column, as_index=False).cumsum()['qualifier']
        streak_df.loc[:,'prev_total'] = streak_df.loc[:,'cumsum'].where(~streak_df.loc[:,'qualifier'], np.nan)
        streak_df.loc[:,'ffill'] = streak_df.groupby(groupby_column, as_index=False)['prev_total'].transform(lambda x: x.ffill().fillna(0))['prev_total']
        streak_df.loc[:,'cumsumffill'] = streak_df.loc[:,'cumsum'] - streak_df.loc[:,'ffill']
        streak_df.loc[:,'after'] = streak_df.groupby(groupby_column, as_index=False)['qualifier'].shift(-1)['qualifier'].fillna(False)
        streak_df.loc[:,'streak'] = streak_df.loc[:,'cumsumffill'].where(~streak_df.loc[:,'after'], 0)

        summary_columns = ['date_game','streak']
        if groupby == 'player':
            summary_columns = ['last_name','first_name', 'team'] + summary_columns
        else:
            summary_columns = [groupby_column] + summary_columns

        return_summary = streak_df[streak_df.loc[:,'streak'] > 1]
        return_summary = return_summary.sort_values('streak', ascending=False)[summary_columns].reset_index(drop=True)
        return_summary.index = return_summary.index+1

        return return_summary.rename(columns={'date_game':'streak_end'})

    def clean_games(self, summary_name=None, game_total=False):
        '''Classifies rows as home team, checks games data matches boxscore data, removes surplus columns.

        Keyword Argurments:
        summary_name - The summary to work on (default None)
        '''
        if summary_name != None:
            self.set_summary(summary_name)
        df = self.get_summary()

        if game_total:
            pts_errors = sum(df.loc[:,'pts'] != (df.loc[:,'home_pts'] + df.loc[:,'visitor_pts']))
            assert pts_errors == 0, "{} games with points mismatch".format(pts_errors)
        else:
            home_index = (df.loc[:,'home_team'] == df.loc[:,'team'])
            df.loc[:,'is_home'] = home_index

            expected_homes = sum(home_index) / len(home_index)

            assert expected_homes  == 0.5, "Expected 50% homes games, returned {:.2%}".format(expected_homes)
            home_games = df[home_index]
            visitor_games = df[~home_index]
            home_errors = sum(home_games.loc[:,'pts'] != home_games.loc[:,'home_pts'])
            visitor_errors = sum(visitor_games.loc[:,'pts'] != visitor_games.loc[:,'visitor_pts'])
            assert home_errors == 0, "{} home games with points mismatch".format(home_errors)
            assert home_errors == 0, "{} visitor games with points mismatch".format(visitor_errors)

        remove_cols = ['home_pts','visitor_pts']
        if not game_total:
            remove_cols += ['home_team','visitor_team']

        df = df.drop(labels=remove_cols, axis=1)
        self.summary_cats[self.current_summary] = [x for x in self.summary_cats[self.current_summary] if x not in remove_cols]
        self.summary[self.current_summary] = df

    def season_games(self, seasons=(1984,2100), convert_ids=True, summary_name='games'):
        '''Reads the game data for the given season: each game's date, scores etc. Df is then stored in summary named "games".

        Keyword arguments:
        seasons - The season or min/max seasons to read games for, must be an integer or tuple/list of length 2.
        convert_ids - Boolean, if set will convert ids to values (default True)
        summary_name - The name to store the df as (default 'games')
        '''
        self.set_summary(summary_name)

        if type(seasons) == list or type(seasons) == tuple:
            assert len(seasons) == 2, 'Must provide int or tuple/list of length 2'
            min_season = seasons[0]
            max_season = seasons[1]
            season_str = 'g.season as season, '
        else:
            assert type(seasons) == int, 'Must provide int or tuple/list of length 2'
            min_season, max_season = seasons, seasons
            season_str = ''

        if convert_ids:
            team_str = 't1.abbreviation AS home_team, t2.abbreviation AS visitor_team, '
            team_join = '''LEFT JOIN teams t1 on g.home_team_id = t1.team_id
                            LEFT JOIN teams t2 on g.visitor_team_id = t2.team_id'''
        else:
            team_str = 'g.home_team_id AS home_team, g.visitor_team_id AS visitor_team, '
            team_join = ''

        sql_str = '''SELECT {2}date_game, {3}
                        home_pts > visitor_pts AS home_victory FROM games g
                    LEFT JOIN playoffgames p on g.game_id = p.game_id
                    {4}
                    WHERE g.season >= {0} AND g.season <= {1} AND (p.series_id = 23 or p.series_id IS NULL)'''.format(
                        min_season, max_season, season_str, team_str, team_join)

        season_games = self.read_table(get_str=sql_str)
        season_games.loc[:,'visitor_victory'] = 1 - season_games.loc[:,'home_victory']

        self.summary[self.current_summary] = season_games

    def standings(self, max_date=None, rank_method='min', override_df=pd.DataFrame()):
        '''Calculates the standings at the date provided.
        The games summary for the given season must be already stored. Run season_games if it is not.
        Returns the games included in the stanndings and the league standings. Ties are all given the same ranking by default.

        Keyword arguments:
        max_date - The date to take the standings on. If not provided, will give current standings (default None)
        rank_method - The method used to sort the standings (default 'min')
        override_df - If provided, will use this df instead of stored df (default None)
        '''
        if override_df.empty:
            assert 'games' in self.summary.keys(), '"games" summary not loaded, please run season_games function.'
            games_restricted = self.summary['games'].copy()
        else:
            games_restricted = override_df.copy()

        if max_date != None:
            season_start = games_restricted['date_game'].min()
            assert max_date >= season_start, 'Date provided is before the beginning of the season. Please provide date after: {}'.format(season_start)

            games_restricted = games_restricted[games_restricted['date_game'] <= max_date]

        home_visitor = ['home', 'visitor']
        standings_list = []
        for i in range(2):
            standings_temp = games_restricted.groupby(home_visitor[i] + '_team').agg(
                {home_visitor[i]+'_victory':sum,
                home_visitor[-1-i]+'_victory':sum}).rename(columns={home_visitor[i]+'_victory':'W_'+home_visitor[i],
                                                                    home_visitor[-1-i]+'_victory':'L_'+home_visitor[i]})
            standings_temp.index.name = None
            standings_list.append(standings_temp)

        standings = pd.concat(standings_list, axis=1)
        for W_L in ['W', 'L']:
            standings.loc[:,W_L] = standings[W_L+'_home']+ standings[W_L+'_visitor']
        standings.loc[:,'Played'] = standings['W'] + standings['L']
        standings.loc[:,'W_pct'] = standings['W'] / standings['Played']
        standings.loc[:,'position'] = standings['W_pct'].rank(ascending=False, method=rank_method).astype(int)

        return games_restricted, standings.sort_values('position')

    def wpct_all(self, override_df=pd.DataFrame()):
        '''Returns a df containings the W pct for teams per season.
        Must run the season games query separately over more than 1 season.

        Keyword arguments:
        override_df - If provided this df will be used instead of the stored games df
        '''
        if override_df.empty:
            if 'games' not in self.summary.keys():
                print('"games" summary loaded, season_games function run for all modern (>1983) seasons.')
                self.season_games()
            all_games = self.summary['games'].copy()
        else:
            all_games = override_df.copy()

        all_standings = []
        for season, season_df in all_games.groupby('season'):
            temp_df = season_df.drop('season', axis=1)
            _, temp_standings = self.standings(override_df=temp_df)
            temp_standings = temp_standings.reset_index().rename(columns={'index':'team'})
            temp_standings.loc[:,'season'] = season
            all_standings.append(temp_standings[['team','season','W_pct']])

        return pd.concat(all_standings)

    def playoffseries(self, games_name='playoffgames', series_name='playoffseries', modern_only=True):
        '''Queries database for playoff games. Runs function to derive information from the raw playoff game data.
        i.e. number of games in the series, possible number of games, cumulative series score
        Creates a df containing each individual game and a df with each series. Dfs are stored in summary dict.

        Keyword arguments:
        games_name - The key to use in the summary dict for the playoff games df (default 'playoffgames')
        series_name - The key to use in the summary dict for the playoff series df (default 'playoffseries')
        modern_only - Only return seasons after 1983 (default True)
        '''
        sql_str = '''SELECT p.playoffgames_id, p.game_id, p.series_id, p.game_no, g.season, g.date_game, g.home_team_id, g.visitor_team_id, g.home_pts > g.visitor_pts AS home_wins
            FROM playoffgames p
            INNER JOIN games g ON g.game_id = p.game_id
            WHERE p.series_id <> 23'''
        if modern_only:
            sql_str += ' and g.season > 1983'

        playoffgames = self.read_table(get_str=sql_str)

        team_ids = playoffgames[['home_team_id','visitor_team_id']]
        playoffgames.loc[:,'teams_combo'] = team_ids.min(axis=1).astype(str) + ',' + team_ids.max(axis=1).astype(str)

        # create a unique value for each series so it is easy to group each series
        playoffgames = playoffgames.sort_values(['season', 'series_id', 'teams_combo'])
        false_series_ids = playoffgames.groupby(['season', 'series_id', 'teams_combo'], as_index=False)['game_id'].min().rename(columns={'game_id':'false_series_id'})
        playoffgames = playoffgames.merge(false_series_ids, how='left', on=['season', 'series_id', 'teams_combo'])
        playoffgames = playoffgames.drop(['teams_combo'], axis=1)

        playoffgames.loc[:,'homecourt_team_id'] = playoffgames['home_team_id'].where(playoffgames['game_no']==1, np.nan)
        playoffgames.loc[:,'visitorcourt_team_id'] = playoffgames['visitor_team_id'].where(playoffgames['game_no']==1, np.nan)
        playoffgames.loc[:,'homecourt_team_id'] = playoffgames.loc[:,'homecourt_team_id'].fillna(method='ffill').astype(int)
        playoffgames.loc[:,'visitorcourt_team_id'] = playoffgames.loc[:,'visitorcourt_team_id'].fillna(method='ffill').astype(int)

        playoffgames.loc[:,'homecourt_wins'] = playoffgames['home_wins'].where(playoffgames['home_team_id'] == playoffgames['homecourt_team_id'], 1-playoffgames['home_wins'])
        playoffgames.loc[:,'homecourt_cumwins'] = playoffgames.groupby('false_series_id')['homecourt_wins'].cumsum()
        playoffgames.loc[:,'visitorcourt_cumwins'] = playoffgames['game_no'] - playoffgames['homecourt_cumwins']
        playoffgames.loc[:,'series_score'] = playoffgames['homecourt_cumwins'].astype(str) + '-' + playoffgames['visitorcourt_cumwins'].astype(str)

        assert (playoffgames['homecourt_cumwins'] + playoffgames['visitorcourt_cumwins']).max() <= 7, 'More than 7 games in a series'
        assert playoffgames['homecourt_cumwins'].max() <= 4, 'More than 4 wins by a team with homecourt in a series'
        assert playoffgames['visitorcourt_cumwins'].max() <= 4, 'More than 4 wins by a team without homecourt in a series'

        # create df that has one row per series, most columns are just duplicated in initial table, others require some aggregation
        playoffseries = playoffgames.groupby('false_series_id').agg({'series_id':'first',
                                                    'season':'first',
                                                    'homecourt_team_id':'first',
                                                    'visitorcourt_team_id':'first',
                                                    'game_no':'max',
                                                    'homecourt_cumwins':'max',
                                                    'visitorcourt_cumwins':'max',
                                                    'series_score':','.join}
                                                ).rename(columns={'game_no':'games_required',
                                                                  'homecourt_cumwins':'homecourt_wins',
                                                                  'visitorcourt_cumwins':'visitorcourt_wins',
                                                                 'series_score':'series_timeline'}
                                                ).reset_index()
        playoffseries.loc[:,'series_timeline'] = '0-0,'+playoffseries['series_timeline']

        self.season_games(convert_ids=False)
        w_pct = self.wpct_all()
        for h_v in ['home', 'visitor']:
            playoffseries = playoffseries.merge(w_pct, how='left', left_on=[h_v+'court_team_id','season'], right_on=['team','season']
                    ).drop(columns=['team']
                    ).rename(columns={'W_pct':h_v+'court_wpct'})
        playoffseries.loc[:,'wpct_diff'] = playoffseries['homecourt_wpct'] - playoffseries['visitorcourt_wpct']

        for team_column in ['homecourt_team_id', 'visitorcourt_team_id']:
            playoffseries = self.df_ids(playoffseries, 'team', ['abbreviation'], column_convert={'abbreviation':team_column.replace('_id','')}, header_override=team_column)
        playoffseries = self.df_ids(playoffseries, 'series', ['series_name','conference','round'])
        playoffseries = playoffseries.set_index('false_series_id')

        playoffseries.loc[:,'series_games'] = playoffseries[['homecourt_wins','visitorcourt_wins']].max(axis=1) * 2 - 1
        current_check = playoffseries['season'] == dt.date.today().year
        assert ((playoffseries['series_games'].isin([5,7])) | (current_check)).mean() == 1, 'series games must be 5 or 7.'
        gamesreq_check = ((playoffseries['games_required'] > playoffseries['series_games']) & (~current_check)).sum()
        assert gamesreq_check == 0, 'Games required must be less than or equal to series games, in {} cases it is not'.format(gamesreq_check)

        playoffseries.loc[:,'homecourt_victory'] = playoffseries['homecourt_wins'] > playoffseries['visitorcourt_wins']
        playoffseries.loc[:,'victor'] = playoffseries['homecourt_team'].where(playoffseries['homecourt_victory'], playoffseries['visitorcourt_team'])
        playoffseries.loc[:,'loser'] = playoffseries['homecourt_team'].where(~playoffseries['homecourt_victory'], playoffseries['visitorcourt_team'])

        series_wins = playoffgames[['false_series_id','game_no', 'home_wins']].set_index(['false_series_id','game_no']).unstack()
        series_wins.columns = ['homegame_'+str(header[1]) for header in series_wins.columns]
        playoffseries = playoffseries.merge(series_wins, how='left', left_index=True, right_index=True)

        self.set_summary(games_name)
        self.summary[self.current_summary] = playoffgames
        self.set_summary(series_name)
        self.summary[self.current_summary] = playoffseries

    def win_probability(self, score, playoffseries=pd.DataFrame(), comeback=True, flipscore=True, force_game=None,
                        rounds=[1,2,3,4], series_games=[5,7], wpct_column='wpct_diff', wpct=(-1,1), seasons=1983):
        '''Calculates the probability of a team winning or forcing a game given a series score.
        By default, calculates the probability of a comeback win and is ambivalent to the home team in the score.
        If the score is even, it is considered a comeback for the team without homecourt advantage to win.
        Prints the probability and returns all instances where a team was successul.

        Keyword arguments:
        score - A string of the score of interest, in format "[0-4]-[0-4]"
        playoffseries - The df containing playoff series. If not provided, will use default query of db (default pd.DataFrame())
        comeback - If True, the probability refers to the team currently losing coming back and winning (default True)
        flipscore - If True, the score provided will also be flipped, i.e. ambivalent to homecourt advantage (default True)
        force_game - Set this to calculate the probabilty of forcing a game instead of just winning (default None)
        rounds - The playoff rounds to include, provide a list of ints.
                    Round 1 is the finals, each subsequent number is the round less significant.
                    Round 4 is the current first round of the playoffs  (default [1,2,3,4])
        series_games - Specify the length of rounds to include as a list of ints (default [5,7])
        wpct_column - The column to apply the wpct filter to (default 'wpct_diff')
        wpct - Games with a wpct in this range will be included (default (-1,1))
        seasons - Specify the seasons to include. If an int is provided, it will be used a min,
                    otherwise provide a min/max list/tuple (default 1983)
        '''
        # if no df is specified, query db if required and use store playoffseries df
        if playoffseries.empty:
            if 'playoffseries' not in self.summary.keys():
                self.playoffseries()
            playoffseries = self.summary['playoffseries']

        filtered_series = playoffseries[playoffseries['round'].isin(rounds)]
        filtered_series = filtered_series[filtered_series['series_games'].isin(series_games)]
        filtered_series = filtered_series[filtered_series[wpct_column].between(wpct[0],wpct[1])]

        if type(seasons) == int:
            seasons = (seasons, filtered_series['season'].max()+1)

        assert type(seasons) == tuple or type(seasons) == list, 'seasons must be an int or list/tuple'
        assert len(seasons) == 2, 'seasons must be an int or of length 2, {} of {} provided'.format(type(seasons), len(seasons))
        filtered_series = filtered_series[filtered_series['season'].between(seasons[0], seasons[1])]

        home_wins = int(score[0])
        visitor_wins = int(score[-1])

        if home_wins == visitor_wins:
            home_ahead = True
            flipscore = False
        else:
            home_ahead = home_wins > visitor_wins
        check_homewin = not home_ahead if comeback else home_ahead
        scores = [score, score[::-1]] if flipscore else [score]

        if flipscore:
            prob_str = 'losing'
        else:
            prob_str = 'home' if check_homewin else 'visitor'
        winning_str = 'forcing game {}'.format(force_game) if force_game else 'winning'
        print('Probability of {} team {}: '.format(prob_str, winning_str), end='')

        packages = []
        for filter_score in scores:
            filter1 = filtered_series['series_timeline'].str.contains(filter_score)
            packages.append(winfilter(filtered_series, filter1, home_victor=check_homewin, force_game=force_game))
            check_homewin = not check_homewin

        package2 = (0,0,pd.DataFrame()) if len(packages) == 1 else packages[1]

        return playoff_probability(filtered_series, packages[0], package2)

def winfilter(playoffseries, filter1, home_victor=True, force_game=None):
    '''Applies the filter specified and then finds the occasions where teams have won/forced game x from this position.
    Returns:
    num - the number of times teams in the applied filter have won.
    den - the total number of times this filter has occurred.
    combined_filter - the filter matching scenarios where teams have won in this scenario

    Keyword arguments:
    playoffseries - The df containing playoff series.
    filter1 - The filter to use to get the series in question, must be same size as playoffseries df
    home_victor - If True, will return results for cases where the home team is victorious (default True)
    force_game - If set, the series reaching the given game will be considered a "victory"
                    i.e. results will now relate to the scenario of forcing game x (default None)
    '''
    if force_game:
        filter2 = playoffseries['games_required'] >= force_game
    else:
        filter2 = playoffseries['homecourt_victory'] if home_victor else ~playoffseries['homecourt_victory']
    combined_filter = filter1 & filter2
    num = sum(combined_filter)
    den = sum(filter1)
    return num, den, combined_filter

def playoff_probability(playoffseries, package1, package2=(0,0,pd.DataFrame())):
    '''Takes the num, den and combined filter from winfilter function.
    Returns the probability results and instances where teams have been successful.

    Keyword arguments:
    playoffseries - The df containing playoffseries.
    package1 - a tuple containig num, den and combined_filter.
    package2 - a tuple containig num, den and combined_filter (default (0,0,pd.DataFrame()))
    '''
    num = package1[0] + package2[0]
    den = package1[1] + package2[1]
    print('{:.1%} ({}/{})'.format(num/den, num, den))
    des_columns = ['season','series_name','homecourt_team','homecourt_wpct','visitorcourt_team','visitorcourt_wpct',
        'wpct_diff','homecourt_wins','visitorcourt_wins','victor','loser','homecourt_victory','series_timeline']

    if not package2[2].empty:
        return playoffseries[package1[2]|package2[2]][des_columns].sort_values('season', ascending=False)
    else:
        return playoffseries[package1[2]][des_columns].sort_values('season', ascending=False)
