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
            Raise("unix socket or port/host must be provided.")
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

            float_columns = [header for header, item in zip(table_columns, table_lines[0]) if isfloat(item)]
            
            try:
                df[float_columns] = df[float_columns].apply(pd.to_numeric)
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
        
    def basic_summary(self, player=None, categories=[], aggregator='AVG', groupby='season', team=False, 
        convert_ids=True, summary_name=None, modern_only=True, playoffs='all', player_fields=True, years=None):
        '''Reads boxscore data and returns a summary dataframe grouped over desired category.
        Writes the df to the desired summary key so multiple summaries can be stored.

        player - The player to return stats for, must be a tuple/list in the form of LastName, FirstName (default None)
		categories - THe fields to read from the boxscore table (default [])
		aggregator - The aggregator to summarise boxscore stats (default 'AVG')
		groupby - The field to groupby stats over (default 'season')
		team - Boolean, set to true if team field required per line (default False)
		convert_ids - Boolean, set to True to convert ids to values (default True)
		summary_name - The summary to write the df to, if none given will write to current summary (default None)
		modern_only - Boolean, if true only returns seasons after 1983 (default True)
        playoffs - 'all': all games are returned, 'playoffs': only playoff games returned, 'regular': only regular season games returned (default 'all')
        '''
        extra_group, stat_str, where_clause, extra_groupby = '', '', '', ''
        
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

#         aggregator = 'SUM' if cum else 'AVG'
        boxscore_columns = list(self.read_table(get_str='SHOW COLUMNS FROM boxscores').loc[:,'Field'])
        des_categories = [x for x in boxscore_columns if '_id' not in x] if categories==[] else categories
        stat_str = ', '. join(['{}(b.{}) AS {}'.format(aggregator, cat, cat) for cat in des_categories])

        for shot_type in ['fg','fg3','ft']:
            if shot_type in des_categories and shot_type + 'a' in des_categories:
                shot_str = ', {0}(b.{1})/{0}(b.{1}a) AS {1}_pct'.format(aggregator, shot_type)
                stat_str += shot_str
        if 'ast' in des_categories and 'tov' in des_categories:
            stat_str += ', {0}(b.ast)/{0}(b.tov) AS assist_tov'.format(aggregator)
        if sum([(x in des_categories) for x in ['pts','fga','fta']]) == 3:
            stat_str += ', {0}(b.pts)/(2*({0}(b.fga)+0.44*{0}(b.fta))) as ts_pct'.format(aggregator)
        
        if groupby == 'game_id':
            extra_group = 'g.home_pts, g.home_team_id, g.visitor_pts, g.visitor_team_id, g.date_game, '
        elif player == None and player_fields:
            extra_group = 'b.player_id, '
            extra_groupby = 'b.player_id, '
        if team:
            extra_groupby = 'b.team_id, ' + extra_groupby
            extra_group = 'b.team_id, ' + extra_group
        no_groupby = (groupby == '' or groupby == None)
        if no_groupby:
            return_season = 'MIN(g.season) AS min_season, MAX(g.season) AS max_season, '  
        else:
            return_season = 'g.season, '

        if aggregator == '':
            full_str = '''SELECT {0}g.season, g.date_game, {1} 
                        FROM boxscores b 
                        LEFT JOIN games g ON b.game_id = g.game_id
                        {2}
                        {3}
                        ORDER BY g.date_game ASC'''.format(extra_group, stat_str, playoff_table, where_clause)
        else:
            full_str = '''SELECT {0}{5} COUNT(b.pts) AS game_count, {1} 
                        FROM boxscores b 
                        LEFT JOIN games g ON b.game_id = g.game_id
                        {2}
                        {3}
                        GROUP BY {4}'''.format(extra_group, stat_str, playoff_table, where_clause, extra_groupby, return_season)
            if not no_groupby:
                full_str += '''g.{0}
                        ORDER BY g.{0} ASC'''.format(groupby)

        to_add = ['min_season','max_season'] if no_groupby else ['season']
        self.summary_cats[self.current_summary] += to_add

        print(full_str)
        start_time = time.time()
        self.summary[self.current_summary] = self.read_table(get_str=full_str)
        print('{} SQL query takes {:.1f} seconds.'.format(self.current_summary, time.time() - start_time))

        # add fg2 stats if fg and fg3 stats are in data
        headers_set = set(self.summary[self.current_summary].columns)
        if categories == [] and set(['fg','fga','fg3','fg3da']).issubset(headers_set):
            self.summary[self.current_summary].loc[:,'fg2'] = self.summary[self.current_summary].loc[:,'fg'] - self.summary[self.current_summary].loc[:,'fg3']
            self.summary[self.current_summary].loc[:,'fg2a'] = self.summary[self.current_summary].loc[:,'fga'] - self.summary[self.current_summary].loc[:,'fg3a']
            self.summary[self.current_summary].loc[:,'fg2_pct'] = self.summary[self.current_summary].loc[:,'fg2'] / self.summary[self.current_summary].loc[:,'fg2a']

        
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

    def season_games(self, season):
        '''Reads the game data for the given season: each game's date, scores etc. Df is then stored in summary named "games".

        Keyword arguments:
        season - The season to read games for, must be an integer less than or equal to current year.
        '''
        self.set_summary('games')

        sql_str = '''SELECT date_game, t1.abbreviation AS home_team, t2.abbreviation AS visitor_team, home_pts > visitor_pts AS home_victory FROM games g
                    LEFT JOIN playoffgames p on g.game_id = p.game_id
                    LEFT JOIN teams t1 on g.home_team_id = t1.team_id
                    LEFT JOIN teams t2 on g.visitor_team_id = t2.team_id
                    WHERE g.season = {} and (p.series_id = 23 or p.series_id IS NULL)'''.format(season)

        season_games = self.read_table(get_str=sql_str) 
        season_games.loc[:,'visitor_victory'] = 1 - season_games.loc[:,'home_victory']

        self.summary[self.current_summary] = season_games

    def standings(self, max_date=None, rank_method='min'):
        '''Calculates the standings at the date provided. 
        The games summary for the given season must be already stored. Run season_games if it is not.
        Returns the games included in the stanndings and the league standings. Ties are all given the same ranking by default.

        Keyword arguments:
        max_date - The date to take the standings on. If not provided, will give current standings (default None)
        rank_method - The method used to sort the standings (default 'min')
        '''
        assert 'games' in self.summary.keys(), '"games" summary not loaded, please run season_games function.'
        games_restricted = self.summary['games'].copy()

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