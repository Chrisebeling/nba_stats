import argparse
import logging
import datetime as dt
import os
import configparser

from nba_stats.read_write.db_insert import SqlDataframes
from nba_stats.scraping.build_db import get_boxscore_htmls_year, get_game_soups,add_basic_gamestats, get_players_urls, get_all_players, get_colleges, get_teams, get_playoff_games
from nba_stats.read_write.config import get_dbconfig

logger = logging.getLogger()
handler = logging.StreamHandler()
file_handler = logging.FileHandler(os.path.join("logs","{}.log".format(dt.datetime.today().strftime('%Y%m%d'))))
formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-10s %(message)s')
for a_handler in [handler, file_handler]:
    a_handler.setFormatter(formatter)
logger.addHandler(handler)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)

def scrape_games(loops, count_max, action):
    stats_db = SqlDataframes()
    games_table = stats_db.read_table('games',['game_id','bref'])

    for i in range(loops):
        logger.info('Current Loop: {}'.format(i+1))
        logger.info('Running game soups...')

        cfg = get_dbconfig(section='scraping')
        checktables_cfg = cfg['check_tables'].split(',')

        id_bref_soup = get_game_soups(games_table, limit=count_max, check_tables=checktables_cfg)
        if not id_bref_soup:
            break
        add_basic_gamestats(id_bref_soup, commit_changes=action)
        if i == loops - 1:
            stats_db = SqlDataframes()
            games_max = stats_db.read_max('games','game_id')
            boxs_max = stats_db.read_max('boxscores', 'game_id')
            logger.info('FINISHED...Games remaining to scrape: {}'.format(games_max-boxs_max))

def update_games(year):
    stats_db = SqlDataframes()

    players_soups = get_players_urls()

    players = get_all_players(players_soups)
    colleges = get_colleges(players)
    teams = get_teams()

    stats_db.add_to_db(colleges, 'colleges', check_column='college')
    players_ids = stats_db.apply_mappings(players, 'colleges', ['college1', 'college2'])
    stats_db.add_to_db(players_ids, 'players', check_column='bref')

    season_boxscore_htmls = get_boxscore_htmls_year(year, regular_length=False)
    games_ids = stats_db.apply_mappings(season_boxscore_htmls, 'teams', ['home_team', 'visitor_team'])
    stats_db.add_to_db(games_ids, 'games', 'bref', 'date_game')

    playoffs_ids = get_playoff_games((year, year))
    stats_db.add_to_db(playoffs_ids, 'playoffgames', 'game_id', 'game_id')


def scrape_function():
    parser = argparse.ArgumentParser(description="Scrape boxscores where boxscore not already in db.")
    parser.add_argument('-l', '--loops', nargs='?', type=int, default=1, help='The number of loops to run.')
    parser.add_argument('-c', '--count_max', nargs='?', type=int, default=100, help='The number of games to scrape per loop.')
    parser.add_argument('-n', '--action', action='store_false', help='If set, scrape will not be sent to db')
    parser.add_argument('-t', '--scrape', nargs='?', type=str, default='both', choices=['games_only','boxscores_only','both'], 
        help='Set function use. Able to scrape games, boxscores or both.')
    parser.add_argument('-y', '--year', nargs='?', type=int, default=0, help='The season to scrape games. Must be given if scraping games.')

    args = parser.parse_args()

    if args.scrape == 'both' or args.scrape == 'games_only':
        assert args.year != 0, 'Year argument must be given to scrape games.'
        update_games(args.year)
    if args.scrape == 'both' or args.scrape == 'boxscores_only':
        scrape_games(args.loops, args.count_max, args.action)

if __name__== '__main__':
    scrape_function()