import argparse
import logging
import datetime as dt
import os

from nba_stats.read_write.db_insert import SqlDataframes
from nba_stats.scraping.build_db import get_boxscore_htmls_year, get_game_soups,add_basic_gamestats

logger = logging.getLogger()
handler = logging.StreamHandler()
file_handler = logging.FileHandler(os.path.join("logs","{}.log".format(dt.datetime.today().strftime('%Y%m%d'))))
formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-10s %(message)s')
for a_handler in [handler, file_handler]:
    a_handler.setFormatter(formatter)
logger.addHandler(handler)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)

def scrape_function():
    
    parser = argparse.ArgumentParser(description="Scrape boxscores where boxscore not already in db.")
    parser.add_argument('-l', '--loops', nargs='?', type=int, default=1, help='The number of loops to run.')
    parser.add_argument('-c', '--count_max', nargs='?', type=int, default=100, help='The number of games to scrape per loop.')
    parser.add_argument('-n', '--action', action='store_false')

    args = parser.parse_args()

    stats_db = SqlDataframes(_host="nba-stats-inst.clmw4mwgj0eg.ap-southeast-2.rds.amazonaws.com", _password="23cHcGN9PNxxUKtAzGp28kJ7u")
    games_table = stats_db.read_table('games',['game_id','bref'])

    for i in range(args.loops):
        logger.info('Current Loop: {}'.format(i+1))
        logger.info('Running game soups...')
        id_bref_soup = get_game_soups(games_table, limit=args.count_max, check_tables=['boxscores', 'fourfactors'])
        if not id_bref_soup:
            break
        add_basic_gamestats(id_bref_soup, commit_changes=args.action)
        if i == args.loops - 1:
            stats_db = SqlDataframes(_host="nba-stats-inst.clmw4mwgj0eg.ap-southeast-2.rds.amazonaws.com", _password="23cHcGN9PNxxUKtAzGp28kJ7u")
            games_max = stats_db.read_max('games','game_id')
            boxs_max = stats_db.read_max('boxscores', 'game_id')
            logger.info('FINISHED...Games remaining to scrape: {}'.format(games_max-boxs_max))
        # else:
        #     clear_output()

if __name__== '__main__':
    scrape_function()