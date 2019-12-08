# nba_stats
Contains functions used to create, manage and use a database of nba statistics.

To set database config use function update_dbconfig function. (To import us "from nba_stats.read_write.config import update_dbconfig").
This function requires a dict of settings and values.
Settings are: 'host', 'port', 'db', 'user', 'password'
Only needs to be set once.