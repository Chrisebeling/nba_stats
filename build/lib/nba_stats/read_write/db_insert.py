import datetime as dt
import pandas as pd
import numpy as np
import mysql.connector as sql
import logging

from nba_stats.read_write.config import get_dbconfig

cfg = get_dbconfig()
HOST = cfg['host'] 
PORT = int(cfg['port'])
USER = cfg['user']
PASSWORD = cfg['password']
DB = cfg['db']

logger_insert = logging.getLogger(__name__)
# handler = logging.StreamHandler()
# file_handler = logging.FileHandler("logging\\%s.log" % dt.datetime.today().strftime('%Y%m%d'))
# formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-10s %(message)s')
# for a_handler in [handler, file_handler]:
#     a_handler.setFormatter(formatter)
# logger_insert.addHandler(handler)
# logger_insert.addHandler(file_handler)
# logger_insert.setLevel(logging.INFO)

class SqlDataframes(object):
    def __init__(self, _password=PASSWORD, _user=USER, _db=DB, _host=HOST, _port=PORT,_unix=None):
        self.establish_connection()
    
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
        
    def check_schema(self, df, table_name):
        check = True
        select_str = "SELECT DISTINCT(COLUMN_NAME) FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '%s'" % table_name
        cursor = self.establish_cursor()
        cursor.execute(select_str)
        columns = [x[0] for x in cursor.fetchall()]
        for df_column in df.columns:
            if df_column not in columns:
                check = False
                logger_insert.error('Error: {} not in table schema (table: {})'.format(df_column, table_name))

        return check
        
    def create_sql_str(self, df, table_name):
        assert self.check_schema(df, table_name), 'Some df columns not in table schema'

        headers = [x.strip() for x in df.columns]
        string = "INSERT INTO %s (%s) VALUES " % (table_name, ', '.join(headers))
        row_template = "(%str%)"
        rows_str = []
        df_copy = df.copy().replace('', np.nan, regex=True).fillna('NULL')
        for idx, row in df_copy.iterrows():
            row_list = [x if x == 'NULL' else "'" + str(x).replace("'","") + "'" for x in row]
            if row_list and not set(row_list) == {'NULL'}:
                # check that row is not full of empties
                row_str = ', '.join(row_list)
                rows_str.append(row_template.replace("%str%", row_str))

        string += ', '.join(rows_str)

        return string
    
    def add_to_db(self, df, table_name, check_column=None, info_column=None):
        '''Adds a df to the db.
        When check_column is set, will only add new entries, using the check column as a cross reference.
        All columns in df must exist in table schema.
        
        Keyword attributes:
        df -- the df to add to the db
        table_name -- the name of the table in the db
        check_column -- the column to check if entries are new or already exist (Default None)
        '''
        if check_column:
            db_table = self.read_table(table_name)
            
            assert check_column in db_table.columns, 'check column (%s) not in table columns' % check_column
            assert check_column in df.columns, 'check column (%s) not in df columns' % check_column
            
            df_to_add = df[~df[check_column].isin(db_table[check_column])]
        else:
            df_to_add = df.copy()
            
        if not df_to_add.empty:
            if info_column:
                max_add = df_to_add.loc[:,info_column].max()
                min_add = df_to_add.loc[:,info_column].min()
                extra_info = '%s range added: %s - %s' % (info_column, min_add, max_add)
            else:
                extra_info = ''
            logger_insert.info('Adding %s entries to table %s. %s' % (len(df_to_add), table_name, extra_info))
            sql_str = self.create_sql_str(df_to_add, table_name)
            if table_name == 'colleges':
                logger_insert.info('college insert string: {}'.format(sql_str))

            cursor = self.establish_cursor()
            cursor.execute(sql_str)
        else:
            logger_insert.info('No new entries. No alterations made.')
        
    def str_to_db(self, string, return_results=False):
        """Executes the given string on the connection."""
        cursor = self.establish_cursor()
        cursor.execute(string)
        if return_results:
            return cursor.fetchall()
        
    def read_table(self, table_name=None, columns=None, get_str=None, distinct_only=False):
        "Returns the given table, all columns if none given"
        assert type(columns) == list or columns == None, 'Columns must be given in list, not %s' % type(columns)

        if get_str:
            sql_string = get_str
        else:            
            columns_str = "*" if columns == None else ', '.join(columns) 
            distinct_str = ' DISTINCT ' if distinct_only else ''
            sql_string = "SELECT %s%s FROM %s;" % (distinct_str, columns_str, table_name)

        cursor = self.establish_cursor()

        try:
            cursor.execute(sql_string)
        except sql.errors.ProgrammingError:
            logger_insert.error(sql_string)
            raise
            
        table_columns = cursor.column_names
        table_lines = cursor.fetchall()
        
        return pd.DataFrame(table_lines, columns = table_columns)

    def read_max(self, table_name, column, max_value=True):
        """Returns max or min value for given column"""
        max_min = "MAX" if max_value else "MIN"
        sql_string = "SELECT %s(%s) FROM %s;" % (max_min, column, table_name)
        cursor = self.establish_cursor()
        cursor.execute(sql_string)

        return cursor.fetchall()[0][0] 
    
    def get_mappings(self, table_name, id_label=None, mapping_column=None, inverse=True):
        '''Returns a dict of mappings from column to id.
        When id_label not given, takes primary key as id label
        When mapping_column not given, derived from id label, removing "_id"
        If value is None, no mapping is returned for given item

        Keyword arguments:
        table_name -- the name of the table in db to source the mappings from
        mapping_column -- the column of the given mapping (default None)
        id_label -- the name of the id column desired (default None)
        inverse -- if true will return dict as value:id (default True)
        '''
        mapping_table = self.read_table(table_name)
        # define id_label and mapping_column if both/either are undefined
        # id_label always set to first column, mapping column defined by id_label
        if id_label == None:
            id_label = mapping_table.columns[0]
        if mapping_column == None:
            mapping_column = id_label.replace('_id','')

        assert id_label in mapping_table.columns, 'id label (%s) not in table columns.' % id_label
        assert mapping_column in mapping_table.columns, 'mapping column (%s) not in table columns' % mapping_column

        mapping = mapping_table.set_index(id_label).to_dict()[mapping_column]
        if inverse:
            # swaps keys and values so mapping can be used to return ids
            return {value.strip():key for key,value in mapping.items() if value != None}
        else:
            return mapping

    def apply_mappings(self, df, mapping_table, apply_columns, mapping_column=None, id_label=None, abort=False):
        '''Replaces values with relevant ids, taking a current db table as mapping input

        Keyword arguments:
        df -- the df to alter
        mapping_table -- the table in db to use as mapping input
        apply_columns -- the columns to alter
        mapping_column -- the column in mapping table to use for mapping (default None)
        id_label -- the id in the mapping table to use (default None)
        '''
        def mapping_apply(x, id_mapping):
            '''Apply function to be used with mapping dict. Treats '' and "NULL" as empty values''' 
            stripped_x = str(x).replace("'", "")
            if x == '' or x == "NULL":
                return "NULL"
            elif stripped_x not in id_mapping.keys():
                logger_insert.info('%s not in mapping keys' % x)
                if abort:
                    raise "Mapping Error"
                else:
                    return "NULL"
            else:
                return id_mapping[stripped_x]

        id_mapping = self.get_mappings(mapping_table, id_label, mapping_column)
        df_copy = df.copy().fillna("NULL")
        for column in apply_columns:
            df_copy[column] = df_copy[column].apply(lambda x: mapping_apply(x, id_mapping))
        df_copy = df_copy.rename(columns={key:key+'_id' for key in apply_columns})

        return df_copy