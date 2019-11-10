import pandas as pd

def export_txt(string, file_name, append=True):
    if append:
        open_type = 'a'
    else:
        open_type = 'w'
    with open(file_name, open_type) as f:
        f.write(string)
        
def create_schema_str(df, table_name):
        lengths = {}
        headers = list(df.columns)
        floats, ints = [], []
        for column in headers:
            if type(df[column][0]) == str:
                len_series = df[column].str.len()
                lengths[column] = (len_series.min(), len_series.max(), float(len_series.mean()))
            elif type(df[column][0]) == float:
                floats.append(column)
            elif type(df[column][0]) == int:
                ints.append(column)

        string = 'CREATE TABLE %s (\n' % table_name
        string += '\t%s_id INTEGER NOT NULL AUTO_INCREMENT,\n' % table_name[:-1]
        for header in headers:
            if header in lengths.keys():
                string += '\t%s %s, %s, %.1f,\n' % (header, lengths[header][0], lengths[header][1], lengths[header][2])
            elif header in ints:
                string += '\t%s INTEGER,\n' % header
            elif header in floats:
                string += '\t%s FLOAT,\n' % header
            else:
                string += '\t%s\t,\n' % header
        string += '\tPRIMARY KEY (%s_id)\n) \n\n' % table_name[:-1]

        return string