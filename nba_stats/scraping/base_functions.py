from bs4 import BeautifulSoup
import urllib
import pandas as pd

def get_soup(url_str, headers=None, timeout=None):
    '''Returns a soup object of the given url. Uses mozilla headers by default.
    
    Keyword arguments:
    url_str -- the url of the desired webpage
    headers -- override headers for soup object (default None)
    '''
    if headers == None:
        headers = {'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"}
    try:
        soup = BeautifulSoup(urllib.request.urlopen(urllib.request.Request(url_str, headers=headers), timeout=timeout),'html.parser')
    except urllib.error.HTTPError as err:
        if err.code == 404:
            return None
        else:
            raise err
    
    return soup

def get_bref_soup(bref_link, headers=None):
    '''Returns a soup object given the bref link.
    
    Keyword arguments:
    bref_link -- The link for the bref page to be targeted
    headers -- override headers to be used when getting the soup object (default None)
    '''
    bref_html = 'https://www.basketball-reference.com'
    full_html = bref_html + bref_link
    return get_soup(full_html, headers)

def get_bref_tables(soup, desired_tables=True, href_category=None):
    '''Returns tables from the given soup of a bref url.
    Should work on most bref pages. Index is data-append-csv if this exists in header, othereise the header.
    '''
    tables = {}
    
    for div in soup.find_all('div',{'id':desired_tables}):
        idx = 0
        title = div['id']
        for table in div.find_all('tbody'):
            table_dict = {}
            for row in table.find_all('tr'):
                header = row.find('th')

                if 'data-append-csv' in header.attrs:
                    table_dict[idx] = {'bref':header['data-append-csv'].strip()}
                if 'data-stat' in header.attrs:
                    category = header['data-stat']
                    if idx in table_dict.keys():
                        table_dict[idx][category] = header.text.strip()
                    else:
                        table_dict[idx] = {header['data-stat']:header.text.strip()}
                    if category == href_category and header.find('a'):
                        table_dict[idx]['bref'] = header.find('a')['href']
                for column in row.find_all('td'):
                    category = column['data-stat']
                    table_dict[idx][category] = column.text
                    if category == href_category and column.find('a'):
                        table_dict[idx]['bref'] = column.find('a')['href']
                
                # increment index for next row
                idx += 1

        tables[title] = pd.DataFrame(table_dict).transpose()

    return tables

def get_table(soup, desired_table, href_column=None):
    '''Returns the table of a given bref_soup. Works well for loosely defined tables.
    Where columns are missing in a row, fills with empty columns at the end of row.

    Keyword Arguments
    soup -- The soup to extract the table from
    desired_table -- The name of the table to be extracted
    href_column -- The column to extract an href from (default None)
    '''
    table_rows=[]

    table = soup.find('div',{'id':desired_table})
    header_row = table.find('thead').find_all('tr', class_=lambda x: x != 'over_header')[0]
    rows = table.find('tbody').find_all('tr', class_=lambda x: x != 'thead')

    header = [x.text for x in header_row.find_all()]
    exp_cols = len(header)

    for row in rows:
        row_contents = row.find_all(['td','th'])
        row_data = [x.text for x in row_contents]
        missing_cols = exp_cols - len(row_data)
        row_data += ['']*missing_cols
        if href_column:
            row_data.append(row_contents[href_column].find('a')['href'])

        table_rows.append(row_data)

    return pd.DataFrame(table_rows, columns=header+['href'])