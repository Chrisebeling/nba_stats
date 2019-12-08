import configparser

def update_dbconfig(config_dict, config_file='databaseconfig.ini', section='mysql'):
    '''Update settings in database config file.

    Keyword Arguments:
    config_dict -- A dict of keys and settings, keys must be in current config file
    config_file -- The config file to update (Default 'database_config.ini')
    section -- The section of the config file to update (Default 'mysql')
    '''
    config = configparser.ConfigParser(allow_no_value=True)
    config.read(config_file)
    
    for key in config_dict.keys():
        assert key in config[section], 'key must be in current config. {} not in current config'.format(key)
        config.set(section, key, config_dict[key])
    
    cfg_file = open(config_file,'w')
    config.write(cfg_file)
    cfg_file.close()