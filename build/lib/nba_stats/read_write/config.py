import configparser
import pkg_resources
import os

def set_dbconfig(config_dict, config_file='databaseconfig.conf', section='mysql'):
    '''Update settings in database config file.

    Keyword Arguments:
    config_dict -- A dict of keys and settings, keys must be in current config file
    config_file -- The config file to update (Default 'database_config.ini')
    section -- The section of the config file to update (Default 'mysql')
    '''
    path = pkg_resources.resource_filename(
            __name__,
            os.path.join(os.pardir, 'resources', config_file)
        )
    print(path)

    config = configparser.ConfigParser(allow_no_value=True)
    config.read(path)
    
    for key in config_dict.keys():
        assert key in config[section], 'key must be in current config. {} not in current config'.format(key)
        config.set(section, key, config_dict[key])
    
    cfg_file = open(path,'w')
    config.write(cfg_file)
    cfg_file.close()

def get_dbconfig(config_file='databaseconfig.conf', section='mysql'):
    path = pkg_resources.resource_filename(
            __name__,
            os.path.join(os.pardir, 'resources', config_file)
        )

    config = configparser.ConfigParser(allow_no_value=True)
    config.read(path)
    print(path)

    for option in config.options(section):
        print("{}:::{}".format(option,
                               config.get(section, option)))