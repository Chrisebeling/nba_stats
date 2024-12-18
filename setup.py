from setuptools import find_packages, setup

with open('requirements.txt') as f:
    requirements = f.readlines()

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(name='nba_stats',
      version='0.1.46',
      description='Contains functions used to create, manage and use a database of nba statistics.',
      long_description=readme,
      author='Chris Ebeling',
      author_email='chris.ebeling.93@gmail.com',
      classifiers=['Developer Status::3 - Alpha',
                   'Programming Language :: Python :: 3.7',
                   'Environment :: Win32 (MS Windows)',
                   'License :: OSI Approved :: MIT License'],
      license=license,
      package_data={
                    '': ['*.conf'],
                    },
      include_package_data=True,
      packages=find_packages(),
      entry_points={
          'console_scripts':[
              'scrape_games = nba_stats.scripts.scrape_games:scrape_function']
          },
      install_requires=requirements
      )
