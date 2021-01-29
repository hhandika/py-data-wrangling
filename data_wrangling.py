"""
A module use to work on specimen datasets.

The data manipulation is done using pandas. 

Functions:
1. Clean duplicates specimens from Specify datasets. 
Useful for counting the numbers of specimens for a given
ranges of catalog numbers. Specify by its nature will return 
several numbers of lines based on the tissue types. This function
can quickly clean it up.

2. 
"""

import os
from glob import glob

import pandas as pd


def clean_duplicates(df,params):
    """Clean specify duplicates specimens.
    Keep the first row.

    Args:
        dataframe ([type]): [description]
    """
    df = df.drop_duplicates([params], keep='first')
    return df

def clean_column_names(df: pd.DataFrame):
    """
    Convert Specify Darwin Core column names to human readable names.
    """
    df = df.rename(columns={
        '1.collectionobject.catalogNumber': 'CatNo',
        '1.collectionobject.fieldNumber': 'FieldNo',
        '1,10,30-collectors,5.agent.lastName': 'Collector',
        '1,9-determinations,4.taxon.Order': 'Order',
        '1,9-determinations,4.taxon.Family': 'Family',
        '1,9-determinations,4.taxon.Genus': 'Genus',
        '1,9-determinations,4.taxon.Species': 'Species',
        '1,10,2,3.geography.Country': 'Country',
        '1,10,2,3.geography.State': 'StateProvince',
        '1,10,2,3.geography.County': 'CountyDistrict',
        '1,10,2.locality.localityName': 'SpecificLocality',
        '1,10,2.locality.latitude1': 'Latitude',
        '1,10,2.locality.longitude1': 'Longitude',
        '1,10,2.locality.verbatimElevation': 'Elevation',
        '1,10,2.locality.originalElevationUnit': 'Unit',
        '1,63-preparations,65.preptype.name': 'PrepType',
        '1,63-preparations.preparation.text1': 'TissueType',
        '1,63-preparations.preparation.text2': 'Preservation',
        '1,63-preparations.preparation.storageLocation': 'StorageLocation',
        '1,93.collectionobjectattribute.text1': 'Sex',
        '1,93.collectionobjectattribute.text7': 'TotalLength',
        '1,93.collectionobjectattribute.text8': 'TailLength',
        '1,93.collectionobjectattribute.text9': 'HindFoot',
        '1,93.collectionobjectattribute.text10': 'EarLength',
        '1,93.collectionobjectattribute.text2': 'Weight',
        '1,10.collectingevent.startDate': 'StartDate',
        '1,93.collectionobjectattribute.text4': 'Stage',
        '1.collectionobject.remarks': 'Remarks',
        '1.collectionobject.altCatalogNumber': 'AltCatNo'})
    return df

def clean_whitespace(df, columns):
    """
    Clean leading/trailing whitespace from specific columns

    Args:
        df (table): pandas dataframe
        columns (list): column labels

    Returns:
        dataframe    
    """
    df[columns] = df[columns].apply(lambda x: x.str.strip())
    return df

def trimmed_df_whitespace(df):
    """Trimmed white space for a whole dataframe.

    Args:
        df (table): pandas table to clean.

    Returns:
        table: pandas data frame.
    """
    df = df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)
    return df
    
def extract_data(df_database, df_filters, column_names):
    """
    Extract data from one database to match with the other database.
    The function requires the same column names for both dataframe.

    Args:
        df_database (pandas table): [description]
        df_filters (pandas table): [description]
        column_names (string): [description]
    """
    filters = df_database[column_names].isin(df_filters[column_names])
    df_database = df_database[filters]
    
    return df_database

def combine_dataframes(filepath, output_path):
    """Combine multiple dataframes into one.
    The function assume the same column names for each dataframe.

    Args:
        filepath (string): the file's path with matching filenames 
        using wildcard
        output_path (string): the final file's path and name

    Returns:
        csv: save into output path folder. 
    """
    filenames = glob(filepath)
    combined_df = []
    for df in filenames:
        dataframes = pd.read_csv(df)
        combined_df.append(dataframes)
    results = pd.concat(combined_df, axis=0)
    return results.to_csv(output_path, index=0)

def concat_column_values(df, first_column, second_column, new_column_names):
    
    df[new_column_names] = '(' + df[first_column].map(str) + ',' + df[second_column].map(str) + ')'

    return df

def convert_excel_to_csv(filepath):
    """Batch converting excel files to csv.

    Args:
        filepath (string): Use wildcard to match filenames.
    """
    filenames = glob(filepath)
    for excel in filenames:
        out_filenames = excel.replace('.xlsx','.csv')
        dataframe = pd.read_excel(excel)
        dataframe.to_csv(out_filenames, index=False)
    print("Done converting to csv!")

def convert_windows_path(file_path: str) -> str :
    """Convert windows path to unix path

    Args:
        file_path (str): Windows path

    Returns:
        str: Unix path
    """
    file_path = file_path.replace("\\", "/")

    return file_path

def sort_preptype(df):
    """[summary]

    Args:
        df ([type]): [description]
    """
    #Make the preptype categorical
    prep_type = [
        'Skin',
        'Alcohol',
        'Skull',
        'skeleton',
        'Tissue',
        'Intestine',
        'Small intestine',
        'Colon',
        'GI Tract',
        'Cecum',
        'Glands',
        'Testicle',
        'Embryo'
        ]
    df['PrepType'] = pd.Categorical(df['PrepType'], prep_type)
    df = df.sort_values(by=['CatNo','PrepType'])

    return df


def get_column_names(df):
    column_names = []
    for column in df.columns:
        column_names.append(column)
    return column_names

def filter_results(df, column_name, params):
    """
    Filtered the data based on specific values. 

    Args:
        df (table): pandas table 
        column_name (string): pandas column names
        params (list): value names to filters
    """
    filters = df[column_name].isin(params)
    filtered_results = df[filters].sort_values(by=['Genus','Species'])
    return filtered_results

def count_specimen_groups(df, params):
    """
    Count the number of specimens based on pre-defined groups

    Args:
        df (table): [description]
    """
    df = df.fillna('No Data')
    species_count = df.groupby(params).count()

    #Use field number as unique values for counting
    species_count = species_count.filter(['CatNo']) 
    species_count = species_count.rename(columns={'CatNo': 'Counts'})

    return species_count.reset_index()

def merge_dataframes(df1, df2, df1_column_names, column_keys):
    """
    Merge two dataframes using a column value as a key.

    Args:
        df1 (table): [description]
        df2 (table): [description]
        column_keys (string): [description]
    """
    df1 = df1[df1_column_names]
    df1[column_keys] = df1[column_keys].astype(int)
    df1[column_keys] = df1[column_keys].astype(int)
    df_merge = pd.merge(df1, df2, on=column_keys)

    return df_merge

def open_csv(path: str, filenames: str) -> pd.DataFrame:
    """Open csv file based on specified 
    path and filenames. Useful for deeply
    nested folders.

    Args:
        path (string): path locations
        filenames (string): filenames with the extension.

    Returns:
        [type]: [description]
    """
    csv_file = path + '/' + filenames
    df = pd.read_csv(csv_file)
    return df

def save_csv(df, parent_path, filenames):
    """Save pandas's dataframe to csv.
    The function check if the path exists.
    If not, it will create the defined path.
    
    Args:
        df ([type]): [description]
        parent_path ([type]): [description]
        filenames ([type]): [description]

    Returns:
        [type]: [description]
    """
    file_path = parent_path + '/' + filenames

    try:
        df.to_csv(file_path, index=False)
        print('File saved!')

    except FileNotFoundError:
        os.mkdir(parent_path)
        print(f'A new folder is created. File path: {parent_path}/')

        df.to_csv(file_path, index=False)
        print(f'File is saved in {file_path}.')
        

def save_with_index(df, filename):

    return df.to_csv('cleaned_data/' + filename)

def split_columns(df, separator, new_columns, column_names):
    """
    Split column in data frame into two.

    Args:
        df (pandas table): 
        separator (string): values separator to split
        new_columns (list): names of the new columns
        column_names (string): names of the column to split

    Returns:
        table: new data frame with the column splited into its values
    """
    df[new_columns] = df[column_names].str.split(separator, expand=True)
    return df

class MuseumNumbers():
    """
    A class to get the museum number from a dataset.
    """
    def __init__(self, df_origin, df_result):
        self.df_origin = df_origin
        self.df_result = df_result

    def filter_database(self):
        """
        df1 is the database
        df2 is the resulting data
        """
        #Extract LSUMZ
        df_origin = self.df_origin[['ColInitial', 'CatNo']]
        #Use field no as a key and match the key names of the two database.
        filters = self.df_origin['ColInitial'].isin(self.df_result['ColInitial'])
        df_origin = df_origin[filters]

        return df_origin

    def merge_database(self, df_origin):
        df_merge = pd.merge(df_origin, self.df_result)

        return df_merge
    
    def get_numbers(self):
        filter_df = self.filter_database()
        merge_df = self.merge_database(filter_df)
        return merge_df
    
    def save_results(self, path, file_name):
        final_df = self.get_numbers()
        return save_csv(final_df, path, file_name)

class FieldNumbers():
    def __init__(self, df, names, initials):
        self.df = df
        self.names = names
        self.initials = initials
    
    def add_initial_columns(self):
        """
        Add initial column using collector names.

        Args:
            names (list): list of collector names
            initials (list): list of collector initials

        Returns:
            dataframe with collector initials added at the far
            right of the table.
        """
        
        self.df['Initials'] = self.df['Collector'].replace(self.names, self.initials)

        return self.df

    def merge_initials(self):
        """
        

        Args:
            names ([type]): [description]
            initials ([type]): [description]

        Returns:
            [type]: [description]
        """

        df_result = self.add_initial_columns()

        df_result['FieldNo'] = df_result['FieldNo'].astype(str)
        df_result['ColInitial'] = df_result['Initials'] + df_result['FieldNo']
        df_result = df_result.drop('Initials', axis = 1)

        return df_result