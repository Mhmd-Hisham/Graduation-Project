import os
import json
import zipfile
import functools
from typing import Union, List, Dict, Tuple

from IPython.display import display
import numpy as np
import pandas as pd
import requests

print = functools.partial(print, flush=True)

def get_filename_from_headers(url, headers):
    # print(headers.items())
    try:
        filename = headers["content-disposition"]
        idx = filename.find("=")
        filename = filename[idx+1:]
    except Exception as E:
        filename = url.split("/")[-1]

    return filename

def download_file(url, dest, override=False, create_dirs=True):
    filename = requests.head(url, allow_redirects=True).url.split("/")[-1]

    if create_dirs and not os.path.exists(dest):
        os.makedirs(dest)

    # filename = get_filename_from_headers(url, res.headers)
    filepath = os.path.join(dest, filename)

    if (not os.path.exists(filepath)) or (override):
        if (override):
            print(f"File '{filename}' exists! Overriding.. ", end="")

        else:
            print(f"Downloading '{filename}'.. ", end="")

            with open(filepath, "wb+") as fh:
                res = requests.get(url)

                if (res.status_code != 200):
                    print(f"Error! Couldn't download from this url={url}")
                    return

                fh.write(res.content)
            print("Done!")
    else:
        print(f"File '{filename}' exists! Enable override to override it.")
    return filename

def download_from_list(urls, dest, override=False, create_dirs=True):
    files = []
    if (isinstance(urls, str)):
        urls = parse_urls(urls)
        
    for url in urls:
        filename = download_file(url, dest, override)
        files.append(filename)
    
    return files

def parse_urls(urls):
    return urls.strip().split()

def unzip(path_to_zip, target_dir, extract_directly=False):
    print(f"Unzipping '{path_to_zip}'.. ", end="", flush=True)

    # get the absolute paths
    path_to_zip = os.path.abspath(path_to_zip)
    target_dir = os.path.abspath(target_dir)
    parent_dir = os.path.splitext(path_to_zip)[0].split(os.sep)[-1] + os.sep

    # target dir 
    with zipfile.ZipFile(path_to_zip, "r") as zip_ref:
        # create a folder with the same name as the zip file
        # only if requested
        if not extract_directly:
            names = zip_ref.namelist()
            
            if names[0] != parent_dir:
                target_dir = os.path.join(target_dir, parent_dir)

        zip_ref.extractall(target_dir)
    
    print("Done!")

    return os.path.join(target_dir, parent_dir)

def unrar(path_to_rar, target_dir):
    import patoolib
    print(f"Extracting '{path_to_rar}'.. ", end="", flush=True)
    parent_dir = os.path.splitext(path_to_rar)[0]
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    
    target_dir = os.path.join(target_dir, parent_dir)
    patoolib.extract_archive(path_to_rar, outdir=target_dir)    
    print("Done!")

    return target_dir

def read_jsonl(path_to_jsonl_file):
    json_objs = []

    # load the file
    with open(path_to_jsonl_file, "r") as fh:

        # split the lines
        lines = fh.read().splitlines()

        # parse each line as a json object
        for line in lines:
            json_objs.append(json.loads(line))

    # return a list of json objects
    return json_objs

def read_json(path_to_json_file):
    with open(path_to_json_file, "r") as fh:
        data = json.load(fh)
    
    return data

def jsonl_to_df(json_objs):
    """ 
        Takes a list of json objects (from a jsonl file)
        and converts it to a pandas dataframe.
    """

    # get a set of all avalible dictionary keys in the json objects
    keys = set()
    for row in json_objs:
        keys |= set(list(row.keys()))

    # convert to a list to follow a specific order (but a random order)
    keys = list(keys)

    # extract the data from eact dictionary and fill nulls
    list_of_data = [[d.get(key, np.nan) for key in keys] for d in json_objs]

    # create the dataframe and set the column names
    df = pd.DataFrame(data=list_of_data)
    df.columns = keys

    return df

def save_as_json(obj, filename, destination):
    with open(os.path.join(destination, filename), "w+") as fh:
        json.dump(obj, fh)

def format_number(number: float) -> str:
    """
        Formats numbers in a nice presentable way.
        Mainly adds a K, M, or B prefix to large numbers.

        Parameters
        ----------
        number: the number you want to format.

        Returns
        -------
        A string containing the formatted number.
    """
    formatter = lambda v: int(v) if v < 1e3 else \
                          f"{v//1e3:,.0f}K" if v < 1e6 else \
                          f"{v/1e6:,.2f}M" if v < 1e9 else \
                          f"{v/1e9:,.2f}B"
    
    return formatter(number)

def pprint_df(df: pd.DataFrame, columns: List[str]=None, display: bool=False) -> pd.DataFrame:
    """
        Pretty print a dataframe in a jupyter notebook.
        Re-formats numbers in each column and adds a prefix (K, M, B) to make it easier to read large numbers.

        Parameters
        ----------
        df : the pandas Dataframe you want to print prettily.
        columns : the list of 'numeric' columns you want print nicely. Uses all numeric columns if no columns were provided.
        display : whether to display the dataframe on Jupter or not

        Returns
        -------
        Returns a copy of the dataframe formatted nicely. The numeric columns will be converted to str columns.
    """

    # make a copy of the dataframe
    df = df.copy()

    # select numeric columns if no columns were provided
    columns = columns if columns else df.select_dtypes([np.number]).columns

    # format the columns
    for col in columns:
        df[col] = df[col].apply(format_number)

    # display in Jupyter
    if display:
        display(df)

    # return the formatted dataframe
    return df

def remove_outliers(obj: Union[pd.DataFrame, pd.Series, np.array], column: str=None, std_range: float=3):
    """
        Removes outliers from the given series/dataframe using the z-score method.

        Parameters
        ----------
        obj : the object you want to remove outliers from. Could be a Pandas series, dataframe, or a numpy array.
        std_range: how many standard deviations should the points be from the mean. Non-outliers: -std_range < x < std_range
        column: Specifies the numerical column to compute the z-score upon. Only needed when obj is a dataframe.

        Returns
        -------
        Returns a boolean numpy array (mask) with outliers set to False.
    """
    array = None
    if isinstance(obj, pd.DataFrame):
        array = np.array(obj[column])

    elif isinstance(obj, pd.Series):
        array = np.array(obj.values)

    mu = np.mean(array)
    sigma = np.std(array)
    z_score = (array - mu)/sigma

    return (-std_range < z_score)&(z_score < std_range)
