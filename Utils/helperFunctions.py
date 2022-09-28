import os
import json
import zipfile
import functools

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


def pprint_df(df, columns=None):
    """
        Pretty print a dataframe in a jupyter notebook.
        Re-formats numbers in each column and adds a prefix (K, M, B) to make it easier to read large numbers.
    """
    formatter = lambda v: int(v) if v < 1e3 else \
                          f"{v//1e3:,.0f}K" if v < 1e6 else \
                          f"{v/1e6:,.2f}M" if v < 1e9 else \
                          f"{v/1e9:,.2f}B"
    df = df.copy()
    columns = columns if columns else df.columns
    for col in columns:
        df[col] = df[col].apply(formatter)
    display(df)