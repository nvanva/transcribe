import os

import pandas as pd
from urllib.request import urlretrieve

PATH_SPHINX_STRESS_DICT = "sphinx_stress.txt"
URL_SPHINX_STRESS_DICT = 'https://github.com/zamiron/ru4sphinx/blob/master/text2dict/all_form.txt?raw=true'


def add_stress(phrase_ser, u_ser):
    """
    Adds stress to each word in each row of phrase_ser. Use u_ser as stress dictionary.
    """
    return phrase_ser.str.split(' ').apply(lambda wl: ' '.join((u_ser.get(w, default=w) for w in wl)))


def load_stress_dict():
    """
    Loads stress dict.
    :return: pandas Series with plain tokens as index and tokens with stress as values
    """
    return get_ser(load_stress2())


def load_stress1():
    ud1 = pd.read_csv('./stress.txt.gz',header=None,sep='\t',names=['stok'])
    ud1['tok']=ud1.stok.str.replace("`",'')
    ud1['stok'] = ud1.stok.str.replace(r'(.)`',r'+\1')
    return ud1


def load_stress2():
    URL, PATH = URL_SPHINX_STRESS_DICT, PATH_SPHINX_STRESS_DICT
    if not os.path.exists(PATH_SPHINX_STRESS_DICT):
        print('Downloading stress dictionary from %s ...' % URL)
        urlretrieve(URL, PATH)
        print('Stress dictionary saved to %s' % PATH)

    ud2 = pd.read_csv(PATH_SPHINX_STRESS_DICT, sep=' ', header=None, names=['tok','stok'])
    l = len(ud2)
    assert l > 1.5e6, "Stress dictionary %s has only %d entries, it is probably corrupted. " \
                      "Try download it manually from %s" % (PATH, l, URL)
    return ud2


def get_ser(udf):
    udf = udf.drop_duplicates().groupby('tok').agg(['count','first'])
    mask = udf['stok']['count'] == 1
    print('%f %% tokens have unambiguous stress' % (100*mask.mean()))
    udf = udf[mask] # use only unambiguous entries
    udf.columns = [x[-1] for x in udf.columns.ravel()]
    u_ser = udf['first']
    return u_ser


