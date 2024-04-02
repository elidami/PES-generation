import json
#from .exceptions import ReadParamsError

def read_json(jsonfile):
    """
    Shortcut to easily read a json file.
        
    """
    
    with open(jsonfile, 'r') as f:
        data = json.load(f)  
    return data


def read_default_params(default_file, default_key, dict_params, allow_unknown=False):
    """
    Read the default argument from a JSON file and compare them with the keys 
    of a dictionary. If some data is missing, is substituted with default
    values. It is a generalization of `read_runtask_params`, because it takes a
    dictionary as input and can be used outside Firetasks.

    Parameters
    ----------
    default_file : str
        Path to the JSON file containing the default values.

    default_key : str
        Key of the JSON dict containing the default parameters.

    dict_params : dict
        Dictionary containing the parameters to be updated with defaults. It
        should be a subset of the dictionary extracted from `default_file`.

    Returns
    -------
    dict
        Final dictionary containing the elements of `dict_params` if present,
        else the default values read from JSON file.

    Raises
    ------
    ReadParamsError
        When unknown keys are present ind `dict_params`.

    """

    # Read the JSON file with defaults and extract the corresponding key
    defaults = read_json(default_file)
    defaults = defaults[default_key]
    # Check if there are some unknown parameters
    if not allow_unknown:
        if not set(dict_params.keys()).issubset(set(defaults.keys())):
            raise ValueError(f"The values passed as dictionary params are not known. Allowed values for {default_key}, read in {default_file}: {defaults.keys()}")
        params = {}
    else:
        params = dict_params.copy()
    # Set the parameters, missing parameters are substituted with defaults
    for key, value in defaults.items():
        params[key] = dict_params.get(key, value)

    return params
