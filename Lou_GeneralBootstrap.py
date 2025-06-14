import numpy as np
import time as tm
import concurrent.futures as cf
import pickle
from collections import defaultdict
from warnings import warn
from scipy import stats

def GenSamples(num,num_sims,seed):
    """
    Generates a 'num' amount of samples and returns the results as a list of all the samples.
    Args:
        num (int): The resulting number of samples
        num_sims (int): number of simulations, determined by the data provided
        seed (int): The seed for randomizing

    Returns:
        list: a list of lists, each containing a sample portion
    """    
    np.random.seed(seed)
    
    list_ = []
    for _ in range(num):
        list_.append(np.random.choice(range(num_sims), size = num_sims,replace = True))
        
    return list_
    
def GenOneSample(num_sims):
    """
    Generates a single list of samples.
    Args:
        num_sims (int): number of simulations, determined by the data provided. (size of array)

    Returns:
        list: a list of samples
    """    
    
    sample = np.random.choice(range(num_sims), size = num_sims,replace = True)
        
    return sample
    
def ObjectSplit(object, num_chunks,list=True):
    """
    The Function takes in an object (number or iterable) and splits it into a given number of chunks.
    Args:
        object (number or iterable): The object to split
        num_chunks (int): The number of chunks to split the object into

    Returns:
        list: a list of chunks
    """ 
    if list:
        object = np.array_split(object, num_chunks)
        return [i for i in object]
    else:        
        chunk_size = object // num_chunks
        remainder = object % num_chunks
        chunks = [chunk_size] * num_chunks
        for i in range(remainder):
            chunks[i] += 1
        return chunks

def MergeResultedList(resultedList):
    """
    The Function takes in the resulted list of numpy arrays or other iterables from the multiprocessing code, 
    and merges them into a singular list with the results from each bootstrap.

    Args:
        resultedList (List): A list containing iterables

    Returns:
        List: a list containing all the results from the bootstraps
    """
    holder = [[] for i in range(len(resultedList[0][0]))]
    for i in resultedList:
        for j in i:
            for k in range(len(j)):
                holder[k].append(j[k])

    finalHolder = [np.array(i) for i in holder]
    return finalHolder

def MergeResultedDict(resultedDict):
    """
    The Function takes in the resulted list of dictionaries from the multiprocessing code, 
    and merges them into a singular dictionary with the results from each bootstrap.

    Args:
        resultedDict (List): A list containing dictionaries

    Returns:
        Dict: a dictionary containing all the results from the bootstraps
    """
    holderDict = defaultdict(list)
    for list_ in resultedDict:
        for dict in list_:
            for key,value in dict.items():
                holderDict[key].append(value)
    
    finalHolder = {}
    for key,value in holderDict.items():
        finalHolder[key] = np.array(value)
    return finalHolder

def ConfInterval(data = np.ndarray,confInt = float,boots = int):
    """
    This function takes in a numpy array and returns the confidence interval. 
    Works on an array of scalers, as well as, multi-dimensional arrays. 
    When working with multi-dimensional arrays, the function transposes 
    the array to compute the confidence interval element wise.

    If the data given is a dictionary, the function returns None

    Args:
        data (ndarray): a numpy array of scalers or vectors
        confInt (float): the confidence
        boots (int): the number of bootstraps that have been conducted

    Warnings:
        This Function Does not Accept Dictionaries. Returning None
    Returns:
        tuple: a tuple of the upper and lower bounds
    """
    confIdx = int((1-confInt)/2 * boots) - 1
    confInterval = []
    if isinstance(data[0], dict):
        warn("This Function Does not Accept Dictionaries. Returning None")
        return None
    else:
        if isinstance(data[0],np.ndarray):
            transposedArray = data.T
            for column in transposedArray:
                mean = np.mean(column)
                std_err = stats.sem(column)  #Standard error
                confidence_interval = stats.t.interval(confInt, len(column)-1, loc=mean, scale=std_err)
                confInterval.append(confidence_interval)
        else:
            confInterval.insert(0,sorted(data)[confIdx])
            confInterval.insert(1,sorted(data)[-(confIdx+1)])
    return tuple(confInterval)    

def innerBootstrap(userFunction,dataSet = np.array,num_boots = int,seed = None,num_sims=int, sampledAxis=None,**kwargs):
    """
    This function takes in a Function and a dataset and does bootstrapping on it

    Args:
        userFunction (Function): a function that takes in an iterable data structure among its other args
        dataSet (numpy array): an iterable data structure. Defaults to np.array.
        num_boots (int): number of bootstraps. Defaults to int.
        samplePortion (list): a list representing a single chunk of the generated samples. Defaults to list.
        sampledAxis (int, optional): The axis the user desires to sample accros. Defaults to None (0).

    Returns:
        Dictionary: contains the results of bootstrap(s)
    """
    np.random.seed(seed)
    sampledAxis = 0 if sampledAxis is None else sampledAxis
    
    if (dataSet.ndim == 1):
        raise ValueError("Please use a multi dimensional array")

    bootResults = [[] for i in range(num_boots)]
    
    for iboot in range(num_boots):
        sample = GenOneSample(num_sims)
        bootedData = np.take(dataSet,sample,axis=sampledAxis)
        bootResults[iboot] = userFunction(bootedData,**kwargs)
    
    return bootResults

def Bootstrap(userFunction,dataSet = np.array,boots = int,seed = None,cores = int, sampledAxis=None,confInt=None,bootData=True,**kwargs):
    """
    The function uses multiprocessing to bootstrap.

    Args:
        userFunction (Function): a function that takes in an iterable data structure among its other args
        dataSet (numpy array): an iterable data structure. Defaults to np.array.
        boots (int): number of bootstraps. Defaults to int.
        seed (int, optional): The seed to randomize with. Defaults to int.
        cores (int, optional): The number of cores the multiprocessing will use. Defaults to int.
        sampledAxis (int, optional): The axis the user desires to sample accros. Defaults to None (0).
        confInt (float, optional): The confidence interval. Defaults to 0.95.

    Returns:
        list: A list containing the function result on whole data, lower and upper confidence bound
    """
    print("started")
    #default values and check for invalid data
    if (dataSet.ndim == 1):
        raise ValueError("Please use a multi dimensional array")

    if (sampledAxis):
        num_sims = dataSet.shape[sampledAxis]
    else:
        num_sims = dataSet.shape[0]
        sampledAxis = 0
    
    print("running the function on everything")
    #Create the main final results delivarable and set seeds
    finalResults = []
    np.random.seed(seed)
    seedList = [np.random.randint(1,(2**31 - 1)) for i in range(cores)] 
    bootSplits = ObjectSplit(boots,cores,False)
    finalResults.insert(0,[userFunction(dataSet,**kwargs)])

    print("start mp")
    list_ = []
    with cf.ProcessPoolExecutor(max_workers = cores) as exe:
        futures = [exe.submit(innerBootstrap,userFunction,dataSet,bootSplits[i],seedList[i],num_sims,sampledAxis,**kwargs) 
                   for i in range(cores)] # we want to pass in the seeds instead, so that the number of seeds and cores match
        for future in cf.as_completed(futures):
            list_.append(future.result())
    
    print("merge the results and compute confidence interval")
    confInt = 0.95 if confInt is None else confInt
    bootResults = []
    confInterval = []
    #assume the results are either a dictionary or a tuple
    if isinstance(list_[0][0], dict):
        bootResults = MergeResultedDict(list_)
        for key,value in bootResults.items():
            confInterval.append(ConfInterval(value,confInt,boots))
        finalResults.append(tuple(confInterval))
        if bootData:
            finalResults.append(bootResults)
    else:
        bootResults = MergeResultedList(list_)
        for array in bootResults:
            confInterval.append(ConfInterval(array,confInt,boots))
        finalResults.append(tuple(confInterval))
        if bootData:
            finalResults.append(bootResults)
            
    return tuple(finalResults)
