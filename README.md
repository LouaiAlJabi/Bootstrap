# Bootstrap

This repository contains Python code for performing bootstrap resampling and statistical analysis using multiprocessing. The code is designed to handle multi-dimensional data structures and utilizes parallel processing to efficiently compute results.

**Installation:**
**Dependancies:**
numpy (np)
time (tm)
concurrent.futures (cf)
pickle
collections.defaultdict (imported as defaultdict)
warnings.warn (from warnings)
sklearn.linear_model.LinearRegression (imported as LinearRegression from sklearn.linear_model)
scipy.stats (imported as stats from scipy)

**Functions:**
* GenSamples(num, num_sims, seed)

    Generates a specified number of samples based on the provided seed.
    Args:
        num (int): Number of samples to generate.
        num_sims (int): Number of simulations.
        seed (int): Seed for randomization.
    Returns:
        List of lists containing the generated samples.

* GenOneSample(num_sims)

    Generates a single sample based on the number of simulations.
    Args:
        num_sims (int): Number of simulations.
    Returns:
        List representing the generated sample.

* ObjectSplit(object, num_chunks, list=True)

    Splits an object (number or iterable) into a specified number of chunks.
    Args:
        object (number or iterable): Object to split.
        num_chunks (int): Number of chunks.
        list (bool, optional): If True, returns a list of chunks; otherwise, returns chunk sizes.
    Returns:
        List of chunks or chunk sizes.

* MergeResultedList(resultedList)

    Merges a list of numpy arrays or other iterables into a single list.
    Args:
        resultedList (List): List containing iterables.
    Returns:
        List containing all results from the iterables.

* MergeResultedDict(resultedDict)

    Merges a list of dictionaries into a single dictionary.
    Args:
        resultedDict (List): List containing dictionaries.
    Returns:
        Dictionary containing all results from the dictionaries.

* ConfInterval(data, confInt, boots)

    Computes the confidence interval for a numpy array of scalars or vectors.
    Args:
        data (ndarray): Numpy array of scalars or vectors.
        confInt (float): Confidence interval.
        boots (int): Number of bootstraps conducted.
    Returns:
        Tuple of upper and lower bounds of the confidence interval.

* innerBootstrap(userFunction, dataSet, num_boots, seed, num_sims, sampledAxis, **kwargs)

    Performs bootstrapping on a dataset using a specified function.
    Args:
        userFunction (Function): Function that operates on an iterable data structure.
        dataSet (numpy array): Dataset to perform bootstrapping on.
        num_boots (int): Number of bootstraps.
        seed (int, optional): Seed for randomization.
        num_sims (int): Number of simulations.
        sampledAxis (int, optional): Axis to sample across.
    Returns:
        Dictionary containing the results of bootstraps.

* Bootstrap(userFunction, dataSet, boots, seed, cores, sampledAxis, confInt, bootData, **kwargs)

    Utilizes multiprocessing to perform bootstrapping on a dataset.
    Args:
        userFunction (Function): Function that operates on an iterable data structure.
        dataSet (numpy array): Dataset to perform bootstrapping on.
        boots (int): Number of bootstraps.
        seed (int, optional): Seed for randomization.
        cores (int, optional): Number of CPU cores to use.
        sampledAxis (int, optional): Axis to sample across.
        confInt (float, optional): Confidence interval.
        bootData (bool): Whether to include bootstrapped data in results.
    Returns:
        Tuple containing the function result on the whole data, lower and upper confidence bounds.

  **Example Usage:**

  ```
  import Lou_GeneralBootstrap as lbs
  import numpy as np

  def compute_mean(data):
    return np.mean(data)

  data = np.random.normal(loc=0, scale=1, size=1000)

  results = bs.Bootstrap(userFunction=compute_mean, dataSet=data, boots=1000, cores=4)

  print("Bootstrap Means:", results[0])
  print("Confidence Interval (95%):", results[1])
  ```
