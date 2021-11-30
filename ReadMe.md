The two most popular techniques for scaling numerical data prior to modeling are **normalization** and **standardization**.

**Normalization** scales each input variable separately to the range 0-1, which is the range for floating-point values where we have the most precision.

**Standardization** scales each input variable separately by subtracting the mean (called centering) and dividing by the standard deviation to shift the distribution to have a mean of zero and a standard deviation of one.

# Standard Deviation and variance

![](https://i.imgur.com/zO0MQki.png)

## using standard deviation

![](https://i.imgur.com/dVxXrkI.png)

## standard deviation (normal, extra large, extra small)

![](https://i.imgur.com/3cfSC9b.png)

## standard deviatin bigger when difference are big

![](https://i.imgur.com/IfHYmM2.png)

# Purpose of machine learning models

> Machine learning models learn a mapping from input variables to an output variable.


# Data Normalization

> Normalization is a rescaling of the data from the original range so that all values are within the new range of 0 and 1.

A value is normalized as follows:

   y = (x – min) / (max – min)
   
For example, for a dataset, we could guesstimate the min and max observable values as 30 and -10. We can then normalize any value, like 18.8, as follows:

  y = (x – min) / (max – min)
  y = (18.8 – (-10)) / (30 – (-10))
  y = 28.8 / 40
  y = 0.72
  
You can normalize your dataset using the `scikit-learn` object `MinMaxScaler`


## Example of normalization

     # example of a normalization
     from numpy import asarray
     from sklearn.preprocessing import MinMaxScaler
     # define data
     data = asarray([[100, 0.001],
             [8, 0.05],
             [50, 0.005],
             [88, 0.07],
             [4, 0.1]])
     print(data)
     # define min max scaler
     scaler = MinMaxScaler()
     # transform data
     scaled = scaler.fit_transform(data)
     print(scaled)
     
 ## Ouput


      [[1.0e+02 1.0e-03]
       [8.0e+00 5.0e-02]
       [5.0e+01 5.0e-03]
       [8.8e+01 7.0e-02]
       [4.0e+00 1.0e-01]]
      [[1.         0.        ]
       [0.04166667 0.49494949]
       [0.47916667 0.04040404]
       [0.875      0.6969697 ]
       [0.         1.        ]]
       
# Data Standardization

> Standardizing a dataset involves rescaling the distribution of values so that the mean of observed values is 0 and the standard deviation is 1.

> Another […] technique is to calculate the statistical mean and standard deviation of the attribute values, subtract the mean from each value, and divide the result by the standard deviation. This process is called standardizing a statistical variable and results in a set of values whose mean is zero and standard deviation is one.

> Subtracting the mean from the data is called centering, whereas dividing by the standard deviation is called scaling. As such, the method is sometime called “center scaling“.

A value is standardized as follows:

      y = (x – mean) / standard_deviation
Where the mean is calculated as:

      mean = sum(x) / count(x)
And the standard_deviation is calculated as:

      standard_deviation = sqrt( sum( (x – mean)^2 ) / count(x))
We can guesstimate a mean of 10.0 and a standard deviation of about 5.0. Using these values, we can standardize the first value of 20.7 as follows:

      y = (x – mean) / standard_deviation
      y = (20.7 – 10) / 5
      y = (10.7) / 5
      y = 2.14
The mean and standard deviation estimates of a dataset can be more robust to new data than the minimum and maximum.

You can standardize your dataset using the scikit-learn object StandardScaler.

## Example

      # example of a standardization
      from numpy import asarray
      from sklearn.preprocessing import StandardScaler
      # define data
      data = asarray([[100, 0.001],
                  [8, 0.05],
                  [50, 0.005],
                  [88, 0.07],
                  [4, 0.1]])
      print(data)
      # define standard scaler
      scaler = StandardScaler()
      # transform data
      scaled = scaler.fit_transform(data)
      print(scaled)
      
We can see that the mean value in each column is assigned a value of 0.0 if present and the values are centered around 0.0 with values both positive and negative.

      [[1.0e+02 1.0e-03]
       [8.0e+00 5.0e-02]
       [5.0e+01 5.0e-03]
       [8.8e+01 7.0e-02]
       [4.0e+00 1.0e-01]]
      [[ 1.26398112 -1.16389967]
       [-1.06174414  0.12639634]
       [ 0.         -1.05856939]
       [ 0.96062565  0.65304778]
       [-1.16286263  1.44302493]]
       
       
# Q. Should I Normalize or Standardize?

Predictive modeling problems can be complex, and it may not be clear how to best scale input data.

If in doubt, normalize the input sequence. If you have the resources, explore modeling with the raw data, standardized data, and normalized data and see if there is a beneficial difference in the performance of the resulting model.


# preprocessing data

The following subjects will be handled:
      Missing values
      Polynomial features
      Categorical features
      Numerical features
      Custom transformations
      Feature scaling
      Normalization
      
Note that step three and four can be performed interchangeable, since these transformations should be executed independently of each other.


## missing values

      import numpy as np
      import pandas as pd
      X = pd.DataFrame(
          np.array([5,7,8, np.NaN, np.NaN, np.NaN, -5,
                    0,25,999,1,-1, np.NaN, 0, np.NaN])\
                    .reshape((5,3)))
      X.columns = ['f1', 'f2', 'f3'] #feature 1, feature 2, feature 3
      
      
![](https://i.imgur.com/N0s96b8.png)


