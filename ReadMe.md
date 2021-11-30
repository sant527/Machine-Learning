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
