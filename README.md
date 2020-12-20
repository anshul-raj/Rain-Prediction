# Rain-Prediction
Weather forecasting has been in existence since 1835.
But, modern forecasting methods using sensors and
physical techniques demands a lot of resources. In 2009,
the US spent approximately $5.1 billion on weather
forecasting. Hence, our aim is to cut this cost by training a
Machine learning model that will use the previous day’s
data to predict tomorrow’s downpour with appreciable
accuracy.

## Data used
Observations were drawn from numerous weather stations. The daily observations are available from http://www.bom.gov.au/climate/data.
Data source: 
http://www.bom.gov.au/climate/dwo/ <br /> 
http://www.bom.gov.au/climate/data. <br />

### weatherAUS.csv
contains 24 features and 1 output column.
for different models we further make 2 more dataset from the existing dataset.

### dataset_maker.py
 After some preprocessing, This file has a variable numdays, Using this variable it find numdays number of continuous days of data and clups them into a single row giving us data for numdays number of continuous days into a single row, this is repeated for all such continuous dates and finally returns a csv file for all the continuous days.

### balanced_dataset.py
 After some preprocessing, This file first seperates the data into two parts; one with RainTomorrow : 1 and RainTomorrow : 0. Seeing the skewness of the data it takes all the RainTomorrow : 1 data and randomly sample equal number of RainTomorrow : 0, Then concats them into a single data frame and then returnes. Now this dataframe as no skewness.

# EDA and basic models
  In this Notebook we performed EDA on the data, and then applied some simple regression and decision based classifiers to get a gist for the data.

### Predictions
We are using above mentioned data to predict the Rain a week in advance. We have done this in 2 ways,

1. Using a data from a single day and predictiong rain for the comming week
  here we modifiy the dataset using dataset_maker keeping numdays = 7, splitting 1 day data as feature vector and the rest 6 days rain as target values. Then appiles RandomForest classifier on it. This gives us 6 models for each day.
2. Using a data from a Week and predictiong rain for the comming week
  here we modifiy the dataset using dataset_maker keeping numdays = 14, splitting 7 day data as feature vector and the rest 7 days rain as target values. Then appiles RandomForest classifier on it. This gives us 7 models for each day.

We can see that using more data gives us more positive results, although both lack in precision, which brings us to the next part.

### Improving precision
For Improving the precision we are using a modifies training dataset which can be generated using balanced_dataset.py. This will return a dataset with no skewness wrt RainTomorrow.

Now we split the data into train and split and applied *sklearn.ensemble.GradientBoostingClassifier* Which finally gives us a reasonable precision.

### How to Use ?

For using any model just use the following code

 ```
  import pickle
  model = pickle.load(open("Model-Name","rb"))
 ```

To predict use 

 ```
  predictions = model.predict("Your Values")
 ```

If you need the probability distribution for both the classes use

 ```
  predictions = model.predict_prob("Your Values")
 ```
This will return a nx2 matrix which will contain probability for both the classes.

If you want to make any modifications to the model itself, you can download its respective notebook and make the changes accordingly.

### Contact
Anshul Raj (anshul18020@iiitd.ac.in)<br /> 
Siddhant Yadav (siddhant18196@iiitd.ac.in) <br />
Yash Vats (siddhant18204@iiitd.ac.in)

<b>Thank You : )<b> 
