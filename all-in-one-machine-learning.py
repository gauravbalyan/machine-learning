# Import helpful libraries
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
#This imports the OrdinalEncoder class from the sklearn.preprocessing module, which is used to convert categorical data into numerical data.
from sklearn.preprocessing import OrdinalEncoder 
from sklearn.preprocessing import OneHotEncoder



# Load the data, and separate the target
iowa_file_path = './input/melb_data_with_id.csv'
#creating dateset from csv
home_data = pd.read_csv(iowa_file_path)


# print the list of columns in the dataset to find the name of the prediction target
columns= home_data.columns

#print(columns)

# The Melbourne data has some missing values (some houses for which some variables weren't recorded.)  
# Your Iowa data doesn't have missing values in the columns you use. 
# So we will take the simplest option for now, and drop houses from our data. 
# dropna drops missing values (think of na as "not available")
#home_data = home_data.dropna(axis=0)

#Selecting The Prediction Target
#Choosing "Features"
# Create X (After completing the exercise, you can return to modify this line!)
y = home_data.Price
#features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
                        'YearBuilt', 'Lattitude', 'Longtitude']

# Select columns corresponding to features, and preview the data, By convention, this data is called X.
X = home_data[features]

#Let's quickly review the data we'll be using to predict house prices using the describe method and the head method, which shows the top few rows.
descibeX=X.describe()
#print('---describing X--')
#print(descibeX)

#Visually checking your data with these commands is an important part of a data scientist's job. You'll frequently find surprises in the dataset that deserve further inspection.
headdata=X.head()
#print('---printing top rows using head--')
#print(headdata)


#----------Stage 2: Building the Model -------------------
#Define: What type of model will it be? A decision tree? Some other type of model? Some other parameters of the model type are specified too.
#Fit: Capture patterns from provided data. This is the heart of modeling.
#Predict: Just what it sounds like
#Evaluate: Determine how accurate the model's predictions are.

 #Define model. Specify a number for random_state to ensure same results each run
 #Many machine learning models allow some randomness in model training. Specifying a number for random_state ensures you get the same results in each run. 
melbourne_model = DecisionTreeRegressor(random_state=1)

# Fit model
melbourne_model.fit(X, y)

#We now have a fitted model that we can use to make predictions.
#In practice, you'll want to make predictions for new houses coming on the market rather than the houses we already have prices for. 
#But we'll make predictions for the first few rows of the training data to see how the predict function works.

print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))


#------------Stage 3: Model Validation -------------
#You've built a model. But how good is it?
#There are many metrics for summarizing model quality, but we'll start with one called Mean Absolute Error (also called MAE).
# lets calculate the mean absolute error

predicted_home_prices = melbourne_model.predict(X)
MAEVal=mean_absolute_error(y, predicted_home_prices)
print('mean absolute error is :',MAEVal)

#Since this pattern was derived from the training data, the model will appear accurate in the training data.
#But if this pattern doesn't hold when the model sees new data, the model would be very inaccurate when used in practice.
#Since models' practical value come from making predictions on new data, we measure performance on data that wasn't used to build the model.
#The most straightforward way to do this is to exclude some data from the model-building process
#The scikit-learn library has a function train_test_split to break up the data into two pieces.
#Split into validation and training data, the split is based on a random number generator.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model for train_X and train_y
melbourne_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
print('mean absolute error with model validate is :',mean_absolute_error(val_y, val_predictions))


#---------- Stage 4: Underfitting and Overfitting --------------
#overfitting, where a model matches the training data almost perfectly, but does poorly in validation and other new data.
#When model performs poorly even in training data, that is called underfitting
#Most imp is accuracy on new data, which we estimate from our validation data
#max_leaf_nodes argument provides a very sensible way to control overfitting vs underfitting

#utility function to help compare MAE scores from different values for max_leaf_nodes

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))


#------- Stage 6 : random forest (other modeling technique instead of Decision tree) ---------
#The random forest uses many trees, and it makes a prediction by averaging the predictions of each component tree.
#better predictive accuracy than a single decision tree and it works well with default parameters
# using RandomForestRegressor class for random model


#train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Define a random forest model
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))


# To improve accuracy, create a new Random Forest model which you will train on all training data
rf_model_on_full_data = RandomForestRegressor(random_state=1)
rf_model_on_full_data.fit(X,y)

# fit rf_model_on_full_data on all data from the training data


#--------------stage 7: testing with external data --------
# Load the data, and separate the target
test_file_path = './input/melb_data_test_with_id.csv'
#creating dateset from csv
test_home_data = pd.read_csv(test_file_path)

test_X = test_home_data[features]
test_preds = rf_model_on_full_data.predict(test_X)

output = pd.DataFrame({'Id': test_home_data.ID,
                       'Price': test_preds})
output.to_csv('submission.csv', index=False)


#------------Staege 8: Dealing with missing Values ----------
# Three Approaches  
# Drop Columns with Missing Values (not a good one)  - This approach would drop the column entirely!
# Imputation - can fill in the mean value along each column
#An Extension To Imputation - impute the missing values and add a new column that shows the location of the imputed entries

# Remove rows with missing target, separate target from predictors
home_data.dropna(axis=0, subset=['Price'], inplace=True)
y = home_data.Price
# To keep things simple, we'll use only numerical predictors
melb_predictors = home_data.drop(['Price'], axis=1)
X = melb_predictors.select_dtypes(exclude=['object'])

# Divide data into training and validation subsets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)

# Function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

#Using approach one 
# Get names of columns with missing values
cols_with_missing = [col for col in X_train.columns
                     if X_train[col].isnull().any()]

# Drop columns in training and validation data
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

print("MAE from Approach 1 (Drop columns with missing values):")
print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))


#approch 2
# Imputation
my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

# Imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

print("MAE from Approach 2 (Imputation):")
print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))

#approach 3 
# Make copy to avoid changing original data (when imputing)
X_train_plus = X_train.copy()
X_valid_plus = X_valid.copy()

# Make new columns indicating what will be imputed
for col in cols_with_missing:
    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
    X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()

# Imputation
my_imputer = SimpleImputer()
imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))
imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))

# Imputation removed column names; put them back
imputed_X_train_plus.columns = X_train_plus.columns
imputed_X_valid_plus.columns = X_valid_plus.columns

print("MAE from Approach 3 (An Extension to Imputation):")
print(score_dataset(imputed_X_train_plus, imputed_X_valid_plus, y_train, y_valid))

# ---------- partial stage 8: printing missing data detail ----------
# Shape of training data (num_rows, num_columns)
print(X_train.shape)

# Number of missing values in each column of training data
missing_val_count_by_column = (X_train.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])


#------------ stage 9 : categorical variable ---------------
#A categorical variable takes only a limited number of values.
# Three appproaches -  Drop Categorical Variables(not usefull), Ordinal Encoding, One-Hot Encoding 

# task 1
#obtain a list of all of the categorical variables in the training data.
#do this by checking the data type (or dtype) of each column(columns with text indicate categorical variables)

# Get list of categorical variables
traindata = X_train.head()
print(traindata)
s = (X_train.dtypes == 'object')
object_cols = list(s[s].index)

print("Categorical variables:")
print(object_cols)


#approach 1 
#assigned to drop_X_train, which now contains only the non-categorical columns from X_train.
drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_valid = X_valid.select_dtypes(exclude=['object'])

print("MAE from Approach 1 (Drop categorical variables):")
#This line prints a message indicating that the Mean Absolute Error (MAE) will be calculated for the approach where categorical variables are dropped.
print(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))

# approach 2
# Make copy to avoid changing original data 
label_X_train = X_train.copy()
label_X_valid = X_valid.copy()

# Apply ordinal encoder to each column with categorical data
#This creates an instance of the OrdinalEncoder.
ordinal_encoder = OrdinalEncoder()
#fit_transform is applied to the training data. It fits the encoder to the categorical columns in X_train and transforms them into numerical values.
label_X_train[object_cols] = ordinal_encoder.fit_transform(X_train[object_cols])
#transform is applied to the validation data. It uses the same encoding learned from the training data to transform the categorical columns in X_valid
label_X_valid[object_cols] = ordinal_encoder.transform(X_valid[object_cols])

print("MAE from Approach 2 (Ordinal Encoding):") 
print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))


#approach 3 - performing one-hot encoding on categorical features in a dataset and then preparing the data for a machine learning model
# Apply one-hot encoder to each column with categorical data
#OneHotEncoder is initialized with handle_unknown='ignore' to ignore unknown categories during transformation 
# and sparse=False to return a dense array instead of a sparse matrix.
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))

# One-hot encoding removed index; put it back, the indices of the original training and validation data are restored to the one-hot encoded DataFrames.
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# Remove categorical columns (will replace with one-hot encoding), The original categorical columns are removed from the training and validation data.
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

# Ensure all columns have string type. The column names are converted to strings to ensure consistency.
OH_X_train.columns = OH_X_train.columns.astype(str)
OH_X_valid.columns = OH_X_valid.columns.astype(str)

print("MAE from Approach 3 (One-Hot Encoding):") 
print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))



#-------------- Stage 9: pipelines to clean up your modeling code ---------------------
#check new py file








