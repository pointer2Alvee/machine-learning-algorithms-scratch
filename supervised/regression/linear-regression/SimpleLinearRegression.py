# Importing 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

############################ (1) DATA PREPROCESSING ############################

# (1.1) Data-Collection 
# Read Raw Dataset 
dataset = pd.read_csv('D:\Mastery\Software_MachineLearning\datasets\Salary_Data.csv')

# Feature-Matrix (X) & Dependent-Variable(y)
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# (1.2) Data-Cleaning 
# (1.3) Data-Transformation 

# (1.4) Data-Splitting 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

# (1.5) Data-Collection 
# (1.6) Data-Tf (Feature-Scaling) 

# CUSTOM MADE DATASET :-
# from sklearn import datasets
# X, y = datasets.make_regression(n_samples=150, n_features=1, noise=20, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=24)


############################ (2) ML MODEL : SLR ################################

class SimpleLinearRegression: 
    
    # (2.1) Initialize Model-Parameters 
    def __init__(self, lr=0.001, iters=10000) -> None:     
        """
        Initializes the SLR-model's parameters and Hyper-parameters.
        
        Parameters: 
            weights : (np.array) weight-vector of model, Instance-Var so (value stays same for a specific instance)    
            bias    : (float) bias-term o the model, Instance-Var so (value stays same for a specific instance)              
            lr      : (float) Learning-Rate, [Hyper-Parameter]
            iters   : (int) Total number of Iterations for traininig the whole Dataset, [Hyper-Parameter]
                    
        Returns:     
            None
        """
        
        # Model-Parametes
        self.weights = None
        self.bias = None
        
        # Hyper-Parameter
        self.lr = lr
        self.iters = iters
         
        
    # (2.2) Making-Predictions
    def predict(self, X):
        """
        For given Features(x)[x1], predicts the output(ŷ)/y_pred is np array. 
        
        Parameters:
            X : (np.array) Independent-Variable (Features)
        
        Returns : 
            The predited output(ŷ)/y_pred as np array.
        """
        
        # ŷ = wx + b   
        y_pred = np.dot(X, self.weights) + self.bias 
        return y_pred
    

    # (2.3) Cost-Function
    def mean_square_error(self, y_act, y_pred) -> np.ndarray:
        """
        Predicts the error score between real(y) and predicted values(ŷ), using MSE
        
        Parameters:
            y-act  : (np.array) Actual values
            y_pred : (np.array) Predited values
        
        Returns: 
            The loss-score mse as an int in np array 
        """

        mse = None
        mse = np.mean((y_act - y_pred)**2)
        return mse


    # (2.4) Fit Model to Data (Learning-&-Training)  
    def fit(self, X, y):
        """
        Learns the model-Params (w & b) using GD algo and Trains the model with iters
        
        Parameters:
            X : (np.array) Independent-Variable (Features)
            y : (np.array) Dependent-Variable
        """
        
        # Initialize Model-Parameters
        n_data, n_features = X.shape
        
        # Init With zero val : (Shows where two curves against weight, below small curve is where actual learning happens) Starting from 0 val<min, the slope increases and curve shifts from left --> right and eventually goes to min
        # self.weights = np.zeros(n_features)
        # self.bias = 0
        
        # Init With vals : (Shows actually where the MSE score is decreasing with change in weight) actual learning plot. Starting from val>min, the slope decreases and curve shifts from right --> to left and eventually goes to min
        self.weights = np.full(n_features,11401.06472749)
        self.bias = 11170.596516046198

        # Create empty list to Store all Costs & Weights
        cost_history = []
        weight_history = []
        
        
        # Training Model  
        for i in range(self.iters):                 
            # ŷ = wx + b   
            y_pred = self.predict(X)
            
            # Learning : Gradient-Descent / Back-prop Algo
            # Deriv of Cost-Func (J) w.r.t weights(w)
            dj_dw = (-2/n_data) * np.dot(X.T, (y- y_pred))
            
            # Deriv of Cost-Func (J) w.r.t bias(b)
            dj_db = (-2/n_data) * np.sum(y-y_pred)

            # Back-Prop Algo
            self.weights -= self.lr * dj_dw
            self.bias -= self.lr * dj_db
            
            cost = self.mean_square_error(y,y_pred)
            
            cost_history.append(cost)
            weight_history.append(int(self.weights))
            
            if i % 1000 == 0:
                print (f"iter={i} weight={self.weights} bias={self.bias} cost={cost}")               
                
        return y_pred, self.weights, self.bias, cost_history, weight_history             


"""
# CODE EXPLANATION :-
--> (2.4) Fit Model to Data (Learning-&-Training)  
    
    -> # Initialize Model-Parameters 
            # Init with zero vals
        ->  n_data, n_features = X.shape :- 
            * X.shape gives a tuple with values (30,1). So the vars n_data, n_features is assigned with 30 and 1 value respectively, n_data = 30 , n_features = 1
        
        -> self.weights = np.zeros(n_features) :-
           * Creates a numpy array with length/no of elements equals to n_features where each element is zero. Cause assign all weights (one weight per col) to zero. 
           * So for each feature we have a weight and for each weight we have an element in the array. So n_features = 3, means self.weights = [0 0 0]. 
           * In this case we have only one feature one weight as the model is SLR. 
        
        -> Can also use random values for self.weights & self.bias instead of 0  
        
        
    -> # Learning : Gradient-Descent / Back-prop Algo 
        -> y_pred = np.dot(X, self.weights) + self.bias :-
           * Instead of using a for loop to multipy all dataPoints in X and add them here is used dot product which is basically same, but much faster than for loop
           * dot product is ""sum of products"", w1x1 + w2X2 + w3X3 + ...
           * This is basically y = wx + b formula , 
           * here y_pred is the whole col, wx = np.dot(X, self.weights), where X is the whole col
           * Here only 1 weight(w), so for each row, the weight is multiplied by each of the feature element 
           kenoa
           y_pred₁ = wX₁ + self.bias  [ here, np.dot(X[1], self.weights) = wX₁ ]
           y_pred₂ = wX₂ + self.bias  [ here, np.dot(X[2], self.weights) = wX₂ ]
           y_pred₃ = wX₃ + self.bias  [ here, np.dot(X[3], self.weights) = wX₃ ]       
           .
           .
           y_predn = wXn + self.bias  [ here, np.dot(X[n], self.weights) = wXn ]  
           
           * Here np.dot(X, self.weights) = product of each "datapoint and weight" and then sum of those products. 
           * np.dot(X, self.weights) = wX₁ + wX₂ + wX₃.... + wXn
           
        -> dj_dw = (1/n_data) * np.dot(X.T, (y_pred - y)) :-
           * This is the eqn of the deriv of cost_fucn (J) w.r.t weights (w) 
           * np.dot(X.T, (y_pred - y)) = the product of each dataPoints of X and (y_pred - y) and then summation of the products
           * np.dot(X.T, (y_pred - y)) =  X.T[1](y_pred - y)[1] + X.T[2](y_pred - y)[2] + ..
           * the Xi in dw formila is X.T
           
           
           * WHY TRANSPOSE? :- 
           * X.T is X.Transpose, transpose provides compitable dimentions for dot product 
           * Shape = (row, col), dot product condition -> inner dimentions must match
           * like: (2,3).(3,2) is OK, but (3,4).(3,4) can not perform DOT product
           * Because, X's shape = (n_data, n_features), (y_pred - y)'s shape = (n_data, 1)
           * So without transpose if we take dotProduct of X and (y_pred - y) we would have 
           the shape (n_data, n_features).(n_data, 1), here n_features ≠ n_data, so dot product not possible
           * Withdot transpose X.T has shape = (n_features , n_data), so if we take dotProduct 
           of X.T and (y_pred - y) we would have the shape  (n_features , n_data).(n_data, 1), here  n_data = n_data so inner-dimen matched hence dot product possible
           
           
            
    -> WHY X & y are not stored as instance variables (self.X = X & self.y =y) ?
        -> X and y are only needed during the fitting process to calculate the coefficients (w & b)
        Once the model is fitted, these coefficients are the only values required to make predictions.
        
        -> Memory Efficiency: Storing large datasets (X & y) as instance variables can consume a significant amount of memory. 
        By not storing X and y, the model only retains the necessary parameters (slope and intercept), which are much smaller in size.
        
        -> Simplicity : avoid adding unnecessary instance vars
        
        -> Immutability of Data: Once the model is trained, X and y are no longer needed for making predictions or evaluating the model. 
        If you store X and y, they become part of the object's state, which might lead to unintended side effects if the data is accidentally modified.
-->

"""

            
############################ IMPLEMENTING THE MODEL ############################
# Creating object of Type SimpleLinearRegression
regressor = SimpleLinearRegression(0.001,10000)

# Training The SLR model with the Training-Datasets
y_train_pred, weights, bias, cost_history, weight_history = regressor.fit(X_train,y_train)

# Predicting on Test-Set using the trained SLR model (by the learned 'weights & bias')
y_pred = regressor.predict(X_test)



############################ (3) ML MODEL EVALUATION ###########################

# (3.1) Visualization 
# Multiple-Plots on Single-Figure 
fig, ax = plt.subplots(3, 1, figsize=(6, 9))

# Training-Set plot (X_train Vs y_train_pred)
ax[0].scatter(X_train, y_train, color='red')
ax[0].plot(X_train, y_train_pred, color='blue')
ax[0].set_title("SLR on Training-Set")
ax[0].set_xlabel('Years Of Experience')
ax[0].set_ylabel('Salary')

# Test-Set plot (X_test Vs y_pred)
ax[1].scatter(X_test, y_test, color='red')
ax[1].plot(X_test, y_pred, color='blue')
ax[1].set_title("SLR on Test-Set")
ax[1].set_xlabel('Years Of Experience')
ax[1].set_ylabel('Salary')

# Cost-Function / loss-Curve plot (MSE score Vs Weights)
ax[2].scatter(weight_history, cost_history, color='red', marker='+', s=10)
ax[2].set_title("Cost Function / Loss Curve")
ax[2].set_xlabel('Weights')
ax[2].set_ylabel('Cost (MSE)')

# Adjust the layout to prevent overlapping
plt.tight_layout()

####### Show all plots in one figure #########
plt.show()

# Single-Plot on Single-Figure 
# Training-Set plot
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, y_train_pred, color = 'blue')
plt.title("SLR on Training-Set ")
plt.xlabel('Years Of Experience')
plt.ylabel('Salary')
#plt.show()


# (3.2) Performance-Metrics 


"""
# CODE EXPLANATION :-
--> (3.1) Visualization
    
    -> # Multiple-Plots on Single-Figure
        ->  fig, ax = plt.subplots(3, 1, figsize=(5, 9)) :- 
            * fig = The overall figure object that contains all the subplots.
            * ax = An array of Axes objects, each representing one of the subplots within the figure.
              (or a single object if there's only one subplot)  
            * Each Axes object in ax is where you actually draw your plots (like scatter plots, line plots, etc.). 
            * In the code provided, ax[0], ax[1], and ax[2] refer to the train-set, test-set, and mse-curve subplot areas respectively.
            
            * 3,1 = 3 subplots in 1 col
            * figsize =(5,9) = (width, height)
            
"""