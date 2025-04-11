# imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

# Read Raw Dataset 
col_names = ['x0', 'x1', 'x2', 'x3', 'x4', 'y']
dataset = pd.read_csv("../../../datasets/airfoil_self_noise.csv", skiprows=1, header=None, names=col_names)
print(dataset.head(5)) # values of y are real continous numbers , hence we can be sure that its a regression problem


# Feature-Matrix (X) & Dependent-Variable(y)
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values.reshape(-1,1)

# Data-Splitting 
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=.2, random_state=41) # 21% test size 79% train size


# Custom Dataset :-
# X, y = datasets.make_regression(n_samples=150, n_features=1, noise=20, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=24)

# Data Pre-Processing
# Normalize (if needed)
# Data Visualization


########### DECISION-TREE NODE ###########
class Node():
  
    def __init__(self, feature_index=None, threshold=None, left_child=None, 
                 right_child=None, variance_reduction=None, value=None) -> None:
        """
        Describes the node of the decision tree. Based on feature_index and threshold we
        define the condition of the decision nodes. Like x2 <= 32 , 
        here x2 = x feature, with feature_index = 2
            32 is the threshold which when satisfied or not we go either left or right child
        
        Parameters-Variables: 
            feature_index : [param]() Index of a particular feature col
            threshold     : [param]() Threshold of that particular feature col
            left_child    : [param]() Left node of the root node
            right_child   : [param]() Right node of the root node
            variance_reduction     : [param]() Inormation gain
            value         : [param]()    Majority class of datapoints in the leafnode

        """      
        # for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.variance_reduction = variance_reduction

        # for leaf node 
        self.value = value 


########### DECISION-TREE CLASS ###########
class DecisionTreeRegressor():
    def __init__(self, min_samples_split=2, max_depth=2) -> None:
        """
        Constructor
        
        Parameter-Variable:
            min_samples_split
            max_depth
        Returns:
            None
        """
        
        # Root node of the tree / needed to traverse the tree
        self.root_node = None 

        # stopping conditions to see if a node is leafnode or we dont want to traverse deeper
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        
        
    def build_tree(self, dataset, curr_depth=0) -> Node:
        """
        A recursive function to build the binary tree using recursion
        """
        
        # Separating Features and labels into two separate variables
        X, y = dataset[:,:-1], dataset[:,-1]
        n_data, n_features = np.shape(X) # n_data == n_samples same thing
        
        # split datapoints from dataset into left/right and build the tree until conditions are met
        if n_data >= self.min_samples_split and curr_depth <= self.max_depth:
            
            # Find the best split / best_split is a dicionary
            best_split = self.get_best_split(dataset, n_data, n_features)
            
            # check if info-gain is positive
            
            if best_split["variance_reduction"]>0: 
            # if variance_reduction = 0, means node is leaf node or pure node meaning contains data points of only one class
            # if variance_reduction = 1, means each class has equal datapoints in that node
                # recurrsion left / build left sub tree
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1)
                
                # recurrsion right / build right sub tree
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)
                
                # return decision node / # 5 params as its a decision node
                return Node(best_split["feature_index"], best_split["threshold"],
                            left_subtree, right_subtree, best_split["variance_reduction"])
                
               
        # compute and return leaf node
        leaf_value = self.calculate_leaf_value(y)
        return Node(value=leaf_value) # only 1 param as its a leaf node
    
    
    
    def get_best_split(self, dataset, n_data, n_features):
        """
        get the best split of data based traversing all features_index and features values
        """

        # dictionary to store the best split
        # best_split = {
        #     "feature_index" : None,                    
        #     "threshold": None,
        #     "dataset_left": None,
        #     "dataset_right": None,
        #     "variance_reduction": 0
            
        # }
        best_split = {}
        max_variance_reduction = -float("inf")
        
        # Loop over all features
        for feature_index in range(n_features): # n_features = no of cols
            feature_values = dataset[:, feature_index] # all rows/values of that col/feature
            possible_thresholds = np.unique(feature_values) # unique values of each Feature-col
            
            # Loop over all possible thresholds / unique values of a particular feature
            for threshold in possible_thresholds:
                # split dataset (data points for left or right subtree) based on curr feature_idx and curr threshold
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold) # 
                
                # check if child are not Null
                if len(dataset_left)>0 and len(dataset_right)>0 :
                    
                    # from the split datasets we extract the label column
                    y, left_y , right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:,-1] 
                    
                    # compute info gain, gini method
                    curr_variance_reduction = self.get_variance_reduction(y, left_y, right_y)
                    
                    # update the best split if needed, meaning if in the current loops we get the beter info gain , we just update the currest best infomations
                    if curr_variance_reduction > max_variance_reduction:
                        best_split["feature_index"] = feature_index                        
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["variance_reduction"] = curr_variance_reduction
                        max_variance_reduction = curr_variance_reduction

        # returns best split
        return best_split            
    
    
    # split function
    def split(self, dataset, feature_index, threshold):
        # traverse each-row for each feature and distribute datapoints/ rows accoring to threshold conditon
        dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold]) 
        dataset_right = np.array([row for row in dataset if row[feature_index]>threshold]) 
        return dataset_left, dataset_right
    
    
    # info gain in classification but in regression its variance reuction
    def get_variance_reduction(self, parent, l_child, r_child): 
        weight_l = len(l_child)/ len(parent)
        weight_r = len(r_child)/ len(parent)
        reduction = np.var(parent) - (weight_l*np.var(l_child) + weight_r*np.var(r_child))
        return reduction

   
    
    # calculate the class of the majority items in the leaf node
    def calculate_leaf_value(self, Y):
        val = np.mean(Y)
        return val

    # customizely print the tree
    def print_tree(self, tree=None, indent=" "):
        
        if not tree:
            tree = self.root_node
        
        if tree.value is not None:
            print(tree.value)
                
        else:
            # preorder traversal
            print("X_"+str(tree.feature_index), "<=", tree.threshold, "?", tree.variance_reduction)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left_child, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right_child, indent + indent)
    
            
    # fit function (training and learning splitting thresholds and building the best possible tree)
    def fit(self, X, Y):
        dataset = np.concatenate((X,Y), axis=1)
        self.root_node = self.build_tree(dataset) # build the tree based on the best split
        """
        Here like other ML alogs, where we learn weight and bias, here we learn the spliting thresholds
        for which we get the best splitted tree
        """
        
    # predict when we get a new value by passing in to the best splitted / trained tree
    def predict(self, X):
        predictions = [self.make_predictions(x, self.root_node) for x in X]
        return predictions
    
    
    # make predictions()
    def make_predictions(self, x, tree) :
        if tree.value!= None : return tree.value
        feature_val = x[tree.feature_index]
        if feature_val<=tree.threshold:
            return self.make_predictions(x, tree.left_child)
        else:
            return self.make_predictions(x, tree.right_child)
        
        
regressor = DecisionTreeRegressor(min_samples_split=3, max_depth=3)
regressor.fit(X_train, Y_train)
print("The Decision Tree: ")
regressor.print_tree()

Y_pred = regressor.predict(X_test)
print(f"root mean squared error: {root_mean_squared_error(Y_test, Y_pred)}")