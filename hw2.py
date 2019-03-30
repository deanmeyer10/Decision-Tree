import numpy as np
np.random.seed(42)

chi_table = {0.01  : 6.635,
             0.005 : 7.879,
             0.001 : 10.828,
             0.0005 : 12.116,
             0.0001 : 15.140,
             0.00001: 19.511}

def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns the gini impurity of the dataset.    
    """
    gini = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    #Isolating the last column
    lastCol = data[:,-1]
    #length on data (how many rows)
    colLen = len(data)
    #get number of zeros and ones
    num_zeros = (lastCol == 0).sum()
    num_ones = (lastCol == 1).sum()
    #gini formula
    gini = 1 - ((num_zeros/colLen)**2 + (num_ones/colLen)**2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return gini

def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns the entropy of the dataset.    
    """
# =============================================================================
#     entropy = 0.0
#     ###########################################################################
#     # TODO: Implement the function.                                           #
#     ###########################################################################
#     lastCol = data[:,-1]
#     #length on data (how many rows)
#     colLen = len(data)
#     #Create counterDict that will contain the counted 0s and 1s
#     num_zeros = (lastCol == 0).sum()
#     num_ones = (lastCol == 1).sum()
#     #entropy formula
#     entropy = -((num_zeros/colLen)*np.log2(num_zeros/colLen)) - ((num_ones/colLen)*np.log2(num_ones/colLen)) 
# =============================================================================
    
    y = data[:,-1]
    _, counts = np.unique(y, return_counts=True)
    entropy = (counts / len(y)) * np.log2(counts / len(y))
    entropy = -np.sum(entropy)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return entropy



def info_gain(data, feature_index, threshold, impurity):
    """
    Calculate the Information Gain for a specified attribute_name given the data set (data).

    Input:
    - data: any dataset where the last column holds the labels.
    - feature_index: the feature (feature index) by which the tree will be split by.
    - threshold: the value that divides the two children nodes.
    - impurity: gini or entropy formulas.

    Returns the Information Gain for the specified feature and its two children.    
    """
    #Calculate the information of the total dataset.
    feature_info = impurity(data)
    
    #Find weighted child information gain.
    #children contain the matrix of the instances left over that hold (child1) and dont (child2) hold the threshold condition.
    child1 = data[data[:,feature_index] <= threshold]
    child2 = data[data[:,feature_index] > threshold]
    
    #calculate weighted information result for children.
    child_info = len(child1)/len(data)*impurity(child1) + len(child2)/len(data)*impurity(child2)
    
    info_gain = feature_info - child_info
    
    return info_gain, child1, child2

def best_threshold(data, index, impurity):
    """
    Calculate the the best threshold for this given time.

    Input:
    - data: any dataset where the last column holds the labels.
    - index: the current index of the feature (column)
    - impurity: gini or entropy formulas.

    Returns both the best threshold and the Information Gain when using this threshold (max info gain).    
    """
    #max info gain and its related best threshold
    maximum_info = 0
    best_threshold_val = 0
    #indexed column
    column = data[:,index]
    #values contains a sorted list of all unique elements
    values = np.unique(column)
    for i in range(len(values)-1):
        #if we are at the last value, we must break.
        #find the best threshold.
        threshold = (values[i] + values[i+1])/2
        info_gain_val, _, _ = info_gain(data, index, threshold, impurity)
        if (info_gain_val >= maximum_info):
            maximum_info = info_gain_val
            best_threshold_val = threshold
    
    #we will return both the best info gain and its correspoding threshold
    return maximum_info, best_threshold_val
    


def get_best_feature(data, impurity):
    
    """
    Calculate the the best feature for this given time.

    Input:
    - data: any dataset where the last column holds the labels.
    - impurity: gini or entropy formulas.

    Returns the index of the best feature and its best threshold.    
    """
    #maximum Information gane.
    maximum = 0
    best_threshold_val = 0
    #find the best column index
    for index in range(len(data[0])-1):
        #check for the best feautue by finding its best threshold and comparing against others best thresholds.
        current_info_gain, threshold = best_threshold(data, index, impurity)
        if (current_info_gain >= maximum):
            maximum = current_info_gain
            best_index_column = index
            best_threshold_val = threshold
            
    return best_index_column, best_threshold_val

def get_best_features_children(data, index, threshold):
    """
    Calculate the the best features children.

    Input:
    - data: any dataset where the last column holds the labels.
    - index: current index given
    - impurity: gini or entropy formulas.

    Returns the index of at the feature.    
    """
    #Find weighted child information gain.
    #children contain the matrix of the instances left over that hold (child1) and dont (child2) hold the threshold condition.
    child1 = data[data[:,index] <= threshold]
    child2 = data[data[:,index] > threshold]
    
    return child1, child2
    
        

class DecisionNode:

    # This class will hold everything you require to construct a decision tree.
    # The structure of this class is up to you. However, you need to support basic 
    # functionality as described in the notebook. It is highly recommended that you 
    # first read and understand the entire exercise before diving into this class.
    
    def __init__(self, feature, value, data):
        self.feature = feature # column index of criteria being tested
        self.value = value # value necessary to get a true result
        self.children = [] #array of children
        self.data = data #data of the current itteration
        
    def add_child(self, child):
        self.children.append(child)



def build_tree_helper(root_of_tree, impurity):
    """
    Builds a tree using the given impurity measure and a root (decisionNode). 

    Input:
    - decisionNode: root of the tree of type decisionNode
    - impurity: the chosen impurity measure.

    Output: the root node of the tree.
    """
    
    #check purity of current DicisionNode
    if (impurity(root_of_tree.data) == 0):
        return root_of_tree
    
    #get best best_feature and best_threshold for given data
    best_feature, best_threshold = get_best_feature(root_of_tree.data, impurity)
    #set feature and values fields of this decisionnNode object.
    root_of_tree.feature = best_feature
    root_of_tree.value = best_threshold
    #get children after split    
    child1, child2 = get_best_features_children(root_of_tree.data, best_feature, best_threshold)
    #get best values for children in order to make them correct desicionNodes
    child1_best_feature, child1_best_threshold = get_best_feature(child1, impurity)
    child2_best_feature, child2_best_threshold = get_best_feature(child2, impurity)
    #create decisonNodes for child1,child2
    child1_dec_node = DecisionNode(child1_best_feature, child1_best_threshold, child1)
    child2_dec_node = DecisionNode(child2_best_feature, child2_best_threshold, child2)
    #recursively add children to the roots values.
    root_of_tree.add_child(build_tree_helper(child1_dec_node, impurity))
    root_of_tree.add_child(build_tree_helper(child2_dec_node, impurity))
    
    return root_of_tree
    
    

def build_tree(data, impurity):
    """
    Build a tree using the given impurity measure and training dataset. 
    You are required to fully grow the tree until all leaves are pure. 

    Input:
    - data: the training dataset.
    - impurity: the chosen impurity measure. Notice that you can send a function
                as an argument in python.

    Output: the root node of the tree.
    """
    
        
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    #
    root_of_tree = DecisionNode(None, None, data)
    #build_tree_helper will build and return the root of the tree recursively
    root = build_tree_helper(root_of_tree, impurity)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return root

    

def predict(node, instance):
    """
    Predict a given instance using the decision tree

    Input:
    - root: the root of the decision tree.
    - instance: a row vector from the dataset. Note that the last element 
                of this vector is the label of the instance.

    Output: the prediction of the instance.
    """
    pred = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    
    
    
# =============================================================================
#     first_feature = node.feature
#     first_threshold = node.value
#     if(instance[first_feature] <= first_threshold):
#         go_right = True
#         go_left = False
#     else:
#         go_left = True 
#         go_right = False
#         
#         
#     for i in range(len(root.children)):
#         if()
#         
# =============================================================================
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pred

def calc_accuracy(node, dataset):
    """
    calculate the accuracy starting from some node of the decision tree using
    the given dataset.

    Input:
    - node: a node in the decision tree.
    - dataset: the dataset on which the accuracy is evaluated

    Output: the accuracy of the decision tree on the given dataset (%).
    """
    accuracy = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return accuracy

def print_tree(node):
    '''
    prints the tree according to the example in the notebook

	Input:
	- node: a node in the decision tree

	This function has no return value
	'''

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################    
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
