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

    col = data[:,-1]
    _, counts = np.unique(col, return_counts=True)
    entropy = (counts / len(col)) * np.log2(counts / len(col))
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
    best_index_column = 0
    #find the best column index
    for index in range(len(data[0])-1):
        #check for the best feature by finding its best threshold and comparing against others best thresholds.
        current_info_gain, threshold = best_threshold(data, index, impurity)
        if (current_info_gain > maximum):
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
    
    def __init__(self, feature, value, data, chi_square):
        self.feature = feature # column index of criteria being tested
        self.value = value # value necessary to get a true result
        self.children = [] #array of children
        self.data = data #data of the current itteration
        self.chi_square = chi_square
        self.parent = None
        self.root = True
        self.isLeaf = None
        
    def add_child(self, child):
        self.children.append(child)
    
    def has_children(self):
        if (self.children == []):
            return False
        else:
            return True
    def set_parent(self, parent):
        self.parent = parent
        
    def get_parent(self):
        return self.parent
    
    def get_P_val(self):
        return self.chi_square


def build_tree_helper(root_of_tree, impurity, p_val = 1):
    """
    Builds a tree using the given impurity measure and a root (decisionNode).

    Input:
    - decisionNode: root of the tree of type decisionNode
    - impurity: the chosen impurity measure.

    Output: the root node of the tree.
    """
    
    #check purity of current DicisionNode
    #TODO  make condition for the case of pruning (where p_val != 1) as this means not all leaves have impurity zero.
    if (impurity(root_of_tree.data) == 0):
        root_of_tree.isLeaf = True
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
    child1_dec_node = DecisionNode(child1_best_feature, child1_best_threshold, child1, p_val)
    child2_dec_node = DecisionNode(child2_best_feature, child2_best_threshold, child2, p_val)
    #assign them their parents
    child1_dec_node.set_parent(root_of_tree)
    child2_dec_node.set_parent(root_of_tree)
    child1_dec_node.root = False
    child2_dec_node.root = False
    
    #recursively add children to the roots values.
    #if chi_square == 1 then continue building as normal
    if(p_val == 1):
        root_of_tree.add_child(build_tree_helper(child1_dec_node, impurity, p_val))
        root_of_tree.add_child(build_tree_helper(child2_dec_node, impurity, p_val))
    else:
        if(pre_prune(root_of_tree) >= chi_table[p_val]):
            root_of_tree.isLeaf = True
            root_of_tree.add_child(build_tree_helper(child1_dec_node, impurity, p_val))
            root_of_tree.add_child(build_tree_helper(child2_dec_node, impurity, p_val))
        
    return root_of_tree
    
    

def build_tree(data, impurity, p_val=1):
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
    root_of_tree = DecisionNode(None, None, data, p_val)
    #build_tree_helper will build and return the root of the tree recursively
    root = build_tree_helper(root_of_tree, impurity, p_val)
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
    while((not (calc_entropy(node.data) == 0)) and (node.has_children())):
        if ((instance[node.feature] <= node.value)):
            node = node.children[0]
        else:
            node = node.children[1]
    pred = node.data[0, -1]
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
    Sum = 0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    for row in dataset:
        #get actual prediction
        actual_predic = row[-1]
        predicted_val = predict(node, row)
        if(actual_predic == predicted_val):
            Sum+=1
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    accuracy = (Sum/len(dataset))*100
    
    return accuracy

def pre_prune(node):
    '''
    returns the chi-value according to the formula

	Input:
	- node: a node in the decision tree

	This function returns the chi-value
	'''
    chi_square = 0
    #num_ofs contains at index 0 how many instances where Y=0 and at index 1 how many Y=1 instances
    #last column containing lables
    last_col = node.data[:,-1]
    num_zeros = (last_col == 0).sum()
    num_ones = (last_col == 1).sum()
    num_of_instances = len(node.data)
    prob_y0 = num_zeros / num_of_instances
    prob_y1 = num_ones / num_of_instances
    #child sizes
    child1 = node.data[node.data[:,node.feature] <= node.value]
    child2 = node.data[node.data[:,node.feature] > node.value]
    C_counts0 = len(child1)
    C_counts1 = len(child2)
    C_array = np.array([C_counts0, C_counts1])
    pf_val = np.array([0, 0])
    df_val = np.array([0, 0])
    
    pf_val[0] = (child1[:,-1] == 0).sum()
    pf_val[1] = (child2[:,-1] == 0).sum()
    
    df_val[0] = (child1[:,-1] == 1).sum()
    df_val[1] = (child2[:,-1] == 1).sum()
    values = [0,1]
    for val in values:
        E0 = C_array[int(val)] * prob_y0
        E1 = C_array[int(val)] * prob_y1
        chi_square += (np.square((pf_val[int(val)] - E0)) / E0) + (np.square(df_val[int(val)] - E1) / E1)
    
    
    return chi_square


#SMALL AUX FUNCTIONS USED FOR POST_PRUNE    

def find_numNodes(node):
    #Return the num of internal nodes
    if (calc_entropy(node.data) == 0 or len(node.children) == 0):
        return 0
    else:
        return 1 + find_numNodes(node.children[0]) + find_numNodes(node.children[1])

def possible_parents(root):
    #return all possible parents using queue.
    NodeQueue = [root]
    possibleParents = []
    while(len(NodeQueue)>0):
        curNode = NodeQueue.pop(0)
        if(curNode.children[0].isLeaf or curNode.children[1].isLeaf): possibleParents.append(curNode)   
        if (not curNode.children[0].isLeaf): NodeQueue.append(curNode.children[0])
        if (not curNode.children[1].isLeaf): NodeQueue.append(curNode.children[1])       

    return possibleParents 

def post_pruning(root, trainData, testData):
    #predefine
    trainAccuracysArr = [calc_accuracy(root, trainData)]
    testAccuracysArr = [calc_accuracy(root, testData)]
    numberOfNodesArr = [find_numNodes(root)]
    while (root.children != []):
        bestAccuracy = -1
        bestParent = None
        possibleParents = possible_parents(root)
        #test accuracy for this iteration
        for currentParent in possibleParents:
            #store values 
            tempChildren = currentParent.children
            currentParent.children = []
            accuracy = calc_accuracy(root, trainData)
            if accuracy > bestAccuracy:
                bestParent = currentParent
                bestAccuracy = accuracy
            #return the trimmed children to the parent node 
            currentParent.children = tempChildren


        #use found values to continue cutting tree
        bestParent.children = []
        bestParent.isLeaf = True

        trainAccuracysArr.append(bestAccuracy)
        testAccuracysArr.append(calc_accuracy(root, testData))
        numberOfNodesArr.append(find_numNodes(root))

    return (numberOfNodesArr,trainAccuracysArr,testAccuracysArr)           



def print_tree(node, i=0):
    '''
    prints the tree according to the example in the notebook

	Input:
	- node: a node in the decision tree

	This function has no return value
	'''

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################    

    if(calc_gini(node.data) == 0):
        print("  {}leaf: {{{}: {}}}".format(('  '*i),node.data[0,-1], len(node.data)))
    else:
        i = i+1
        print("{}[X{} <= {}],".format(('  '*i),node.feature, node.value))
        print_tree(node.children[0], i)
        print_tree(node.children[1], i)
    
    if((node.children == [])):
        return
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
