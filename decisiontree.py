import numpy as np
from collections import Counter

#The following decision tree implementation allows for splitting based on entropy or gini index

class Node:
    """Represents a node in the decision tree."""    
    def __init__(self, feature=None, threshold=None, left=None, right=None, cellType=None, numSamples=None, impurity=None):
        self.feature = feature          #Index of feature that is being split with
        self.threshold = threshold      #Threshold value for split
        self.left = left                #Left subtree
        self.right = right              #Right subtree
        self.cellType =  cellType       #(Leaves only): Cell type prediction  
        self.numSamples = numSamples    #Number of samples at this node
        self.impurity = impurity            #Purity measure based on the feature being split on
    
    def isLeaf(self):
        """Check if node is a leaf."""
        return self.cellType is not None #Only leaves have cellType label



class DecisionTree:
    """Decision Tree for classification and regression."""
    def __init__(self, purityMeasure=None, maxDepth = None, minSamplesSplit = None, minSamplesLeaf=None):
        self.purityMeasure = purityMeasure      #Purity measure (gini or entropy) to compute infogain with
        self.maxDepth = maxDepth                #Max depth tree can grow to (None = unlimited)
        self.minSamplesSplit = minSamplesSplit  #Minimum samples required to split a node
        self.minSamplesLeaf = minSamplesLeaf    #Minimum samples required at a leaf node
        self.tree = None
        self.numFeatures = None
        self.classes = None


    def _growTree(self, X, y, depth=0):
        """Recursively grow the decision tree"""

        numSamples = X.shape[0]
        numClasses = len(np.unique(y))

        #Stopping criteria
        if (self.maxDepth is not None and depth >= self.maxDepth) or numSamples < self.minSamplesSplit or numClasses == 1:
            return self._createLeaf(y)
        
        #Try all splits
        bestSplit = self._findBestSplit(X, y)
        
        if bestSplit is None:
            return self._createLeaf(y)
        
        feature, threshold, leftIndex, rightIndex = bestSplit
        
        # Recursively build left and right subtrees
        leftSubtree = self._growTree(X[leftIndex], y[leftIndex], depth + 1)
        rightSubtree = self._growTree(X[rightIndex], y[rightIndex], depth + 1)
        
        # Calculate impurity
        impurity = self._impurity(y)
        
        return Node(
            feature=feature,
            threshold=threshold,
            left=leftSubtree,
            right=rightSubtree,
            numSamples=numSamples,
            impurity=impurity
        )
    

    def _findBestSplit(self, X, y):
        """Find the best feature and threshold to split a node"""
        numSamples, numFeatures = X.shape
        bestGain = -np.inf
        bestSplit = None

        for feature in range(numFeatures):
            #Sort values for this feature
            sortedIndices = np.argsort(X[:, feature])
            xSorted = X[sortedIndices]
            ySorted = y[sortedIndices]

            #Get all unique values for current feature
            values = xSorted[:, feature]
            uniqueValues = np.unique(values)

            #If only one value -> No split possible -> Continue
            if len(uniqueValues) == 1:
                continue
            
            #Candidate thresholds = Midpoints between consecutive unique values
            thresholds = (uniqueValues[:-1] + uniqueValues[1:])/2

            for threshold in thresholds:
                #Split data indices based on current feature
                leftIndex = X[:, feature] <= threshold
                rightIndex = ~leftIndex

                #Skip if split leads to too few samples per leaf
                if np.sum(leftIndex) < self.minSamplesLeaf or np.sum(rightIndex) < self.minSamplesLeaf:
                    continue
                
                gain = self._infoGain(y, y[leftIndex], y[rightIndex])

                #If feature + threshold lead to higher info gain -> Update
                if gain > bestGain:
                    bestGain = gain
                    bestSplit = (feature, threshold, leftIndex, rightIndex)

        return bestSplit


    def _impurity(self, y):
        """Compute impurity based on criterion."""
        if self.purityMeasure == "entropy":
            return entropy(y)
        elif self.purityMeasure == "gini":
            return gini(y)
        

    def _infoGain(self, node, leftChild, rightChild):
        """Compute the information gain resulting from a feature split for any purity measure"""

        nodeLength = len(node)
        leftLength = len(leftChild)
        rightLength = len(rightChild)

        #If either left or right child has no samples -> Splitting on this feature yields no info gain
        if leftLength == 0 or rightLength == 0:
            return 0.0
    
        nodeImpurity = self._impurity(node)
        
        leftImpurity = self._impurity(leftChild)
        rightImpurity = self._impurity(rightChild)

        #Weighted average of children's impurities
        childImpurity = (leftLength/nodeLength) * leftImpurity + (rightLength/nodeLength) * rightImpurity

        #Information Gain = (Node's Impurity) - (Children's Weighted Average Impurity)
        return nodeImpurity - childImpurity


    def _createLeaf(self, y):
        """Create a leaf node."""

        #Pick most common class via majority vote
        value = Counter(y).most_common(1)[0][0]
        return Node(cellType=value, numSamples=len(y), impurity=0)
    

    def _traverseTree(self, x, node):
        """Traverse tree to find prediction for sample x."""

        #Base case
        if node.isLeaf():
            return node.cellType
        
        #Recusively traverse tree
        if x[node.feature] <= node.threshold:
            return self._traverseTree(x, node.left)
        else:
            return self._traverseTree(x, node.right)
    

    #Methods
    def fit(self, X, y):
        """Build decision tree classifier"""
        X = np.array(X)
        y = np.array(y)
        
        self.numFeatures = X.shape[1]
        self.classes = np.unique(y)
        
        #Grow tree
        self.tree = self._growTree(X, y)
        return self

    def predict(self, X):
        """Predict class/value for all samples in data"""
        X = np.array(X)
        return np.array([self._traverseTree(x, self.tree) for x in X])

    def score(self, X, y):
        """Calculate accuracy of predicitions."""
        yPred = self.predict(X)
        return np.mean(yPred == y)


def entropy(labels):
    """Calculate conditional entropy"""
    length = len(labels)
    entropy = 0.0
   
    #Make a dict of the label counts of the data at the current node
    labelCounts = Counter(labels)

    #Entropy = -Summation of labelProbs*log_2(labelProbs)
    for count in labelCounts.values():
        labelProb = count/length
        if labelProb > 0:
            entropy -= labelProb*np.log2(labelProb)
    return entropy


def gini(labels):
    """Calculate Gini impurity"""
    length = len(labels)
    gini = 1.0

    #Make a dict of the label counts of the data at the current node
    labelCounts = Counter(labels)

    #Gini = 1 - Summation of labelProbs**2
    for count in labelCounts.values():
        labelProb = count/length
        gini -= labelProb**2

    return gini