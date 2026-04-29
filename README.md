# How to use our scRNA cell type classifier

Input:
1. annotations file that contains the cell barcode and cell type
2. a tsv of any scRNA dataset

Output: 
1. an acquisition landscape for each model from the Gaussian Process
2. a line graph that shows the CV accuracy for each iteration of the Gaussian Process
3. a confusion matrix alongside the F1 scores of each model

Steps to running our pipeline:
1. Input your dataset into our preprocessing.py file
2. Use the x_pca_coordinates.csv file that is outputted alongside the annotations file for the gaussian process pipeline
3. Set the parameters you want to optimize for and the number of iterations you want to run for gaussian process
4. The model will output optimized parameters based on the training set and will train the final model with these parameters on the test set



