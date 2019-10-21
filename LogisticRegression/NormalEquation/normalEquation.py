import numpy as np

#Formula: ((XtX)-1 Xty)
def linearNormalEquation(X,y):
	#theta = np.dot(np.dot(np.linalg.pinv(np.dot(X,X.transpose())),X.transpose),y)
	#transposeX = X.transpose()
    #samples,_ = X.shape
    #ones = np.ones((samples,1),dtype='i4')
    #X = np.append(ones,X,axis=1)
    first = np.linalg.pinv(np.dot(X.transpose(),X))
	#first = np.linalg.pinv(first)
    second = np.dot(first,X.transpose())
    theta = np.dot(second,y)
    return theta
	
if __name__ == '__main__':
	print("Usage: Can't be run Independently")