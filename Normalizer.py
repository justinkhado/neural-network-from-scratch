class Normalizer(object):
    '''
    A preprocessing class that normalizes datasets
    '''
    def __init__(self):
        self.max = 0
        self.min = 0

    def fit(self, X):
        '''
        Returns the min and max of X, respectively
        '''
        self.max = X.max()
        self.min = X.min()

    def transform(self, X):
        '''
        Returns a matrix of the normalized values of X
        '''
        return (X - self.min) / (self.max - self.min)
        