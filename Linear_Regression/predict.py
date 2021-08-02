#The first step is to develop a function that can make predictions. This will be needed both
#in the evaluation of candidate coefficient values in stochastic gradient descent and after the
#model is finalized and we wish to start making predictions on test data or new data. 

# Make a prediction with coefficients
def predict(row, coefficients):
    yhat = coefficients[0]
    for i in range(len(row)-1):
        yhat += coefficients[i + 1] * row[i]
    return yhat

if __name__ == '__main__':

    """There is a single input value (x) and two coefficient values (b0 and b1).
    The prediction equation we have modeled for this problem is:
    y = b0 + b1 × x
    Or, with the specific coefficient values we chose by hand as:
    y = 0.4 + 0.8 × x
    """

    dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
    coef = [0.4, 0.8]
    for row in dataset:
        yhat = predict(row, coef)
        print("Expected=%.3f, Predicted=%.3f" % (row[-1], yhat))