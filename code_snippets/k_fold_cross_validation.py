# k-fold Cross-Validation Split

"""A limitation of using the train and test split method is that you get a noisy estimate of
algorithm performance. The k-fold cross-validation method (also called just cross-validation) is
a resampling method that provides a more accurate estimate of algorithm performance.
It does this by first splitting the data into k groups. The algorithm is then trained and
evaluated k times and the performance summarized by taking the mean performance score.
Each group of data is called a fold, hence the name k-fold cross-validation. It works by first
training the algorithm on the k-1 groups of the data and evaluating it on the kth hold-out group
as the test set. This is repeated so that each of the k groups is given an opportunity to be held
out and used as the test set. As such, the value of k should be divisible by the number of rows
in your training dataset, to ensure each of the k groups has the same number of rows.
You should choose a value for k that splits the data into groups with enough rows that each
group is still representative of the original dataset. A good default to use is k=3 for a small
dataset or k=10 for a larger dataset. A quick way to check if the fold sizes are representative is
to calculate summary statistics such as mean and standard deviation and see how much the
values differ from the same statistics on the whole dataset.

fold size = count(rows) / count(folds)
    """

# Example of Creating a Cross Validation Split
from random import seed
from random import randrange
# Split a dataset into k folds
def cross_validation_split(dataset, folds=3):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / folds)
    for _ in range(folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)

    return dataset_split
# test cross validation split
if __name__ == '__main__':
    seed(1)
    dataset = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
    folds = cross_validation_split(dataset, 4)
    print(folds)