# rotten-tomatoes-prediction
Used gradient boosting regression to predict Rotten Tomatoes scores for movies of various genres. [Dataset](https://www.kaggle.com/stefanoleone992/rotten-tomatoes-movies-and-critics-datasets/download) retrieved from Kaggle, and categorical encoding was done on my own. Will update this soon, but some quick facts: mean absolute error for predicting Tomatometer score was about 19. I was able to reduce it further by increasing the number of estimators, but the training time did take much longer. Films rated NR showed the least mean absolute error, about 12.

[Jump to the script.](https://github.com/kaisubr/rotten-tomatoes-prediction/blob/master/rottentomatoesscorepredictor-kernel45f954f728.py)
