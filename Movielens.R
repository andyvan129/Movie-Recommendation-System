##########################################################
# Create edx and final_holdout_test sets 
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(dplyr)
library(stringr)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


# Install required packages and enable parallel processing to reduce computing time
packages <- c('parallel', 'doParallel')
for (pack in packages) {
  if (!require(pack, character.only = TRUE)) install.packages(pack)
  library(pack, character.only = TRUE)
}

if (detectCores() > 1){
  num_core <- detectCores() - 1
  pl <- makeCluster(num_core)
  registerDoParallel(pl)
} else {
  registerDoSEQ()
}

# Goal: use a unique userid and movieID pair to predict rating of this movie

# function to process information from raw data to useful predictors
process_data <- function(data){
  
  # Predictor 1: all ratings from user u
  user_avg <- data %>%
    group_by(userId) %>%
    summarise(user_avg = mean(rating))
  
  # Predictor 2: all ratings of movie i
  movie_avg <- data %>%
    group_by(movieId) %>%
    summarise(movie_avg = mean(rating))
  
  # Predictor 3: ratings of movies similar to i
  genres_avg <- data %>%
    group_by(genres) %>%
    summarise(genres_avg = mean(rating))
  
  # Predictor 4: ratings from users similar to u
  
  # create data parameters for training and testing
  modified_data <- data %>%
    left_join(user_avg, by = "userId") %>%
    left_join(movie_avg, by = "movieId") %>%
    left_join(genres_avg, by = "genres") %>%
    select(rating, user_avg, movie_avg, genres_avg)
}

train_data <- process_data(edx)
test_data <- process_data(final_holdout_test)

# use the above predictors to train a model
control <- trainControl(method = "cv", p = 0.75)
fit <- train(rating ~ ., data = train_data, 
             method = "glm", 
             trControl = control, 
             metric = "RMSE")

# function to calculate RMSE
RMSE <- function(y_hat, y){
  sqrt(mean((y_hat - y)^2))
  }

# pull test parameters from final test data
test_parameters <- test_data %>%
  select(-rating)
# pull the correct rating data from final test data, for RMSE calculation
final_rating <- test_data %>%
  pull(rating)

# predict rating using model
rating_hat <- predict(fit, test_parameters)

# calculate RMSE
RMSE(rating_hat, final_rating)