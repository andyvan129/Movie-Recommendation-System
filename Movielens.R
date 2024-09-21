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


# Rating = mean_of_all + movie_effect + user_effect
#          + mean_of_sim_movies
#          + regularized_movie_rating
#          + regularized_user

# Goal: predict rating of movie i by user u
# Predictor 1: all ratings from user u
user_avg <- edx %>%
  group_by(userId) %>%
  summarise(user_avg = mean(rating))
train_data <- left_join(edx, user_avg, by = "userId") %>%
  select(userId, movieId, rating, genres, user_avg)

# Predictor 2: all ratings of movie i
movie_avg <- edx %>%
  group_by(movieId) %>%
  summarise(movie_avg = mean(rating))
train_data <- left_join(train_data, movie_avg, by = "movieId")

# Predictor 3: ratings of movies similar to i
genres_avg <- edx %>%
  group_by(genres) %>%
  summarise(genres_avg = mean(rating))
train_data <- left_join(train_data, genres_avg, by = "genres") %>%
  select(-genres)

# Predictor 4: ratings from users similar to u

# The goal is to use a unique userid and movieID pair to predict rating of this movie

# calculate RSME
RSME <- function(y_hat, y){
  sqrt(mean((y_hat - y)^2))
  }


# use test set to ensemble models and select criteria
