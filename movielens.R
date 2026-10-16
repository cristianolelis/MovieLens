# Note on sources:
# - The code that downloads the data and creates the edx and final_holdout_test sets
#   was provided by the course staff (HarvardX PH125.9x).
# - The training and test sets were created from edx following the same approach
#   as the course-provided code.
# - A generative AI tool (Claude, by Anthropic) was used to help with small portions
#   of the exploratory analysis code, which were reviewed and adapted by the author.

##########################################################
# Create edx and final_holdout_test sets
##########################################################

# Note: this process could take a couple of minutes
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(recosystem)) install.packages("recosystem", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(recosystem)
options(pillar.sigfig = 6)   # show more digits when printing tibbles

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 600)

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

##########################################################
# Split edx into training and test sets
##########################################################

# The final_holdout_test set is reserved for the final evaluation only.
# All model development and tuning is done on partitions of edx.

set.seed(42)
test_index <- createDataPartition(edx$rating, times = 1, p = 0.1, list = FALSE)
train_set <- edx[-test_index, ]
temp      <- edx[test_index, ]

# Keep only users and movies in the test set that also appear in the training set
test_set <- temp %>%
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Add the rows removed from the test set back into the training set
removed   <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed)

rm(temp, removed, test_index)

##########################################################
# Evaluation metric
##########################################################

# Root Mean Squared Error between true and predicted ratings
RMSE <- function(true_ratings, predicted_ratings) {
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

##########################################################
# Data exploration
##########################################################

# Overview: number of ratings, unique users and unique movies
edx %>%
  summarize(n_ratings = n(),
            n_users   = n_distinct(userId),
            n_movies  = n_distinct(movieId),
            avg_rating = mean(rating))

# Distribution of ratings
edx %>%
  ggplot(aes(x = rating)) +
  geom_bar(fill = "steelblue") +
  scale_y_continuous(labels = scales::comma) +
  labs(title = "Distribution of ratings",
       x = "Rating", y = "Number of ratings")

# Number of ratings per movie
edx %>%
  count(movieId) %>%
  ggplot(aes(x = n)) +
  geom_histogram(bins = 30, fill = "steelblue", color = "white") +
  scale_x_log10() +
  labs(title = "Number of ratings per movie",
       x = "Number of ratings (log scale)", y = "Number of movies")

# Number of ratings per user
edx %>%
  count(userId) %>%
  ggplot(aes(x = n)) +
  geom_histogram(bins = 30, fill = "steelblue", color = "white") +
  scale_x_log10() +
  labs(title = "Number of ratings per user",
       x = "Number of ratings (log scale)", y = "Number of users")

# Average rating by movie release year
edx %>% 
  mutate(release_year = as.integer(str_extract(title, "(?<=\\()\\d{4}(?=\\)\\s*$)"))) %>% # extracts the year from the title
  group_by(release_year) %>% # one group per release year
  summarize(avg_rating = mean(rating)) %>% # average rating for each year
  ggplot(aes(x = release_year, y = avg_rating)) + 
  geom_point() + 
  labs(title = "Average Rating by Movie Release Year", x = "Release Year", y = "Average Rating")

# Average rating by week of rating
edx %>%
  mutate(rating_week = round_date(as_datetime(timestamp), unit = "week")) %>%  # converts timestamp to a date and rounds to the week
  group_by(rating_week) %>%                                                   # one group per week of rating
  summarize(avg_rating = mean(rating)) %>%                                    # average rating for each week
  ggplot(aes(x = rating_week, y = avg_rating)) +
  geom_point() +
  geom_smooth() +
  labs(title = "Average Rating by Week of Rating", x = "Week of Rating", y = "Average Rating")

# Average rating by genre (faster: aggregate genre combinations first, then split)
edx %>%
  group_by(genres) %>%
  summarize(sum_rating = sum(rating), n = n()) %>%      # total and count per genre combination
  separate_rows(genres, sep = "\\|") %>%                # one row per individual genre
  group_by(genres) %>%
  summarize(avg_rating = sum(sum_rating) / sum(n),      # weighted average rating per genre
            n = sum(n)) %>%
  ggplot(aes(x = reorder(genres, avg_rating), y = avg_rating)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = "Average Rating by Genre", x = "Genre", y = "Average Rating")

##########################################################
# Modeling
##########################################################

# Model 1: overall average
# Predict the same rating (the overall average) for every user and movie
mu <- mean(train_set$rating)                 # Computes the overall average rating using the training set
mu

rmse_avg <- RMSE(test_set$rating, mu)       # Evaluates the model by comparing predictions against true ratings in the test set
rmse_avg

# Table to collect the RMSE of each model
rmse_results <- tibble(method = "Overall average", RMSE = rmse_avg)
rmse_results

# Model 2: movie effect
# b_i = how much each movie's ratings deviate, on average, from the overall average
movie_avgs <- train_set %>%
  group_by(movieId) %>%                      # Groups data by unique movie identifiers
  summarize(b_i = mean(rating - mu))        # Calculates the movie effect deviation from the overall average

# Prediction = overall average + movie effect
pred_movie <- test_set %>%
  left_join(movie_avgs, by = "movieId") %>%  # Merges the movie effect (b_i) into the test set
  mutate(pred = mu + b_i) %>%                # Adds the movie effect to the overall average
  pull(pred)                                 # Extracts the predictions as a vector

rmse_movie <- RMSE(test_set$rating, pred_movie) # Computes the RMSE for the movie effect model
rmse_movie

# Add the result to the table
rmse_results <- bind_rows(rmse_results,
                          tibble(method = "Movie effect model", RMSE = rmse_movie))
rmse_results

# Model 3: movie + user effect
# b_u = how much each user rates above or below what mu + b_i would predict
user_avgs <- train_set %>%
  left_join(movie_avgs, by = "movieId") %>%    # brings b_i into the training set
  group_by(userId) %>%                         # Groups data by unique user identifiers
  summarize(b_u = mean(rating - mu - b_i))    # Computes user effect by subtracting baseline and movie effect

# Prediction = overall average + movie effect + user effect
pred_user <- test_set %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%     # Merges user effect (b_u) into the test set
  mutate(pred = mu + b_i + b_u) %>%           # Predictions combining average, movie and user effects
  pull(pred)                                  # Extracts the predictions as a vector

rmse_user <- RMSE(test_set$rating, pred_user) # Computes the RMSE for the movie + user effect model
rmse_user

# Add the result to the table
rmse_results <- bind_rows(rmse_results,
                          tibble(method = "Movie + user effect model", RMSE = rmse_user))
rmse_results

##########################################################
# Feature engineering
##########################################################

# Feature engineering: add release year and week of rating
train_set <- train_set %>%
  mutate(release_year = as.integer(str_extract(title, "(?<=\\()\\d{4}(?=\\)\\s*$)")),  # year from the title
         rating_week  = round_date(as_datetime(timestamp), unit = "week"))            # week of the rating

test_set <- test_set %>%
  mutate(release_year = as.integer(str_extract(title, "(?<=\\()\\d{4}(?=\\)\\s*$)")),  # year from the title
         rating_week  = round_date(as_datetime(timestamp), unit = "week"))            # week of the rating

# Model 4: movie + user + release year effect
# b_y = remaining deviation shared by movies released in the same year
year_avgs <- train_set %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  group_by(release_year) %>%                         # one group per release year
  summarize(b_y = mean(rating - mu - b_i - b_u))     # calculates year effect deviation from previous effects

pred_year <- test_set %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(year_avgs, by = "release_year") %>%     # merges release year effect (b_y) into the test set
  mutate(pred = mu + b_i + b_u + b_y) %>%           # predictions combining average, movie, user and year effects
  pull(pred)

rmse_year <- RMSE(test_set$rating, pred_year)
rmse_year

rmse_results <- bind_rows(rmse_results,
                          tibble(method = "Movie + user + release year effect model", RMSE = rmse_year))
rmse_results

# Model 5: movie + user + release year + genre effect
# b_g = remaining deviation shared by movies with the same genre combination
genre_avgs <- train_set %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(year_avgs, by = "release_year") %>%       # merges release year effect (b_y) into the training set
  group_by(genres) %>%                                # one group per genre combination
  summarize(b_g = mean(rating - mu - b_i - b_u - b_y)) # calculates genre effect deviation from previous effects

pred_genre <- test_set %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(year_avgs, by = "release_year") %>%      # merges release year effect (b_y) into the test set
  left_join(genre_avgs, by = "genres") %>%           # merges genre combination effect (b_g) into the test set
  mutate(pred = mu + b_i + b_u + b_y + b_g) %>%      # predictions combining average, movie, user, year and genre effects
  pull(pred)

rmse_genre <- RMSE(test_set$rating, pred_genre)
rmse_genre

rmse_results <- bind_rows(rmse_results,
                          tibble(method = "Movie + user + release year + genre effect model", RMSE = rmse_genre))
rmse_results

# Model 6: movie + user + release year + genre + rating date effect
# b_t = remaining deviation shared by ratings made in the same week
date_avgs <- train_set %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(year_avgs, by = "release_year") %>%
  left_join(genre_avgs, by = "genres") %>%           # merges genre combination effect (b_g) into the training set
  group_by(rating_week) %>%                           # one group per rating week
  summarize(b_t = mean(rating - mu - b_i - b_u - b_y - b_g)) # calculates date effect deviation from previous effects

pred_date <- test_set %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(year_avgs, by = "release_year") %>%      # merges release year effect (b_y) into the test set
  left_join(genre_avgs, by = "genres") %>%           # merges genre combination effect (b_g) into the test set
  left_join(date_avgs, by = "rating_week") %>%       # merges rating date effect (b_t) into the test set
  mutate(pred = mu + b_i + b_u + b_y + b_g + coalesce(b_t, 0)) %>% # predictions combining average, previous effects and handled date effect
  pull(pred)

rmse_date <- RMSE(test_set$rating, pred_date)
rmse_date

rmse_results <- bind_rows(rmse_results,
                          tibble(method = "Movie + user + release year + genre + rating date effect model", RMSE = rmse_date))
rmse_results

# Model 7: regularized movie + user + release year + genre + rating date effects
# Each effect = sum of residuals / (number of ratings + lambda)
# Effects based on few ratings are shrunk towards zero
regularized_rmse <- function(lambda) {
  movie_reg <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu) / (n() + lambda))
  
  user_reg <- train_set %>%
    left_join(movie_reg, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_i) / (n() + lambda))
  
  year_reg <- train_set %>%
    left_join(movie_reg, by = "movieId") %>%
    left_join(user_reg, by = "userId") %>%
    group_by(release_year) %>%
    summarize(b_y = sum(rating - mu - b_i - b_u) / (n() + lambda))
  
  genre_reg <- train_set %>%
    left_join(movie_reg, by = "movieId") %>%
    left_join(user_reg, by = "userId") %>%
    left_join(year_reg, by = "release_year") %>%      # merges regularized release year effect (b_y) into the training set
    group_by(genres) %>%                                # one group per genre combination
    summarize(b_g = sum(rating - mu - b_i - b_u - b_y) / (n() + lambda)) # calculates regularized genre effect deviation
  
  date_reg <- train_set %>%
    left_join(movie_reg, by = "movieId") %>%
    left_join(user_reg, by = "userId") %>%
    left_join(year_reg, by = "release_year") %>%
    left_join(genre_reg, by = "genres") %>%           # merges regularized genre combination effect (b_g) into the training set
    group_by(rating_week) %>%                           # one group per rating week
    summarize(b_t = sum(rating - mu - b_i - b_u - b_y - b_g) / (n() + lambda)) # calculates regularized date effect deviation
  
  pred <- test_set %>%
    left_join(movie_reg, by = "movieId") %>%            # merges regularized movie effect (b_i) into the test set
    left_join(user_reg, by = "userId") %>%              # merges regularized user effect (b_u) into the test set
    left_join(year_reg, by = "release_year") %>%        # merges regularized release year effect (b_y) into the test set
    left_join(genre_reg, by = "genres") %>%             # merges regularized genre combination effect (b_g) into the test set
    left_join(date_reg, by = "rating_week") %>%         # merges regularized rating date effect (b_t) into the test set
    mutate(pred = mu + b_i + b_u + b_y + b_g + coalesce(b_t, 0)) %>% # predictions combining average, previous regularized effects and handled date effect
    pull(pred)
  
  RMSE(test_set$rating, pred)
}

# Try several values of lambda and keep the best one
lambdas <- seq(2, 8, 0.5)
rmses <- sapply(lambdas, regularized_rmse)

# Plot RMSE versus lambda
tibble(lambda = lambdas, RMSE = rmses) %>%
  ggplot(aes(x = lambda, y = RMSE)) +
  geom_point() +
  labs(title = "RMSE vs. Lambda (Regularization Tuning)", x = "Lambda", y = "RMSE")

best_lambda <- lambdas[which.min(rmses)]
best_lambda

rmse_reg <- min(rmses)

rmse_results <- bind_rows(rmse_results,
                          tibble(method = "Regularized movie + user + release year + genre + rating date effect model", RMSE = rmse_reg))
rmse_results

##########################################################
# Model 8: matrix factorization on the residuals of Model 7
##########################################################

# Step 1: regularized effects computed once with the best lambda
lambda <- best_lambda

# Regularized movie, user and release year effects (same as inside regularized_rmse)
movie_reg <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu) / (n() + lambda))

user_reg  <- train_set %>%
  left_join(movie_reg, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu - b_i) / (n() + lambda))

year_reg  <- train_set %>%
  left_join(movie_reg, by = "movieId") %>%
  left_join(user_reg, by = "userId") %>%
  group_by(release_year) %>%
  summarize(b_y = sum(rating - mu - b_i - b_u) / (n() + lambda))

genre_reg <- train_set %>%
  left_join(movie_reg, by = "movieId") %>%            # merges regularized movie effect (b_i) into the training set
  left_join(user_reg, by = "userId") %>%              # merges regularized user effect (b_u) into the training set
  left_join(year_reg, by = "release_year") %>%        # merges regularized release year effect (b_y) into the training set
  group_by(genres) %>%                                # one group per genre combination
  summarize(b_g = sum(rating - mu - b_i - b_u - b_y) / (n() + lambda)) # calculates regularized genre effect deviation

date_reg  <- train_set %>%
  left_join(movie_reg, by = "movieId") %>%            # merges regularized movie effect (b_i) into the training set
  left_join(user_reg, by = "userId") %>%              # merges regularized user effect (b_u) into the training set
  left_join(year_reg, by = "release_year") %>%        # merges regularized release year effect (b_y) into the training set
  left_join(genre_reg, by = "genres") %>%             # merges regularized genre combination effect (b_g) into the training set
  group_by(rating_week) %>%                           # one group per rating week
  summarize(b_t = sum(rating - mu - b_i - b_u - b_y - b_g) / (n() + lambda)) # calculates regularized date effect deviation

# Step 2: residuals in the training set (what the bias model does not explain)
train_resid <- train_set %>%
  left_join(movie_reg, by = "movieId") %>%            # merges regularized movie effect (b_i) into the training set
  left_join(user_reg, by = "userId") %>%              # merges regularized user effect (b_u) into the training set
  left_join(year_reg, by = "release_year") %>%        # merges regularized release year effect (b_y) into the training set
  left_join(genre_reg, by = "genres") %>%             # merges regularized genre combination effect (b_g) into the training set
  left_join(date_reg, by = "rating_week") %>%         # merges regularized rating date effect (b_t) into the training set
  mutate(resid = rating - (mu + b_i + b_u + b_y + b_g + b_t)) %>% # residual = rating minus the Model 7 prediction
  select(userId, movieId, resid)

# Step 3: Model 7 predictions on the test set (will be combined with the factorization)
pred_bias <- test_set %>%
  left_join(movie_reg, by = "movieId") %>%            # merges regularized movie effect (b_i) into the test set
  left_join(user_reg, by = "userId") %>%              # merges regularized user effect (b_u) into the test set
  left_join(year_reg, by = "release_year") %>%        # merges regularized release year effect (b_y) into the test set
  left_join(genre_reg, by = "genres") %>%             # merges regularized genre combination effect (b_g) into the test set
  left_join(date_reg, by = "rating_week") %>%         # merges regularized rating date effect (b_t) into the test set
  mutate(pred = mu + b_i + b_u + b_y + b_g + coalesce(b_t, 0)) %>% # predictions combining average, previous regularized effects and handled date effect
  pull(pred)

# Check: should reproduce the Model 7 RMSE shown above
RMSE(test_set$rating, pred_bias)

# Step 4: matrix factorization of the residuals with recosystem
set.seed(123)   # the factorization starts from random values; the seed makes results reproducible

# Convert data to recosystem format (user, movie and the value to learn)
train_reco <- with(train_resid, data_memory(user_index = userId, item_index = movieId,
                                            rating = resid, index1 = TRUE))
test_reco  <- with(test_set, data_memory(user_index = userId, item_index = movieId,
                                         index1 = TRUE))

reco <- Reco()

# Tune hyperparameters with cross-validation on the training residuals
# (reduced grid and 3 folds to keep computation time manageable)
opts <- reco$tune(train_reco, opts = list(dim = c(20, 30),          # number of latent factors
                                          lrate = c(0.1),           # learning rate
                                          costp_l2 = c(0.01),       # regularization of user factors
                                          costq_l2 = c(0.01, 0.1),  # regularization of movie factors
                                          nfold = 3,                # number of cross-validation folds
                                          nthread = 2, niter = 10)) # number of CPU threads and iterations
opts$min        # best combination found

# Train the final factorization with the best hyperparameters
reco$train(train_reco, opts = c(opts$min, nthread = 2, niter = 30)) # number of CPU threads and iterations

# Predict the residuals for the test set
pred_resid <- reco$predict(test_reco, out_memory())

# Final prediction = Model 7 prediction + predicted residual
pred_mf <- pred_bias + pred_resid                                    # combines regularized baseline biases with matrix factorization residuals

# Ratings must be between 0.5 and 5: limit predictions to this range
pred_mf <- pmin(pmax(pred_mf, 0.5), 5.0)                             # caps predictions within the valid MovieLens rating bounds

rmse_mf <- RMSE(test_set$rating, pred_mf)
rmse_mf

rmse_results <- bind_rows(rmse_results,
                          tibble(method = "Regularized effects + matrix factorization model", RMSE = rmse_mf)) # adds result to the summary table
rmse_results

##########################################################
# Final model: retrain on the full edx set and evaluate on final_holdout_test
##########################################################
# All hyperparameters (lambda and factorization settings) were chosen using edx only.
# final_holdout_test is used ONLY here, once, to report the final RMSE.
best_opts <- opts$min      # best factorization hyperparameters found by tuning

# Add release year and week of rating to both sets
edx <- edx %>%
  mutate(release_year = as.integer(str_extract(title, "(?<=\\()\\d{4}(?=\\)\\s*$)")),
         rating_week  = round_date(as_datetime(timestamp), unit = "week"))

final_holdout_test <- final_holdout_test %>%
  mutate(release_year = as.integer(str_extract(title, "(?<=\\()\\d{4}(?=\\)\\s*$)")),
         rating_week  = round_date(as_datetime(timestamp), unit = "week"))

# Regularized effects estimated on the full edx set
# (same steps as Model 8, now using all of edx instead of train_set)
mu <- mean(edx$rating)
lambda <- best_lambda

movie_reg <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu) / (n() + lambda))

user_reg  <- edx %>%
  left_join(movie_reg, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu - b_i) / (n() + lambda))

year_reg  <- edx %>%
  left_join(movie_reg, by = "movieId") %>%
  left_join(user_reg, by = "userId") %>%
  group_by(release_year) %>%
  summarize(b_y = sum(rating - mu - b_i - b_u) / (n() + lambda))

genre_reg <- edx %>%
  left_join(movie_reg, by = "movieId") %>%
  left_join(user_reg, by = "userId") %>%
  left_join(year_reg, by = "release_year") %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - mu - b_i - b_u - b_y) / (n() + lambda))

date_reg  <- edx %>%
  left_join(movie_reg, by = "movieId") %>%
  left_join(user_reg, by = "userId") %>%
  left_join(year_reg, by = "release_year") %>%
  left_join(genre_reg, by = "genres") %>%
  group_by(rating_week) %>%
  summarize(b_t = sum(rating - mu - b_i - b_u - b_y - b_g) / (n() + lambda))

# Residuals on edx
edx_resid <- edx %>%
  left_join(movie_reg, by = "movieId") %>%
  left_join(user_reg, by = "userId") %>%
  left_join(year_reg, by = "release_year") %>%
  left_join(genre_reg, by = "genres") %>%
  left_join(date_reg, by = "rating_week") %>%
  mutate(resid = rating - (mu + b_i + b_u + b_y + b_g + b_t)) %>%
  select(userId, movieId, resid)

# Bias model predictions on final_holdout_test
pred_bias_final <- final_holdout_test %>%
  left_join(movie_reg, by = "movieId") %>%
  left_join(user_reg, by = "userId") %>%
  left_join(year_reg, by = "release_year") %>%
  left_join(genre_reg, by = "genres") %>%
  left_join(date_reg, by = "rating_week") %>%
  mutate(pred = mu + b_i + b_u + b_y + b_g + coalesce(b_t, 0)) %>%
  pull(pred)

# Matrix factorization of the residuals, trained on the full edx set
set.seed(123)
edx_reco     <- with(edx_resid, data_memory(user_index = userId, item_index = movieId,
                                            rating = resid, index1 = TRUE))
holdout_reco <- with(final_holdout_test, data_memory(user_index = userId, item_index = movieId,
                                                     index1 = TRUE))
reco_final <- Reco()
reco_final$train(edx_reco, opts = c(best_opts, nthread = 2, niter = 30))
pred_resid_final <- reco_final$predict(holdout_reco, out_memory())

# Final predictions, limited to the valid rating range
pred_final <- pmin(pmax(pred_bias_final + pred_resid_final, 0.5), 5)

# FINAL RMSE
rmse_final <- RMSE(final_holdout_test$rating, pred_final)
print(rmse_final)

# Add final result to the table
rmse_results <- bind_rows(rmse_results,
                          tibble(method = "Final model (trained on edx, evaluated on final_holdout_test)",
                                 RMSE = rmse_final))
print(rmse_results)

##########################################################
# Save results for the report
##########################################################
# The report (Rmd) loads these small summary tables instead of re-running the whole analysis

report_data <- list(
  overview = edx %>%
    summarize(n_ratings = n(), n_users = n_distinct(userId),
              n_movies = n_distinct(movieId), avg_rating = mean(rating)),
  rating_counts     = edx %>% count(rating),
  ratings_per_movie = edx %>% count(movieId),
  ratings_per_user  = edx %>% count(userId),
  year_avgs = edx %>%
    group_by(release_year) %>%
    summarize(avg_rating = mean(rating), n = n()),
  week_avgs = edx %>%
    group_by(rating_week) %>%
    summarize(avg_rating = mean(rating), n = n()),
  genre_avgs = edx %>%
    group_by(genres) %>%
    summarize(sum_rating = sum(rating), n = n()) %>%
    separate_rows(genres, sep = "\\|") %>%
    group_by(genres) %>%
    summarize(avg_rating = sum(sum_rating) / sum(n), n = sum(n)),
  lambda_tuning = tibble(lambda = lambdas, RMSE = rmses),
  best_lambda   = best_lambda,
  best_opts     = best_opts,
  rmse_results  = rmse_results,
  rmse_final    = rmse_final
)

saveRDS(report_data, "movielens_results.rds")