#############################
## Libraries
#############################

library(tidyverse)
library(vroom)
library(lubridate)

library(tidymodels)
tidymodels_prefer()

set.seed(123)

#############################
## Read data
#############################

# Change the path/filename if needed
kobe_raw <- vroom("kobe.csv")  # or "kobe-bryant-shot-selection/data.csv"

glimpse(kobe_raw)

#############################
## Train / test split
#############################

# In this competition, train & test are in the same file:
# shot_made_flag is NA for rows you must predict (test)
kobe <- kobe_raw %>%
  mutate(
    shot_made_flag = as.factor(shot_made_flag)  # 0 / 1 as factor
  )

kobe_train <- kobe %>%
  filter(!is.na(shot_made_flag))

kobe_test <- kobe %>%
  filter(is.na(shot_made_flag))

#############################
## Feature Engineering
#############################

# Basket is at (0,0) in Kaggle's coordinate system.
# We create:
#  - distance_hyp: Euclidean distance from basket
#  - angle: horizontal angle of shot
#  - home: home vs away from "matchup"
#  - time_remaining: seconds remaining in the period
#  - total_game_seconds: seconds elapsed in the game
#  - date features: year, month, day of week
#  - playoffs as factor

fe_engineer <- function(df) {
  df %>%
    mutate(
      # ---- Geometry -----------------------------------------------------
      distance_hyp = sqrt(loc_x^2 + loc_y^2),
      
      angle = atan2(loc_x, pmax(loc_y, 1e-3)) * 180 / pi,
      
      side = case_when(
        loc_x <= -80 ~ "left",
        loc_x >=  80 ~ "right",
        TRUE         ~ "center"
      ),
      
      distance_bin = cut(
        distance_hyp,
        breaks = c(-Inf, 5, 12, 22, 30, Inf),
        labels = c("0-5", "5-12", "12-22", "22-30", "30+")
      ),
      
      angle_bin = cut(
        angle,
        breaks = c(-180, -67.5, -22.5, 22.5, 67.5, 180),
        labels = c("left_corner", "left_wing", "center", "right_wing", "right_corner")
      ),
      
      # smoother angle encodings
      angle_sin = sin(angle * pi / 180),
      angle_cos = cos(angle * pi / 180),
      
      # extra nonlinearity
      angle_abs   = abs(angle),
      distance_sq = distance_hyp^2,
      
      # positional bins
      loc_x_bin = cut(
        loc_x,
        breaks = c(-250, -150, -50, 50, 150, 250),
        labels = c("far_left","left","center","right","far_right")
      ),
      
      loc_y_bin = cut(
        loc_y,
        breaks = c(-50, 0, 80, 160, 250, 300),
        labels = c("behind","paint","mid_paint","mid_range","long_mid")
      ),
      
      # ---- Time & game context ------------------------------------------
      home = if_else(str_detect(matchup, "vs\\."), 1L, 0L),
      
      time_remaining = minutes_remaining * 60 + seconds_remaining,
      
      total_game_seconds = (pmax(period - 1, 0) * 12 * 60) +
        (12 * 60 - time_remaining),
      
      clutch_period_last_min = if_else(time_remaining <= 60, 1L, 0L),
      clutch_game_last2      = if_else(period >= 4 & time_remaining <= 120, 1L, 0L),
      
      pressure_proxy = case_when(
        loc_y < 30  ~ "heavy",
        loc_y < 100 ~ "medium",
        TRUE        ~ "light"
      ),
      
      # ---- Season & shot type -------------------------------------------
      game_date    = mdy(game_date),
      game_year    = year(game_date),
      game_month   = month(game_date),
      game_dow     = wday(game_date, label = TRUE),
      
      season_start = as.integer(str_sub(season, 1, 4)),
      season_index = season_start - min(season_start, na.rm = TRUE),
      
      is_three = if_else(shot_type == "3PT Field Goal", 1L, 0L),
      
      playoffs       = as.factor(playoffs),
      side           = factor(side),
      distance_bin   = factor(distance_bin),
      angle_bin      = factor(angle_bin),
      loc_x_bin      = factor(loc_x_bin),
      loc_y_bin      = factor(loc_y_bin),
      pressure_proxy = factor(pressure_proxy),
      
      # ---- Safe interaction-style features (no label leakage) ----------
      dist_angle   = distance_hyp * angle_abs,        # distance * |angle|
      three_angle  = is_three     * angle_abs,        # 3pt * |angle|
      clutch_three = clutch_game_last2 * is_three     # late-game 3s
    )
}


kobe_train_fe <- fe_engineer(kobe_train)
kobe_test_fe  <- fe_engineer(kobe_test)


glimpse(kobe_train_fe[, c("distance_hyp","distance_bin","angle","angle_bin",
                          "clutch_period_last_min","clutch_game_last2",
                          "season_index","is_three")])





#############################
## Recipe
#############################

# We keep shot_id as an ID role (not used by model, used for submission)
# Remove some clearly ID-like or redundant fields.

shot_rec <- recipe(shot_made_flag ~ ., data = kobe_train_fe) %>%
  update_role(shot_id, new_role = "id") %>%
  step_rm(game_event_id, game_id, team_id, team_name) %>%
  step_rm(game_date) %>%
  step_nzv(all_predictors()) %>%
  step_novel(all_nominal_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_other(all_nominal_predictors(), threshold = 0.01) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())


#############################
## Resampling
#############################

set.seed(123)
folds <- vfold_cv(kobe_train_fe, v = 5, strata = shot_made_flag)

#############################
## Model specification (XGBoost)
#############################

xgb_spec <- boost_tree(
  trees = tune(),
  tree_depth = tune(),
  min_n = tune(),
  loss_reduction = tune(),      # gamma
  sample_size = tune(),         # subsample
  mtry = tune(),
  learn_rate = tune()
) %>%
  set_mode("classification") %>%
  set_engine("xgboost")

#############################
## Workflow
#############################

xgb_wf <- workflow() %>%
  add_model(xgb_spec) %>%
  add_recipe(shot_rec)

#############################
## Hyperparameter grid
#############################

#############################
## Hyperparameter grid
#############################

# Build a parameter set for the xgboost model
xgb_params <- parameters(
  mtry(),
  trees(),
  tree_depth(),
  min_n(),
  loss_reduction(),
  sample_size = sample_prop(),
  learn_rate()
) %>%
  update(
    mtry        = mtry(c(15L, 35L)),
    trees       = trees(c(800L, 2000L)),
    tree_depth  = tree_depth(c(4L, 10L)),
    min_n       = min_n(c(1L, 10L)),
    sample_size = sample_prop(c(0.7, 1.0)),
    # keep learn_rate in a sensible small range, around your best
    learn_rate  = learn_rate(c(-3, -1.5))  # log10 scale ≈ 0.001–0.03
  )

set.seed(123)
xgb_grid <- grid_latin_hypercube(
  xgb_params,
  size = 40
)





#############################
## Tuning
#############################

set.seed(123)
folds <- vfold_cv(kobe_train_fe, v = 5, strata = shot_made_flag)

xgb_wf <- workflow() %>%
  add_model(xgb_spec) %>%
  add_recipe(shot_rec)

metric_funs <- metric_set(mn_log_loss, roc_auc, accuracy)

set.seed(123)
xgb_res <- tune_grid(
  xgb_wf,
  resamples = folds,
  grid      = xgb_grid,
  metrics   = metric_funs,
  control   = control_grid(save_pred = TRUE)
)

collect_metrics(xgb_res) %>%
  filter(.metric == "mn_log_loss") %>%
  arrange(mean) %>%
  head(10)



#############################
## Select best model by log loss
#############################

best_xgb <- select_best(xgb_res, metric = "mn_log_loss")
best_xgb


final_xgb_wf <- finalize_workflow(xgb_wf, best_xgb)

#############################
## Fit final model on all training data
#############################

final_xgb_fit <- final_xgb_wf %>%
  fit(data = kobe_train_fe)

#############################
## Generate predictions for test set
#############################

test_probs <- predict(
  final_xgb_fit,
  new_data = kobe_test_fe,
  type = "prob"
)

head(test_probs)

#############################
## Build submission
#############################

submission <- kobe_test_fe %>%
  select(shot_id) %>%
  bind_cols(test_probs) %>%
  # .pred_0 = P(shot_made_flag == "0"), .pred_1 = P(shot_made_flag == "1")
  transmute(
    shot_id,
    shot_made_flag = .pred_1
  )

head(submission)

write_csv(submission, "kobe_xgbv1_submission.csv")











