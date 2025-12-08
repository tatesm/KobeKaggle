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

kobe_raw <- vroom("kobe.csv")  # or the Kaggle path
glimpse(kobe_raw)

#############################
## Train / test split
#############################

# shot_made_flag is NA for rows you must predict (test)
kobe <- kobe_raw %>%
  mutate(
    shot_made_flag = as.factor(shot_made_flag)  # levels "0" / "1"
  )

kobe_train <- kobe %>%
  filter(!is.na(shot_made_flag))

kobe_test <- kobe %>%
  filter(is.na(shot_made_flag))

#############################
## Feature Engineering (with interactions)
#############################

fe_engineer <- function(df) {
  df %>%
    mutate(
      # ---- Geometry -----------------------------------------------------
      distance_hyp = sqrt(loc_x^2 + loc_y^2),
      angle_deg    = atan2(loc_x, pmax(loc_y, 1e-3)) * 180 / pi,
      angle        = atan2(loc_y, loc_x),  # radians, alt. angle
      
      angle_abs = abs(angle_deg),
      distance_sq = distance_hyp^2,
      
      # positional bins
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
        angle_deg,
        breaks = c(-180, -67.5, -22.5, 22.5, 67.5, 180),
        labels = c("left_corner", "left_wing", "center", "right_wing", "right_corner")
      ),
      
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
      
      total_game_seconds = (pmax(period - 1, 0) * 12 * 60) + (12 * 60 - time_remaining),
      
      clutch_period_last_min = if_else(time_remaining <= 60, 1L, 0L),
      clutch_game_last2      = if_else(period >= 4 & time_remaining <= 120, 1L, 0L),
      
      # rough “pressure” proxy
      pressure_proxy = case_when(
        loc_y < 30  ~ "heavy",
        loc_y < 100 ~ "medium",
        TRUE        ~ "light"
      ),
      
      # ---- Season & shot type -------------------------------------------
      game_date = mdy(game_date),
      game_year  = year(game_date),
      game_month = month(game_date),
      game_dow   = wday(game_date, label = TRUE),
      
      season_start = as.integer(str_sub(season, 1, 4)),
      season_index = season_start - min(season_start, na.rm = TRUE),
      
      is_three = if_else(shot_type == "3PT Field Goal", 1L, 0L),
      
      # factors
      playoffs       = as.factor(playoffs),
      side           = factor(side),
      distance_bin   = factor(distance_bin),
      angle_bin      = factor(angle_bin),
      loc_x_bin      = factor(loc_x_bin),
      loc_y_bin      = factor(loc_y_bin),
      pressure_proxy = factor(pressure_proxy),
      
      # ---- NEW interaction features XGBoost loves -----------------------
      dist_angle   = distance_hyp * angle_abs,          # distance * angle magnitude
      three_angle  = is_three     * angle_abs,          # 3pt * angle magnitude
      clutch_three = clutch_game_last2 * is_three       # late-game 3s
    )
}

#############################
## Simplified Hot-hand
#############################

add_hot_hand <- function(df) {
  df %>%
    arrange(game_id, total_game_seconds) %>%
    group_by(game_id) %>%
    mutate(
      made_num = if_else(
        is.na(shot_made_flag),
        0,
        as.numeric(as.character(shot_made_flag))
      ),
      
      prev_make = dplyr::lag(made_num, 1, default = 0),
      
      makes_last3 = prev_make +
        dplyr::lag(made_num, 2, default = 0) +
        dplyr::lag(made_num, 3, default = 0),
      
      attempts_last3 = pmin(row_number() - 1L, 3L),
      
      fgpct_last3 = if_else(
        attempts_last3 > 0,
        makes_last3 / attempts_last3,
        0
      )
    ) %>%
    ungroup() %>%
    # keep only the simplified hot-hand features
    select(-made_num, -makes_last3, -attempts_last3)
}

#############################
## Build engineered train/test
#############################

kobe_train_fe <- kobe_train %>%
  fe_engineer() %>%
  add_hot_hand()

kobe_test_fe <- kobe_test %>%
  fe_engineer() %>%
  add_hot_hand()

glimpse(kobe_train_fe[, c(
  "distance_hyp","angle_abs","dist_angle",
  "clutch_game_last2","is_three","three_angle","clutch_three",
  "prev_make","fgpct_last3"
)])

#############################
## Recipe (tree/XGB friendly)
#############################

shot_rec <- recipe(shot_made_flag ~ ., data = kobe_train_fe) %>%
  update_role(shot_id, new_role = "id") %>%
  # drop some IDs / redundant columns
  step_rm(game_event_id, team_id, team_name, game_date, lat, lon) %>%
  # handle novel & unknown factor levels
  step_novel(all_nominal_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  # optionally keep step_other; you can remove it if you want all levels
  step_other(all_nominal_predictors(), threshold = 0.01) %>%
  # one-hot encode
  step_dummy(all_nominal_predictors()) %>%
  # drop zero-variance columns
  step_zv(all_predictors())
# NOTE: no step_normalize() for trees/XGB

#############################
## Stratified Resampling (no grouping)
#############################

set.seed(123)
folds <- vfold_cv(
  kobe_train_fe,
  v      = 5,                 # or 10 if you want more stable CV
  strata = shot_made_flag     # keep class balance across folds
)

#############################
## Metrics
#############################

metric_funs <- metric_set(mn_log_loss, roc_auc, accuracy)

#############################
## XGBoost specification + workflow (richer grid)
#############################

xgb_spec <- boost_tree(
  trees        = tune(),  # total trees
  tree_depth   = tune(),  # max_depth
  min_n        = tune(),  # min_child_weight-ish
  loss_reduction = tune(),
  sample_size  = tune(),  # subsample
  mtry         = tune(),  # colsample_bytree
  learn_rate   = tune()
) %>%
  set_mode("classification") %>%
  set_engine("xgboost", eval_metric = "logloss")

xgb_wf <- workflow() %>%
  add_model(xgb_spec) %>%
  add_recipe(shot_rec)

#############################
## XGBoost hyperparameter grid (slightly bigger)
#############################

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
    mtry        = mtry(c(10L, 60L)),          # ~colsample_bytree
    trees       = trees(c(800L, 2200L)),
    tree_depth  = tree_depth(c(3L, 10L)),     # max_depth
    min_n       = min_n(c(1L, 20L)),         # min_child_weight-ish
    sample_size = sample_prop(c(0.6, 1.0)),  # subsample
    learn_rate  = learn_rate(c(-3, -1.0))    # ≈ 0.001–0.1
  )

set.seed(123)
xgb_grid <- grid_latin_hypercube(
  xgb_params,
  size = 40   # was 30; slightly bigger
)

#############################
## XGBoost tuning
#############################

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
## Select best XGBoost model
#############################

best_xgb <- select_best(xgb_res, metric = "mn_log_loss")
best_xgb

final_xgb_wf <- finalize_workflow(xgb_wf, best_xgb)

#############################
## Fit final XGBoost on all training data
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
  transmute(
    shot_id,
    shot_made_flag = .pred_1
  )

head(submission)

write_csv(submission, "kobe_xgb_interactions_groupcv_submission.csv")
