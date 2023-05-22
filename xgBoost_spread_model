library(data.table)
library(zoo)
library(xgboost)
library(ggimage)
library(ggplot2)
library(ggrepel)
library(tidyverse)
library(cfbscrapR)
library(cfbfastR)
library(gridExtra)
library(ggpubr)
library(gt)
library(gtExtras)
library(caret)
library(cvms)
mean <- base::mean

#API key
Sys.setenv(CFBD_API_KEY = "xxxXXXXFake-KeyxxxXXXX")


#get betting data for training from 2017-2020
betting1 <- cfbd_betting_lines(year = 2017)
betting1 <- betting1 %>% 
  filter(provider == "consensus")
betting1$spread_out <- ifelse((betting1$away_score - betting1$home_score) > betting1$spread, 1, 0)

betting2 <- cfbd_betting_lines(year = 2018)
betting2 <- betting2 %>% 
  filter(provider == "consensus")
betting2$spread_out <- ifelse((betting2$away_score - betting2$home_score) > betting2$spread, 1, 0)

betting3 <- cfbd_betting_lines(year = 2019)
betting3 <- betting3 %>% 
  filter(provider == "consensus")
betting3$spread_out <- ifelse((betting3$away_score - betting3$home_score) > betting3$spread, 1, 0)

betting4 <- cfbd_betting_lines(year = 2020)
betting4 <- betting4 %>% 
  filter(provider == "consensus")
betting4$spread_out <- ifelse((betting4$away_score - betting4$home_score) > betting4$spread, 1, 0)

#combine data
betting <- rbind(betting1, betting2, betting3, betting4)
betting <- betting %>% drop_na(spread_out)
betting$actual <- betting$away_score - betting$home_score

#season pbp, have to get seasons 18, 19, 20
pbp <- data.frame()
seasons <- 2017:2020
progressr::with_progress({
  future::plan("multisession")
  pbp <- cfbfastR::load_cfb_pbp(seasons)
})

pbp[is.na(pbp)] <- 0
pbp$fgmade <- ifelse(pbp$fg_made == TRUE, 1, 0)


#home data

temp1 <- pbp %>% 
  group_by(game_id, def_pos_team) %>% 
  summarise(
    week = first(week),
    season = first(season),
    pen_yards = sum(yds_penalty, na.rm = TRUE),
    punty = sum(yds_punted, na.rm = TRUE),
    kicky = sum(yds_kickoff, na.rm = TRUE),
    sacks = sum(sack, na.rm = TRUE),
    epa = sum(EPA, na.rm = TRUE),
    takeaways = sum(turnover, na.rm = TRUE),
  )
temp1$kick_yards <- temp1$punty + temp1$kicky

temp <- pbp %>% 
  group_by(game_id, pos_team) %>% 
  summarise(
    week = first(week),
    season = first(season),
    yards = sum(yards_gained, na.rm = TRUE),
    tds = sum(touchdown, na.rm = TRUE),
    pen_yards = sum(yds_penalty, na.rm = TRUE),
    ptrt = sum(yds_punt_return, na.rm = TRUE),
    kcrt = sum(yds_kickoff_return, na.rm = TRUE),
    ints = sum(interception_stat, na.rm = TRUE),
    fumbs = sum(interception_stat, na.rm = TRUE),
    epa = sum(EPA, na.rm = TRUE),
    rushes = sum(rush, na.rm = TRUE),
    passes = sum(pass, na.rm = TRUE),
    fgs = sum(fgmade, na.rm = TRUE)
  ) %>%
  left_join(temp1, by = c("game_id" = "game_id", "pos_team" = "def_pos_team"))

temp$turnovers <- temp$ints + temp$fumbs
temp$ret_yards <- temp$ptrt + temp$kcrt

temp$pen_yards <- temp$pen_yards.x+temp$pen_yards.y
temp$epa <- temp$epa.x+temp$epa.y

# get home team stats
game_stats_h <- temp %>% 
  arrange(week.x, season.x) %>% 
  group_by(pos_team) %>% 
  summarise(
    game_id = (game_id),
    yards = slider::slide_dbl(yards, mean, .before = 5, .after = -1),
    tds = slider::slide_dbl(tds, mean, .before = 5, .after = -1),
    pen_yards = slider::slide_dbl(pen_yards, mean, .before = 5, .after = -1),
    ret_yards = slider::slide_dbl(ret_yards, mean, .before = 5, .after = -1),
    kick_yards = slider::slide_dbl(kick_yards, mean, .before = 5, .after = -1),
    turnovers = slider::slide_dbl(turnovers, mean, .before = 5, .after = -1),
    sacks = slider::slide_dbl(sacks, mean, .before = 5, .after = -1),
    epa = slider::slide_dbl(epa, mean, .before = 5, .after = -1),
    takeaways = slider::slide_dbl(takeaways, mean, .before = 5, .after = -1),
    rushes = slider::slide_dbl(rushes, mean, .before = 5, .after = -1),
    passes = slider::slide_dbl(passes, mean, .before = 5, .after = -1),
    fgs = slider::slide_dbl(fgs, mean, .before = 5, .after = -1)
  )
game_stats_h$game_id <- as.numeric(game_stats_h$game_id)

one <- game_stats_h[!duplicated(game_stats_h[, c("game_id")]),]

two <- anti_join(game_stats_h, one)



one <- one %>% 
  left_join(two, by = c("game_id" = "game_id"))


#combine the betting and team stats data
betting <- betting %>% 
  left_join(one, by = c("game_id" = "game_id"))

betting <- betting %>% 
  drop_na(spread)

betting$over_under <- as.numeric(betting$over_under)

betting <- betting %>% 
  drop_na(yards.x, yards.y)
betting$avg_points <- 7*(betting$tds.x+betting$tds.y) + 3*(betting$fgs.x+betting$fgs.y)

#isolate the predicted variable and select desired predictive stats
y <- betting$actual
x <- betting %>% select(-game_id, -season, -season_type, -week,
                        -start_date, -home_team, -away_team, -home_conference,
                        -away_conference, -home_score, -away_score, -provider,
                        -formatted_spread, -spread_open, -over_under_open,
                        -home_moneyline, -away_moneyline, -spread_out,
                        -pos_team.x, -pos_team.y, -actual)
x$spread[is.na(x$spread)] <- 0
#x$game_id <- as.numeric(x$game_id)
x$spread <- as.numeric(x$spread)

#set params
params <- list(set.seed = 1502,
               eval_metric = "auc",
               objective = "binary:logistic"
)

#train the model
model <- xgboost(data = as.matrix(x),
                 label = y, 
                 nrounds = 75, #21 75
                 verbose= 1,
                 gamma = .5, #1.75 1.25
                 #maximize = 7
                 eta = .05, #.05
                 max_depth = 8, #8
                 #early_stopping_rounds = 5,
                 #min_child_weight = 5,
                 #nfold = 6,
                 #subsample = .9889945
)
#plot shap to see the top predictive variables
xgb.plot.shap(data = as.matrix(x),
              model = model,
              top_n = 5)



e <- data.frame(model$evaluation_log)
plot(e$iter, e$trainmlogloss, col = "blue")
lines(e$iter, e$testmlogloss, col = "red")



#summary of model to see what is are most useful variables
summary(model)

#GET TEST DATA, same process as getting training data, but only testing on 2022
#adding columns to betting datase
testing <- cfbd_betting_lines(year = 2022)
testing <- testing %>% 
  filter(provider == "Bovada")
testing$actual <- testing$away_score-testing$home_score
#testing$ou_outcome <- ifelse((testing$home_score + testing$away_score) > testing$over_under, 1, 0)
pbp <- data.frame()
seasons <- 2021
progressr::with_progress({
  future::plan("multisession")
  pbp <- cfbfastR::load_cfb_pbp(seasons)
})

pbp[is.na(pbp)] <- 0
pbp$fgmade <- ifelse(pbp$fg_made == TRUE, 1, 0)


#home=====

temp1 <- pbp %>% 
  group_by(game_id, def_pos_team) %>% 
  summarise(
    week = first(wk),
    pen_yards = sum(yds_penalty, na.rm = TRUE),
    punty = sum(yds_punted, na.rm = TRUE),
    kicky = sum(yds_kickoff, na.rm = TRUE),
    sacks = sum(sack, na.rm = TRUE),
    epa = sum(EPA, na.rm = TRUE),
    takeaways = sum(turnover, na.rm = TRUE),
  )
temp1$kick_yards <- temp1$punty + temp1$kicky


temp <- pbp %>% 
  group_by(game_id, pos_team) %>% 
  summarise(
    week = first(wk),
    yards = sum(yards_gained, na.rm = TRUE),
    tds = sum(touchdown, na.rm = TRUE),
    pen_yards = sum(yds_penalty, na.rm = TRUE),
    ptrt = sum(yds_punt_return, na.rm = TRUE),
    kcrt = sum(yds_kickoff_return, na.rm = TRUE),
    ints = sum(int, na.rm = TRUE),
    fumbs = sum(int, na.rm = TRUE),
    epa = sum(EPA, na.rm = TRUE),
    rushes = sum(rush, na.rm = TRUE),
    passes = sum(pass, na.rm = TRUE),
    fgs = sum(fgmade, na.rm = TRUE)
  ) %>%
  left_join(temp1, by = c("game_id" = "game_id", "pos_team" = "def_pos_team"))

temp$turnovers <- temp$ints + temp$fumbs
temp$ret_yards <- temp$ptrt + temp$kcrt

temp$pen_yards <- temp$pen_yards.x+temp$pen_yards.y
temp$epa <- temp$epa.x+temp$epa.y

game_stats_h <- temp %>% 
  arrange(week.x) %>% 
  group_by(pos_team) %>% 
  mutate(tdss = slider::slide_dbl(tds, mean, .before = 5, .after = -1)) %>% 
  summarise(
    game_id = (game_id),
    yards = slider::slide_dbl(yards, mean, .before = 5, .after = -1),
    tds = slider::slide_dbl(tds, mean, .before = 5, .after = -1),
    pen_yards = slider::slide_dbl(pen_yards, mean, .before = 5, .after = -1),
    ret_yards = slider::slide_dbl(ret_yards, mean, .before = 5, .after = -1),
    kick_yards = slider::slide_dbl(kick_yards, mean, .before = 5, .after = -1),
    turnovers = slider::slide_dbl(turnovers, mean, .before = 5, .after = -1),
    sacks = slider::slide_dbl(sacks, mean, .before = 5, .after = -1),
    epa = slider::slide_dbl(epa, mean, .before = 5, .after = -1),
    takeaways = slider::slide_dbl(takeaways, mean, .before = 5, .after = -1),
    rushes = slider::slide_dbl(rushes, mean, .before = 5, .after = -1),
    passes = slider::slide_dbl(passes, mean, .before = 5, .after = -1),
    fgs = slider::slide_dbl(fgs, mean, .before = 5, .after = -1)
  )
game_stats_h$game_id <- as.numeric(game_stats_h$game_id)

one <- game_stats_h[!duplicated(game_stats_h[, c("game_id")]),]

two <- anti_join(game_stats_h, one)



one <- one %>% 
  left_join(two, by = c("game_id" = "game_id"))

testing <- testing %>% 
  left_join(one, by = c("game_id" = "game_id"))

testing <- testing %>% 
  drop_na(over_under)

testing <- testing %>% 
  drop_na(yards.x, yards.y)


testing$over_under <- as.numeric(testing$over_under)
testing$avg_points <- 7*(testing$tds.x+testing$tds.y) + 3*(testing$fgs.x+testing$fgs.y)
testing$spread_out <- ifelse((testing$away_score - testing$home_score) >= testing$spread, 1, 0)

telp <- testing

testing <- telp

#remove the actual values (spread) from the testing set
y <- testing$actual
x <- testing %>% select(-game_id, -season, -season_type, -week,
                        -start_date, -home_team, -away_team, -home_conference,
                        -away_conference, -home_score, -away_score, -provider,
                        -formatted_spread, -spread_open, -over_under_open,
                        -home_moneyline, -away_moneyline, -spread_out,
                        -pos_team.x, -pos_team.y, -actual) #%>% 
#arrange(game_id)

#make predictions
x$spread[is.na(x$spread)] <- 0
#x$game_id <- as.numeric(x$game_id)
x$spread <- as.numeric(x$spread)
#testing <- testing %>% 
# arrange(game_id)
testing$xSpread <- predict(model, as.matrix(x))
testing$xSpread <- round((testing$xSpread),1)
testing$xSpread[testing$xHit > 1] <- 1




#TESTING
#graph accuracy of predictions
x <- as.numeric(names(table(testing$xSpread)))
y <- sapply(x, function(x) mean(testing$actual[testing$xSpread == x]))
data_plot <- data.table(x = x, y = y)
ggplot(data_plot, aes(x = x, y = y)) +
  geom_point(size = 4, color = "dodgerblue4", alpha = .5) + 
  theme_bw() + 
  xlab("Expected Over Hit") + ylab("Over Probability") +
  ylim(c(0, 1)) + xlim(c(0, 1))+
  geom_abline(slope = 1, intercept = 0, color = "brown4", size = 1)+
  labs(title = "Testing the Accuracy of our Predictions")+
  theme(
    plot.title = element_text(size = 14, hjust = 0.5, face = "bold"),
    panel.background = element_rect(fill = "snow1"),
    plot.background = element_rect(fill = "snow1"),
    text = element_text(color = "black", family = "Palatino"),
  )+
  annotate("text", x=.25, y=.81, label= "*Points closer to the line are 
           more accurate predictions", family = "Palatino")

testing$spread <- as.numeric(testing$spread)

testing$correct <- ifelse((testing$xSpread > testing$spread & testing$spread_out == 1) | (testing$xSpread <= testing$spread & testing$spread_out == 0), 1, 0)
percent_bets_correct = sum(testing$correct) / (nrow(testing))
print(percent_bets_correct)

testing$correct_o <- ifelse((testing$xSpread > testing$spread & testing$spread_out == 1), 1, 0)
percent_bets_correct = sum(testing$correct_o) / nrow(testing[testing$xSpread > testing$spread,])
print(percent_bets_correct)

testing$correct_u <- ifelse((testing$xSpread <= testing$spread & testing$spread_out == 0), 1, 0)
percent_bets_correct = sum(testing$correct_u) / nrow(testing[testing$xSpread <= testing$spread,])
print(percent_bets_correct)

#get conf. interval 
confidence_int <- percent_bets_correct + 1.96 * sqrt(percent_bets_correct*(1-percent_bets_correct)/328)
print(confidence_int)

testing$guess <- ifelse(testing$xSpread > testing$spread, 1, 0)

actual <- factor(rep(c(1, 0), times=c(335, 328)))
pred <- factor(rep(c(1, 0, 1, 0), times=c(219, 116, 173, 155)))
cmat <- confusionMatrix(pred, actual, mode = "everything", positive="1")

fourfoldplot(cmat$table, color = c("skyblue1", "sienna3"),
             conf.level = 0, margin = 1, main = "Confusion Matrix")


library(magick)
showtext_auto()
dev.off()

#Visualizing=====
testing$xSpread <- round((testing$xSpread/5))*5
testing$diff <- round(((testing$spread - testing$xSpread)/5))*5
#accuracy looks
x <- as.numeric(names(table(testing$diff)))
y <- sapply(x, function(x) sum(testing$correct[testing$diff == x])/nrow(testing[testing$diff == x, ]))
z <- sapply(x, function(x) nrow(testing[testing$diff == x, ]))
data_plot <- data.table(x = x, y = y, z = z)
ggplot(data_plot, aes(x = x, y = y))+
  geom_bar(stat = "identity", fill = "sienna3", color = "gray7", alpha = .8)+
  labs(title = "Testing the Accuracy of our Predictions",
       caption = "Data from: cfbscrapR | Visual by: Samuel DiSorbo",
       x = "Spread - xSpread",
       y = "Percent Correct")+
  theme(
    plot.title = element_text(size = 14, hjust = 0.5, face = "bold"),
    plot.caption = element_text(hjust=.5),
    panel.background = element_rect(fill = "white"),
    plot.background = element_rect(fill = "white"),
    text = element_text(color = "black", family = "montserrat"),
    panel.grid.major=element_line(colour="gray47"),
    panel.grid.minor=element_line(colour="gray47")
  )+
  ylim(0, 1)+
  geom_text(aes(label=z), vjust=2, face = "bold", family = "montserrat", color = "white")+
  geom_hline(yintercept =.524, color = "skyblue1", alpha = .9, size = 1.5, linetype = "dashed")


#compare pred. densities to vegas and actual densities
ggplot(testing) + 
  geom_density(alpha=.8, fill = "sienna3", aes(x=xSpread))+
  geom_density(alpha=.8, fill = "lightgoldenrod1", aes(x=spread))+
  geom_density(alpha=.8, fill = "skyblue1", aes(x=actual))+
  labs(
    title = "ALICE Spread Totals Density Plot",
    subtitle = "Yellow is the dist. of Vegas totals, orange is ALICE's, blue is Actual",
    x = "Predicted Spread",
    y = "Density"
  )+
  theme_bw()+
  theme(plot.title = element_text(size = 18, hjust = 0.5, face = "bold"),
        plot.subtitle = element_text(size = 14, hjust = 0.5),
        axis.text = element_text(size = 13),
        axis.title = element_text(size = 11),
        panel.background = element_rect(fill = "white"),
        plot.background = element_rect(fill = "white"),
        text = element_text(color = "black", family = "montserrat"),
  )

#prediction accuracy by conference and team
conf = "FBS Independents"
testing_logos <- testing %>% 
  left_join(cfbteams_data, by = c('pos_team.x' = 'school')) %>% 
  left_join(cfbteams_data, by = c('pos_team.y' = 'school')) %>% 
  filter(home_conference == conf | away_conference == conf)
#filter(xHit > .5 | xHit < .4)

home_teams <- testing_logos %>% 
  group_by(pos_team.x) %>% 
  summarise(
    total_bets = n(),
    correct_bets = sum(correct)
  )
away_teams <- testing_logos %>% 
  group_by(pos_team.y) %>% 
  summarise(
    total_bets = n(),
    correct_bets = sum(correct)
  ) 

away_teams <- merge(away_teams, home_teams, by.x = "pos_team.y", by.y = "pos_team.x", all =TRUE)
away_teams[is.na(away_teams)] <- 0
away_teams <- away_teams %>% 
  left_join(cfbteams_data, by = c('pos_team.y' = 'school')) %>% 
  filter(conference == conf)

away_teams$total_bets.y[is.na(away_teams$total_bets.y)] <- 0
away_teams$correct_bets.y[is.na(away_teams$correct_bets.y)] <- 0

away_teams$wp <- (away_teams$correct_bets.x + away_teams$correct_bets.y)/
  (away_teams$total_bets.x + away_teams$total_bets.y)

#plotting those accuracy results
ggplot(away_teams)+
  geom_col(mapping = aes(y = reorder(pos_team.y, wp), x = wp), color = away_teams$alt_color , fill = away_teams$color, width = .15)+
  geom_image(mapping = aes(y = reorder(pos_team.y, wp), x = wp, image = away_teams$logos1, na.rm = T, asp = 16/9), size = .045)+
  geom_vline(xintercept =.524, color = "red", alpha = .3, size = 1.5, linetype = "dashed")+
  theme_get()+
  labs(x = "Percent Bets Correct",
       y = "Team",
       title = glue::glue("ALICE On The {unique(conf)}"),
       subtitle = glue::glue("Avg. Success Rate: {unique(round((mean(away_teams$wp)),2))}"),
       caption = "Data: @cfbscrapR, Graphic By: Samuel DiSorbo"
  )+
  theme_bw()+
  theme(plot.title = element_text(size = 18, hjust = 0.5, face = "bold"),
        plot.subtitle = element_text(size = 14, hjust = 0.5),
        plot.caption = element_text(hjust=.5),
        axis.text = element_text(size = 8),
        axis.title = element_text(size = 12),
        panel.background = element_rect(fill = "white"),
        plot.background = element_rect(fill = "white"),
        text = element_text(color = "black", family = "montserrat"),
        #axis.text.x=element_blank(),
  )





