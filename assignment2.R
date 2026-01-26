################################################
#
#     Text Mining for Economics & Finance
#
#                Assignment 2
#
#
################################################

# Run these once, if you haven't installed them before
# install.packages("quanteda")
# install.packages("textclean")
# install.packages("ggrepel")
# install.packages("glmnet")

# Run these every time
library(quanteda)
library(ggrepel)
library(textclean)
library(tidyverse)
library(glmnet)
library(pROC)

# some new functions... each saved in a separate R script

#this is the one from last week
source("TMEF_dfm.R")

# this is kendall accuracy
source("kendall_acc.R")

glassdoor<-readRDS("glassdoorReviews.RDS")

# new data! let's explore

# Only a few companies in this data
glassdoor %>%
  with(table(company))

# Split in categories - some big companies, some smaller ones
glassdoor %>%
  with(table(company,FAANG))

# Other important metadata - Overall rating
glassdoor %>%
  with(hist(overall))

# More exploring.... do companies differ by overall rating?
overall_avgs<-glassdoor %>%
  group_by(company) %>%
  summarize(m=mean(overall),
            se=sd(overall)/sqrt(n())) 

# note how we calculate a standard error above
# it is included through ymin and ymax on line 55

overall_avgs %>%
  ggplot(aes(x=company,color=company,
             y=m,ymin=m-se,ymax=m+se)) +
  geom_point() +
  geom_errorbar(width=.2) +
  theme_bw() +
  coord_flip() + # coord_flip makes the axis labels readable!
  scale_y_continuous(limits = c(3,5)) +
  labs(y="Overall Rating")+
  theme(legend.position="none")

# Let's explore the text.... 

# Note - there are two different text boxes!! "pros" and "cons"
glassdoor <- glassdoor%>%
  mutate(pros_wordcount=str_count(pros,"[[:alpha:]]+"),
         cons_wordcount=str_count(cons,"[[:alpha:]]+"))

# for showing a single continuous variable, we use a histogram
glassdoor %>%
  ggplot(aes(x=pros_wordcount)) +
  geom_histogram(bins = 100) +
  theme_bw() +
  xlim(0,100)

glassdoor %>%
  ggplot(aes(x=cons_wordcount)) +
  geom_histogram(bins = 100) +
  theme_bw() +
  xlim(0,100)

# Let's focus on people who actually wrote text in both boxes

gd_small<-glassdoor %>%
  filter(pros_wordcount>5 & cons_wordcount>5)

dim(gd_small)
# Even that's too big so let's get it down to 40,000 texts

# before we randomize, use set.seed() to all get the same split
set.seed(02138)

# grab the first 40,000 rows after randomizing
gd_small<-gd_small %>%
  arrange(sample(1:n())) %>%
  slice(1:40000)

##############################################################
# split into train and test
train_split=sample(1:nrow(gd_small),20000)

gd_train<-gd_small%>%
  slice(train_split)

gd_test<-gd_small%>%
  slice(-train_split)

##############################################################
# Let's just look at amazon for now

gd_amazon_train<-gd_train %>%
  filter(company=="amazon")

gd_amazon_test<-gd_test %>%
  filter(company=="amazon")

# create our prediction variables from the pros text
dfm_amazon_train_pros<-TMEF_dfm(gd_amazon_train$pros,ngrams=1:2) %>%
  convert(to="matrix")

amazon_train_Y<-gd_amazon_train %>%
  pull(overall)

# Put training data into LASSO model

amazon_model_pros<-cv.glmnet(x=dfm_amazon_train_pros,
                             y=amazon_train_Y)

# check the tuning to see if there is useful information
plot(amazon_model_pros)

##################################################################

# let's apply our model to two test sets

# We need the same X features in the test as in training 

# we use dfm_match() to make sure they are the same features

# First, let's test the model on the pros text from amazon
dfm_amazon_test_pros<-TMEF_dfm(gd_amazon_test$pros,
                               ngrams=1:2,
                               min.prop = 0) %>%
  dfm_match(colnames(dfm_amazon_train_pros)) %>%
  convert(to="matrix")

amazon_test_Y<-gd_amazon_test %>%
  pull(overall)


# generate predictions for test data
amazon_test_predict_pros<-predict(amazon_model_pros,
                                  newx = dfm_amazon_test_pros)[,1]

# check distributions - continuous predictor, continuous outcome
hist(amazon_test_predict_pros)
hist(amazon_test_Y)

# estimate accuracy - use kendall's tau
pros_acc<-kendall_acc(amazon_test_predict_pros,amazon_test_Y)

pros_acc

############################################

# Let's apply the same model to the cons text

dfm_amazon_test_cons<-TMEF_dfm(gd_amazon_test$cons,ngrams=1:2,min.prop = 0)  %>%
  dfm_match(colnames(dfm_amazon_train_pros)) %>%
  convert(to="matrix")

# generate predictions for test data
amazon_test_predict_cons<-predict(amazon_model_pros,
                                  newx = dfm_amazon_test_cons)[,1]

hist(amazon_test_predict_cons)
hist(amazon_test_Y)

# estimate accuracy
cons_acc<-kendall_acc(amazon_test_predict_cons,amazon_test_Y)

# why is accuracy so low?
cons_acc

# Combine accuracy estimates for a plot
bind_rows(pros_acc %>%
            mutate(field="Pros ngrams"),
          cons_acc %>%
            mutate(field="Cons ngrams")) %>%
  ggplot(aes(x=field,color=field,
             y=acc,ymin=lower,ymax=upper)) +
  geom_point() +
  geom_errorbar(width=.4) +
  theme_bw() +
  labs(x="Test Data",y="Accuracy") +
  geom_hline(yintercept = 50) +
  theme(axis.text = element_text(size=24),
        axis.title = element_text(size=24),
        panel.grid=element_blank(),
        legend.position="none")

#################################################

