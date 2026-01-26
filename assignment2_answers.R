################################################
#
#    Text Mining for Economics & Finance
#
#                Assignment 2
#                  Answers
#
################################################

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

# Note - there are two different text boxes!! "pros" and "cons"
glassdoor <- glassdoor%>%
  mutate(pros_wordcount=str_count(pros,"[[:alpha:]]+"),
         cons_wordcount=str_count(cons,"[[:alpha:]]+"))

gd_small<-glassdoor %>%
  filter(pros_wordcount>5 & cons_wordcount>5)

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

##################################################################
# Question 1
#################################################

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

# estimate accuracy - use kendall's tau
pros_acc_p<-kendall_acc(amazon_test_predict_pros,amazon_test_Y)

############################################

# Let's apply the same model to the cons text

dfm_amazon_test_cons<-TMEF_dfm(gd_amazon_test$cons,ngrams=1:2)  %>%
  dfm_match(colnames(dfm_amazon_train_pros)) %>%
  convert(to="matrix")

# generate predictions for test data
amazon_test_predict_cons<-predict(amazon_model_pros,
                                  newx = dfm_amazon_test_cons)[,1]

# estimate accuracy
cons_acc_p<-kendall_acc(amazon_test_predict_cons,amazon_test_Y)

#################################################
# Question 2
#################################################

# create our prediction variables from the cons text
dfm_amazon_train_cons<-TMEF_dfm(gd_amazon_train$cons,ngrams=1:2) %>%
  convert(to="matrix")

amazon_model_cons<-cv.glmnet(x=dfm_amazon_train_cons,
                             y=amazon_train_Y)

dfm_amazon_test_cons<-TMEF_dfm(gd_amazon_test$cons,ngrams=1:2)  %>%
  dfm_match(colnames(dfm_amazon_train_cons)) %>%
  convert(to="matrix")

# generate predictions for test data
amazon_test_predict_cons<-predict(amazon_model_cons,
                                  newx = dfm_amazon_test_cons)[,1]
# estimate accuracy
cons_acc_c<-kendall_acc(amazon_test_predict_cons,amazon_test_Y)

dfm_amazon_test_pros<-TMEF_dfm(gd_amazon_test$pros,ngrams=1:2)  %>%
  dfm_match(colnames(dfm_amazon_train_cons)) %>%
  convert(to="matrix")

# generate predictions for test data
amazon_test_predict_pros<-predict(amazon_model_cons,
                                  newx = dfm_amazon_test_pros)[,1]
# estimate accuracy
pros_acc_c<-kendall_acc(amazon_test_predict_pros,amazon_test_Y)

#################################################
# Question 3
#################################################

# Combine accuracy estimates for a plot
bind_rows(pros_acc_p %>%
            mutate(test="Pros",
                   train="Pros Model"),
          cons_acc_p %>%
            mutate(test="Cons",
                   train="Pros Model"),
          pros_acc_c %>%
            mutate(test="Pros",
                   train="Cons Model"),
          cons_acc_c %>%
            mutate(test="Cons",
                   train="Cons Model")) %>%
  ggplot(aes(x=test,color=test,
             y=acc,ymin=lower,ymax=upper)) +
  geom_point() +
  facet_wrap(~train) +
  geom_errorbar(width=.4) +
  theme_bw() +
  labs(x="Test Data",y="Accuracy") +
  geom_hline(yintercept = 50) +
  
  theme(axis.text = element_text(size=24),
        axis.title = element_text(size=24),
        strip.text = element_text(size=24),
        panel.grid=element_blank(),
        strip.background = element_rect(fill="white"),
        legend.position="none")

#################################################
# Question 4
#################################################

# Pros Plot

# extract coefficients
plotCoefs<-amazon_model_pros %>%
  coef() %>%
  drop() %>%
  as.data.frame() %>%
  rownames_to_column(var = "ngram") %>%
  rename(score=".") %>%
  filter(score!=0 & ngram!="(Intercept)" & !is.na(score))  

# merge frequencies
plotDat<-plotCoefs %>%
  left_join(data.frame(ngram=colnames(dfm_amazon_train_pros),
                       freq=colMeans(dfm_amazon_train_pros))) %>%
  mutate_at(vars(score,freq),~round(.,3))

# pipe into ggplot
plotDat %>%
  ggplot(aes(x=score,y=freq,label=ngram,color=score)) +
  scale_color_gradient2(low="navyblue",
                        mid = "grey",
                        high="forestgreen",
                        midpoint = 0)+
  geom_vline(xintercept=0)+
  geom_point() +
  geom_label_repel(max.overlaps = 15)+  
  scale_y_continuous(trans="log2",
                     breaks=c(.01,.05,.1,.2,.5,1,2,5))+
  theme_bw() +
  labs(x="Coefficient in Pros Model",y="Uses per Review")+
  theme(legend.position = "none",
        axis.title=element_text(size=20),
        axis.text=element_text(size=16))

#################
# Cons Plot
#################

# extract coefficients
plotCoefs<-amazon_model_cons %>%
  coef() %>%
  drop() %>%
  as.data.frame() %>%
  rownames_to_column(var = "ngram") %>%
  rename(score=".") %>%
  filter(score!=0 & ngram!="(Intercept)" & !is.na(score))  

# merge frequencies
plotDat<-plotCoefs %>%
  left_join(data.frame(ngram=colnames(dfm_amazon_train_cons),
                       freq=colMeans(dfm_amazon_train_cons))) %>%
  mutate_at(vars(score,freq),~round(.,3))

# pipe into ggplot
plotDat %>%
  ggplot(aes(x=score,y=freq,label=ngram,color=score)) +
  scale_color_gradient2(low="navyblue",
                        mid = "grey",
                        high="forestgreen",
                        midpoint = 0)+
  geom_vline(xintercept=0)+
  geom_point() +
  geom_label_repel(max.overlaps = 15)+  
  scale_y_continuous(trans="log2",
                     breaks=c(.01,.05,.1,.2,.5,1,2,5))+
  theme_bw() +
  labs(x="Coefficient in Cons Model",y="Uses per Review")+
  theme(legend.position = "none",
        axis.title=element_text(size=20),
        axis.text=element_text(size=16))


#################################################
# Question 5
#################################################

# Focus on the "matched" predictions - i.e. when you apply the pros model to the pros dataset, and the cons model to the cons dataset.
# Choose two texts (one for each model) where the model thinks the star rating should be high, but instead it is low. 
# And choose two texts (again, one for each model) where the model thinks the star rating should be low, but instead it is high. 
# Make sure they are not too long (ideally less than 100 words) so you can paste them into your assignment.
# Based on your analysis, what do you think you could do to improve the original model?

############ Find examples

# store predictions in data, calculate accuracy
gd_amazon_test<-gd_amazon_test %>%
  mutate(prediction_cons=amazon_test_predict_cons,
         error_cons=abs(overall-prediction_cons),
         prediction_pros=amazon_test_predict_pros,
         error_pros=abs(overall-prediction_pros),
  )

close_high_pros<-gd_amazon_test %>%
  filter(overall==5 & error_pros<1) %>%
  select(pros,overall,prediction_pros)

close_low_pros<-gd_amazon_test %>%
  filter(overall==1 & error_pros<2) %>%
  select(pros,overall,prediction_pros)

close_high_pros
close_high_pros %>%
  slice(1:2) %>%
  pull(pros)

close_low_pros
close_low_pros %>%
  slice(1:2) %>%
  pull(pros)  


close_high_cons<-gd_amazon_test %>%
  filter(overall==5 & error_cons<1) %>%
  select(cons,overall,prediction_cons)

close_low_cons<-gd_amazon_test %>%
  filter(overall==1 & error_cons<1) %>%
  select(cons,overall,prediction_cons)

close_high_cons
close_high_cons %>%
  slice(1:2) %>%
  pull(cons)

close_low_cons
close_low_cons %>%
  slice(1:2) %>%
  pull(cons)  


#################################################
# Question 6
#################################################

# Let's do bi- and tri-grams

################
################
# Pros text
################
################
dfm_amazon_train_pros<-TMEF_dfm(gd_amazon_train$pros,
                               ngrams=2:3) %>%
  convert(to="matrix")

amazon_model_pros<-cv.glmnet(x=dfm_amazon_train_pros,
                             y=amazon_train_Y)

dfm_amazon_test_pros<-TMEF_dfm(gd_amazon_test$pros,
                              ngrams=2:3, 
                              min.prop = 0)  %>%
  dfm_match(colnames(dfm_amazon_train_pros)) %>%
  convert(to="matrix")

# generate predictions for test data
amazon_test_predict_pros<-predict(amazon_model_pros,
                                  newx = dfm_amazon_test_pros)[,1]
# estimate accuracy
pros_acc_c<-kendall_acc(amazon_test_predict_pros,amazon_test_Y)

pros_acc_c

# Pros Plot

# extract coefficients
plotCoefs<-amazon_model_pros %>%
  coef() %>%
  drop() %>%
  as.data.frame() %>%
  rownames_to_column(var = "ngram") %>%
  rename(score=".") %>%
  filter(score!=0 & ngram!="(Intercept)" & !is.na(score))  

# merge frequencies
plotDat<-plotCoefs %>%
  left_join(data.frame(ngram=colnames(dfm_amazon_train_pros),
                       freq=colMeans(dfm_amazon_train_pros))) %>%
  mutate_at(vars(score,freq),~round(.,3))

# pipe into ggplot
plotDat %>%
  ggplot(aes(x=score,y=freq,label=ngram,color=score)) +
  scale_color_gradient2(low="navyblue",
                        mid = "grey",
                        high="forestgreen",
                        midpoint = 0)+
  geom_vline(xintercept=0)+
  geom_point() +
  geom_label_repel(max.overlaps = 15)+  
  scale_y_continuous(trans="log2",
                     breaks=c(.01,.05,.1,.2,.5,1,2,5))+
  theme_bw() +
  labs(x="Coefficient in Pros Model",y="Uses per Review")+
  theme(legend.position = "none",
        axis.title=element_text(size=20),
        axis.text=element_text(size=16))

################
################
# Cons text
################
################

dfm_amazon_train_cons<-TMEF_dfm(gd_amazon_train$cons,
                               ngrams=2:3) %>%
  convert(to="matrix")

amazon_model_cons<-cv.glmnet(x=dfm_amazon_train_cons,
                             y=amazon_train_Y)

dfm_amazon_test_cons<-TMEF_dfm(gd_amazon_test$cons,
                              ngrams=2:3, 
                              min.prop = 0)  %>%
  dfm_match(colnames(dfm_amazon_train_cons)) %>%
  convert(to="matrix")

# generate predictions for test data
amazon_test_predict_cons<-predict(amazon_model_cons,
                                  newx = dfm_amazon_test_cons)[,1]
# estimate accuracy
cons_acc_c<-kendall_acc(amazon_test_predict_cons,amazon_test_Y)


# extract coefficients
plotCoefs<-amazon_model_cons %>%
  coef() %>%
  drop() %>%
  as.data.frame() %>%
  rownames_to_column(var = "ngram") %>%
  rename(score=".") %>%
  filter(score!=0 & ngram!="(Intercept)" & !is.na(score))  

# merge frequencies
plotDat<-plotCoefs %>%
  left_join(data.frame(ngram=colnames(dfm_amazon_train_cons),
                       freq=colMeans(dfm_amazon_train_cons))) %>%
  mutate_at(vars(score,freq),~round(.,3))

# pipe into ggplot
plotDat %>%
  ggplot(aes(x=score,y=freq,label=ngram,color=score)) +
  scale_color_gradient2(low="navyblue",
                        mid = "grey",
                        high="forestgreen",
                        midpoint = 0)+
  geom_vline(xintercept=0)+
  geom_point() +
  geom_label_repel(max.overlaps = 15)+  
  scale_y_continuous(trans="log2",
                     breaks=c(.01,.05,.1,.2,.5,1,2,5))+
  theme_bw() +
  labs(x="Coefficient in Cons Model",y="Uses per Review")+
  theme(legend.position = "none",
        axis.title=element_text(size=20),
        axis.text=element_text(size=16))


#################################################
# Question 7
#################################################


gd_microsoft_train<-gd_train %>%
  filter(company=="microsoft")

gd_microsoft_test<-gd_test %>%
  filter(company=="microsoft")

microsoft_train_Y<-gd_microsoft_train %>%
  pull(overall)

microsoft_test_Y<-gd_microsoft_test %>%
  pull(overall)


# create our prediction variables from the cons text
dfm_microsoft_train_cons<-TMEF_dfm(gd_microsoft_train$cons,ngrams=1:2) %>%
  convert(to="matrix")

microsoft_model_cons<-cv.glmnet(x=dfm_microsoft_train_cons,
                                y=microsoft_train_Y)

dfm_microsoft_test_cons<-TMEF_dfm(gd_microsoft_test$cons,ngrams=1:2)  %>%
  dfm_match(colnames(dfm_microsoft_train_cons)) %>%
  convert(to="matrix")

# generate predictions for test data
microsoft_test_predict_msft<-predict(microsoft_model_cons,
                                     newx = dfm_microsoft_test_cons)[,1]
# estimate accuracy
cons_acc_msft<-kendall_acc(microsoft_test_predict_msft,microsoft_test_Y)


dfm_microsoft_test_cons<-TMEF_dfm(gd_microsoft_test$cons,ngrams=1:2)  %>%
  dfm_match(colnames(dfm_amazon_train_cons)) %>%
  convert(to="matrix")

# generate predictions for test data
microsoft_test_predict_amzn<-predict(amazon_model_cons,
                                     newx = dfm_microsoft_test_cons)[,1]
# estimate accuracy
cons_acc_amzn<-kendall_acc(microsoft_test_predict_amzn,microsoft_test_Y)


cons_acc_msft

cons_acc_amzn

