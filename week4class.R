################################################
#
#     Text Mining for Economics and Finance
#
#          In-Class Activity - Week 4
#
#
################################################

# one new package this week
#install.packages("sentimentr") # for sentiment

library(quanteda)
library(ggrepel)
library(textclean)
library(tidyverse)
library(glmnet)
library(pROC)
library(sentimentr)

source("TMEF_dfm.R")
source("kendall_acc.R")

# Review data
rev_med<-readRDS("rev_med.RDS")

# Train model

train_split=sample(1:nrow(rev_med),7000)

rev_med_train<-rev_med[train_split,]
rev_med_test<-rev_med[-train_split,]


rev_med_dfm_train<-TMEF_dfm(rev_med_train$text,ngrams=1)

rev_med_dfm_test<-TMEF_dfm(rev_med_test$text,
                           ngrams=1,
                           min.prop=0) %>%
  dfm_match(colnames(rev_med_dfm_train))


rev_model<-glmnet::cv.glmnet(x=rev_med_dfm_train %>%
                               as.matrix(),
                             y=rev_med_train$stars)

plot(rev_model)

#### Interpret with a coefficient plot
plotDat<-rev_model %>%
  coef() %>%
  drop() %>%
  as.data.frame() %>%
  rownames_to_column(var = "ngram") %>%
  rename(score=".") %>%
  filter(score!=0 & ngram!="(Intercept)" & !is.na(score))  %>%
  # add ngram frequencies for plotting
  left_join(data.frame(ngram=colnames(rev_med_dfm_train),
                       freq=colMeans(rev_med_dfm_train)))

plotDat %>%
  mutate_at(vars(score,freq),~round(.,3))

plotDat %>%
  ggplot(aes(x=score,y=freq,label=ngram,color=score)) +
  scale_color_gradient(low="blue",high="red")+
  geom_vline(xintercept=0)+
  geom_point() +
  geom_label_repel(max.overlaps = 30,force = 6)+  
  scale_y_continuous(trans="log2",
                     breaks=c(.01,.05,.1,.2,.5,1,2,5))+
  theme_bw() +
  labs(x="Coefficient in Model",y="Uses per Review")+
  theme(legend.position = "none",
        axis.title=element_text(size=20),
        axis.text=element_text(size=16))


#### Evaluate Accuracy
test_ngram_predict<-predict(rev_model,
                            newx = rev_med_dfm_test %>%
                              as.matrix())[,1]

acc_ngram<-kendall_acc(rev_med_test$stars,test_ngram_predict)

acc_ngram

############ Find examples

# store predictions in data, calculate accuracy
rev_med_test<-rev_med_test %>%
  mutate(prediction=test_ngram_predict,
         error=abs(stars-prediction),
         bias=stars-prediction)

close_high<-rev_med_test %>%
  filter(stars==5 & error<.5) %>%
  select(text,stars,prediction)

close_low<-rev_med_test %>%
  filter(stars==1 & error<.5) %>%
  select(text,stars,prediction)

close_high
close_high %>%
  slice(1:2) %>%
  pull(text)

close_low
close_low %>%
  slice(1:2) %>%
  pull(text)

# Error analysis - find biggest misses

rev_med_test %>%
  ggplot(aes(x=prediction)) +
  geom_histogram()

rev_med_test %>%
  ggplot(aes(x=stars)) +
  geom_histogram()

miss_high<-rev_med_test %>%
  arrange(bias) %>%
  slice(1:10) %>%
  select(text,stars,prediction)

miss_low<-rev_med_test %>%
  arrange(-bias) %>%
  slice(1:10) %>%
  select(text,stars,prediction)

miss_low
miss_low%>%
  slice(1:2) %>%
  pull(text)
miss_high
miss_high%>%
  slice(3) %>%
  pull(text)


#### Evaluate Accuracy
test_ngram_predict<-predict(rev_model,
                            newx = rev_med_dfm_test %>%
                              as.matrix())[,1]

acc_ngram<-kendall_acc(rev_med_test$stars,test_ngram_predict)

acc_ngram


############### Benchmarks

# Create benchmarks

rev_med_test <- rev_med_test %>%
  mutate(text_wdct=str_count(text,"[[:alpha:]]+"),
         model_random=sample(test_ngram_predict))

acc_wdct<-kendall_acc(rev_med_test$stars,
                      rev_med_test$text_wdct)

acc_wdct



acc_random<-kendall_acc(rev_med_test$stars,
                      rev_med_test$model_random)

acc_random


