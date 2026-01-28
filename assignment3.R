########################################################
#
# Machine Learning & Text Analysis for Social Science
#
#               Assignment 3
#
#
########################################################

# new packages this week
#install.packages("sentimentr") # for sentiment
#install.packages("stm") # topic models

library(quanteda)
library(ggrepel)
library(textclean)
library(tidyverse)
library(glmnet)
library(pROC)
library(sentimentr)
library(stm)

source("TMEF_dfm.R")
source("kendall_acc.R")

# Review data
rev_med<-readRDS("data/rev_med.RDS")

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


######################################################################
# Let's get sentiment working
######################################################################

text_sample=rev_med_test %>%
  slice(1:100) %>%
  pull(text)

# the base function does each sentence separately
sentiment(text_sample)

#sentiment_by() calculates once per document
# but then you need the $ave_sentiment column
sentiment_by(text_sample)

#this is the sentiment vector you want
text_sample %>%
  sentiment_by() %>%
  pull(ave_sentiment)

# Note that when the vector gets large, it will throw an annoying warning
# technically, they want you to add an extra step

text_sample %>%
  get_sentences() %>%
  sentiment_by() %>%
  pull(ave_sentiment)

# BUT it will do that step automatically even if you don't do it yourself

#############################################
# extract dictionary the normal way
#############################################

loughran_words<-textdata::lexicon_loughran()

uncertain_dict<-dictionary(list(
  loughran_uncertainty=loughran_words %>%
    filter(sentiment=="uncertainty") %>%
    pull(word)))

# Traditional dictionary approach using dfm_lookup()
rev_med_train_dicts<-rev_med_train %>%
  pull(text) %>%
  tokens() %>%
  dfm() %>%
  dfm_lookup(uncertain_dict) %>%
  convert(to="data.frame")

# Accuracy score using traditional dictionary
kendall_acc(rev_med_train_dicts$loughran_uncertainty,
            rev_med_train$stars)

########################################################
# using L&M dictionary - with/without grammar awareness
########################################################

loughran_words<-textdata::lexicon_loughran()

rev_med_train_dicts<-rev_med_train %>%
  pull(text) %>%
  tokens() %>%
  dfm() %>%
  dfm_lookup(as.dictionary(loughran_words)) %>%
  convert(to="data.frame") %>%
  select(-doc_id)

# all the dictionaries are in there!
head(rev_med_train_dicts)

# usually you want to divide by the word count
rev_med_train_dicts<-rev_med_train_dicts %>%
  mutate_all(~./rev_med_train$word_count)

rev_med_train_dicts<-rev_med_train_dicts %>%
  mutate(sentiment=positive-negative)

kendall_acc(rev_med_train_dicts$sentiment,
            rev_med_train$stars)

rev_med_train<-rev_med_train %>%
  mutate(LMsentiment=sentiment_by(text,
                                  polarity_dt=lexicon::hash_sentiment_loughran_mcdonald) %>%
           pull(ave_sentiment))

kendall_acc(rev_med_train$LMsentiment,
            rev_med_train$stars)

# domain-general dictionary works better than finance
rev_med_train<-rev_med_train %>%
  mutate(BLsentiment=sentiment_by(text) %>%
           pull(ave_sentiment))

kendall_acc(rev_med_train$BLsentiment,
            rev_med_train$stars)


# examples - 
c("this is a bad product","this is not a bad product") %>%
  sentiment_by(polarity_dt=lexicon::hash_sentiment_loughran_mcdonald) 

c("this is a bad product","this is not a bad product") %>%
  tokens() %>%
  dfm() %>%
  dfm_lookup(as.dictionary(loughran_words))

# vader also works this way
library(vader)

vader::vader_df(text = c("this is a bad product","this is not a bad product"))

rev_med_train<-rev_med_train %>%
  mutate(VDsentiment=vader_df(text) %>%
           pull(compound))

kendall_acc(rev_med_train$VDsentiment,
            rev_med_train$stars)

# vader is good for quirks of internet speech
vader::vader_df(text = c("lol this is not bad",
                         ":) this is not bad",
                         "this is not bad"))



######################################################################
# A multinomial classifier example
######################################################################

# the categories are in a text field, so we need to extract them - with dfm!

train_cats<-TMEF_dfm(rev_med_train$categories)%>%
  convert(to="data.frame") %>%
  as_tibble() %>%
  select(chines,sandwich,nightlif,mexican) 

# 3437 that are in only one category ... let's dump the rest
table(rowSums(train_cats))

one_cat_train=rev_med_train %>%
  filter(rowSums(train_cats)==1) %>%
  mutate(category=case_when(
    str_detect(categories,"Chinese") ~ "chinese",
    str_detect(categories,"Sandwich") ~ "sandwich",
    str_detect(categories,"Nightlife") ~ "nightlife",
    str_detect(categories,"Mexican") ~ "mexican"
  ))

table(one_cat_train$category)

# do the same in the test set
test_cats<-TMEF_dfm(rev_med_test$categories)%>%
  convert(to="data.frame") %>%
  as_tibble() %>%
  select(chines,sandwich,nightlif,mexican) 


one_cat_test=rev_med_test %>%
  filter(rowSums(test_cats)==1)%>%
  mutate(category=case_when(
    str_detect(categories,"Chinese") ~ "chinese",
    str_detect(categories,"Sandwich") ~ "sandwich",
    str_detect(categories,"Nightlife") ~ "nightlife",
    str_detect(categories,"Mexican") ~ "mexican"
  ))

table(one_cat_test$category)


# Feature extraction is the same... n-grams

one_cat_dfm_train<-TMEF_dfm(one_cat_train$text,ngrams=1)

one_cat_dfm_test<-TMEF_dfm(one_cat_test$text,
                          ngrams=1,
                          min.prop=0) %>%
  dfm_match(colnames(one_cat_dfm_train))



# Multinomial tends to be a bit slower
one_cat_model<-glmnet::cv.glmnet(x=one_cat_dfm_train,
                                 y=one_cat_train$category,
                                 family="multinomial",
                                 maxit=5000)

plot(one_cat_model)

# With type="class", you can get a single predicted label for each document
cats_predict_label<-predict(one_cat_model,
                            newx = one_cat_dfm_test,
                            type="class")[,1]

# raw accuracy
mean(cats_predict_label==one_cat_test$category)

# Confusion matrix - great for multinomials!

table(cats_predict_label,one_cat_test$category)

# to export the table more easily
table(cats_predict_label,one_cat_test$category) %>%
  write.csv("cats_table.csv")

# type="response" produces a probability that each document is in each class
cats_predict<-predict(one_cat_model,
                      newx = one_cat_dfm_test,
                      type="response")[,,1] %>%
  round(4)

# this way you can set different thresholds for each label
# use the probabilities in a regression instead of the absolute labels, etc.

# returns a matrix - one row per document, one column per class
head(cats_predict,20)
dim(cats_predict)


######################################################################
# A topic model example
######################################################################

# First we need a dfm object (ngram matrix in a quanteda file format)
# Topic models are usually estimated with only unigrams, and without stopwords

# There is one row that is empty! The topic model with break with this
table(rowSums(rev_med_dfm_train)==0)

# You should remove it first, before estimating 
rev_med_train<-rev_med_train[rowSums(rev_med_dfm_train)!=0,]
rev_med_dfm_train<-rev_med_dfm_train[rowSums(rev_med_dfm_train)!=0,]

# Same in the test data!
rev_med_test<-rev_med_test[rowSums(rev_med_dfm_test)!=0,]
rev_med_dfm_test<-rev_med_dfm_test[rowSums(rev_med_dfm_test)!=0,]

# Train a 20-topic model
rev_topicMod20<-stm(rev_med_dfm_train,K=20)

# There are metrics you can use to choose the topic number.
# These are controversial... you are better off adjusting to taste
# This is how you would run that, though....
# Fist convert to stm format, then put the documents and vocab into searchK()

# rev_stm_format<-rev_med_dfm_train %>%
#   convert(to="stm")
# sk<-searchK(rev_stm_format$documents,
#             rev_stm_format$vocab,
#             K=c(10,20,30,40))
# plot(sk)

# Note - you can save topic models as RDS files, too!

saveRDS(rev_topicMod20,file="rev_topicMod20.RDS")


rev_topicMod20<-readRDS("rev_topicMod20.RDS")

topicNum=rev_topicMod20$settings$dim$K

# LDA will not name your topics for you! It's good to come up with some names on your own

topicNames<-paste0("Topic",1:topicNum)

# Most common topics, and most common words from each topic
plot(rev_topicMod20,type="summary",n = 7,xlim=c(0,.3),labeltype = "frex",
     topic.names = topicNames) 

# You can add names to the vector one at a time
topicNames[1]="Payment"
topicNames[9]="Seafood"
topicNames[13]="Food Quality"
topicNames[14]="Ramen"
topicNames[11]="Breakfast"
# We can also grab more words per topic
labelTopics(rev_topicMod20)

findThoughts(model=rev_topicMod20,
             texts=rev_med_train$text,
             topics=11,n = 5)

# We can even put them in a word cloud! If you fancy it

cloud(rev_topicMod20,11)

cloud(rev_topicMod20,9)


stmeffects<-estimateEffect(1:topicNum~stars,
                           rev_topicMod20,
                           meta= rev_med_train %>%
                             select(stars))


# The default plotting function is bad... Here's another version
bind_rows(lapply(summary(stmeffects)$tables,function(x) x[2,1:2])) %>%
  mutate(topic=factor(topicNames,ordered=T,
                      levels=topicNames),
         se_u=Estimate+`Std. Error`,
         se_l=Estimate-`Std. Error`) %>%
  ggplot(aes(x=topic,y=Estimate,ymin=se_l,ymax=se_u)) +
  geom_point() +
  geom_errorbar() +
  coord_flip() +
  geom_hline(yintercept = 0)+
  theme_bw() +
  labs(y="Correlation with Star Rating",x="Topic") +
  theme(panel.grid=element_blank(),
        axis.text=element_text(size=20))


# This contains the topic proportions for each document..
topic_prop_train<-rev_topicMod20$theta
dim(topic_prop_train)
colnames(topic_prop_train)<-topicNames

# We can use these topic proportions just like any other feature
rev_model_stm<-glmnet::cv.glmnet(x=topic_prop_train,
                                 y=rev_med_train$stars)

# Note that we didn't give enough features... there is no U shape
plot(rev_model_stm)

topic_prop_test<-fitNewDocuments(rev_topicMod20,
                                 rev_med_dfm_test %>%
                                   convert(to="stm") %>%
                                   `$`(documents))


test_stm_predict<-predict(rev_model_stm,
                          newx = topic_prop_test$theta)[,1]

# Note the drop in performance, compared to the ngrams
acc_stm<-kendall_acc(rev_med_test$stars,test_stm_predict)

acc_stm


