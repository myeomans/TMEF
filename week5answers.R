################################################
#
#     Text Mining for Economics & Finance
#
#          In-Class Activity - Week 5
#                Getting started
#
################################################

# Run these every time
library(quanteda)
library(ggrepel)
library(textclean)
library(tidyverse)
library(glmnet)
library(sentimentr) # new one.. for sentiment
library(stm) # new one... for topic models

source("TMEF_dfm.R")
source("kendall_acc.R")

######################################################################
# Load complaints data
######################################################################

cfpb<-readRDS("cfpb_small.RDS")

# note - this makes the "random" split identical for all of us, so we get the same results
set.seed(2022)

train_split=sample(1:nrow(cfpb),12000)

cfpb_train<-cfpb[train_split,]
cfpb_test<-cfpb[-train_split,]

######################################################################
# A topic model example
######################################################################


# First we need a dfm object (ngram matrix in a quanteda file format)
# Topic models are usually estimated with only unigrams, and without stopwords
cfpb_dfm_train<-TMEF_dfm(cfpb_train$narrative,ngrams=1)

cfpb_dfm_test<-TMEF_dfm(cfpb_test$narrative,ngrams=1) %>%
  dfm_match(colnames(cfpb_dfm_train))


# Train a 20-topic model
cfpb_topicMod20<-stm(cfpb_dfm_train,K=20)

# Note - you can save topic models as RDS files, too!

saveRDS(cfpb_topicMod20,file="cfpb_topicMod20.RDS")


cfpb_topicMod20<-readRDS("cfpb_topicMod20.RDS")

topicNum=cfpb_topicMod20$settings$dim$K

# LDA will not name your topics for you! It's good to come up with some names on your own

topicNames<-paste0("Topic",1:topicNum)

# Most common topics, and most common words from each topic
plot(cfpb_topicMod20,type="summary",n = 7,xlim=c(0,.3),labeltype = "frex",
     topic.names = topicNames) 

# You can add names to the vector one at a time
topicNames[4]="Legal rights"
topicNames[6]="Broken Website"
topicNames[11]="Loan Interest rates"
topicNames[13]="Medical Billing"
topicNames[17]="Equifax"
topicNames[18]="Experian"
topicNames[19]="Police reports"
topicNames[20]="Government Identification"

# We can also grab more words per topic
labelTopics(cfpb_topicMod20)

findThoughts(model=cfpb_topicMod20,
             texts=cfpb_train$narrative,
             topics=5,n = 1)

# We can even put them in a word cloud! If you fancy it

cloud(cfpb_topicMod20,11)

cloud(cfpb_topicMod20,9)

# Which topics correlate with one another?
plot(topicCorr(cfpb_topicMod20),
     vlabels=topicNames,
     vertex.size=20)

stmEffects<-estimateEffect(1:topicNum~disputed,
                           cfpb_topicMod20,
                           meta= cfpb_train %>%
                             select(disputed))


# The default plotting function is bad... Here's another version
bind_rows(lapply(summary(stmEffects)$tables,function(x) x[2,1:2])) %>%
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
  labs(y="Correlation with Disputed Status",x="Topic") +
  theme(panel.grid=element_blank(),
        axis.text=element_text(size=20))



# This contains the topic proportions for each document..
topic_prop_train<-cfpb_topicMod20$theta
dim(topic_prop_train)
colnames(topic_prop_train)<-topicNames

# We can use these topic proportions just like any other feature
cfpb_model_stm<-glmnet::cv.glmnet(x=topic_prop_train,
                                  y=cfpb_train$disputed)

# Note that we didn't give enough features... there is no U shape
plot(cfpb_model_stm)

topic_prop_test<-fitNewDocuments(cfpb_topicMod20,
                                 cfpb_dfm_test %>%
                                   convert(to="stm") %>%
                                   `$`(documents))

test_stm_predict<-predict(cfpb_model_stm,
                          newx = topic_prop_test$theta)[,1]

# Note the drop in performance, compared to the ngrams
acc_stm<-kendall_acc(cfpb_test$disputed,test_stm_predict)

acc_stm

cfpb_model_dfm<-glmnet::cv.glmnet(x=cfpb_dfm_train,
                                  y=cfpb_train$disputed)

plot(cfpb_model_dfm)

test_dfm_predict<-predict(cfpb_model_dfm,
                          newx = cfpb_dfm_test)[,1]

acc_dfm<-kendall_acc(cfpb_test$disputed,test_dfm_predict)

acc_dfm

cfpb_test<- cfpb_test %>%
  mutate(desc_wdct=str_count(narrative,"[[:alpha:]]+"),
         sentiment=narrative %>%
           sentiment_by() %>%
           pull(ave_sentiment)
  )


# How is our sentiment benchmark doing?
acc_sentiment<-kendall_acc(cfpb_test$disputed,cfpb_test$sentiment)

acc_sentiment

# A wordcount benchmark for good measure
acc_wdct<-kendall_acc(cfpb_test$disputed,cfpb_test$desc_wdct)

acc_wdct





######################################################################
# A multinomial classifier
######################################################################


# Feature extraction is the same... n-grams

cfpb_dfm_train<-TMEF_dfm(cfpb_train$narrative,ngrams=1)

cfpb_dfm_test<-TMEF_dfm(cfpb_test$narrative,
                           ngrams=1,min.prop=0) %>%
  dfm_match(colnames(cfpb_dfm_train))

# Multinomial tends to be a bit slower
cfpb_model<-glmnet::cv.glmnet(x=cfpb_dfm_train,
                                 y=cfpb_train$Issue,
                                 family="multinomial")

plot(cfpb_model)

# With type="class", you can get a single predicted label for each document
cats_predict_label<-predict(cfpb_model,
                            newx = cfpb_dfm_test,
                            type="class")[,1]

# raw accuracy
mean(cats_predict_label==cfpb_test$Issue)

# Confusion matrix - great for multinomials!

table(cats_predict_label,cfpb_test$Issue)

# easier to read in R
table(cats_predict_label,substr(cfpb_test$Issue,0,10))

# but really you should export to a spreadsheet
table(cats_predict_label,cfpb_test$Issue) %>%
  write.csv("cats_table.csv")

# type="response" produces a probability that each document is in each class
cats_predict<-predict(cfpb_model,
                      newx = cfpb_dfm_test,
                      type="response")[,,1] %>%
  round(4)

# this way you can set different thresholds for each label
# use the probabilities in a regression instead of the absolute labels, etc.

# returns a matrix - one row per document, one column per class
head(cats_predict)
dim(cats_predict)

