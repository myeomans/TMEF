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
# Load descriptions data
######################################################################

jobdesc<-readRDS("jobdescriptions.RDS")

# Explore main meta-data
jobdesc %>%
  filter(ContractTime=="permanent" & ContractType=="full_time") %>%
  with(sort(table(Category)))


######################################################################
# Let's get sentiment working
######################################################################

text_sample=jobdesc %>%
  slice(1:1000) %>%
  pull(FullDescription)

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


######################################################################
# A multinomial classifier example
######################################################################

# five  common categories (ranked 5-9)
topcats<-names(rev(sort(table(jobdesc$Category)))[5:9])

# let's grab some descriptions from different categories
jd_cats<- jobdesc %>%
  filter(Category%in%topcats  & !is.na(Category) & ContractTime=="permanent" & ContractType=="full_time") %>%
  rename(salary="SalaryNormalized")


# note - this makes the "random" split identical for all of us, so we get the same results
set.seed(2022)

train_split=sample(1:nrow(jd_cats),6000)

jd_cats_train<-jd_cats[train_split,]
jd_cats_test<-jd_cats[-train_split,]


# Feature extraction is the same... n-grams

jd_cats_dfm_train<-TMEF_dfm(jd_cats_train$FullDescription,ngrams=1)

jd_cats_dfm_test<-TMEF_dfm(jd_cats_test$FullDescription,
                           ngrams=1,min.prop=0) %>%
  dfm_match(colnames(jd_cats_dfm_train))

# Multinomial tends to be a bit slower
jd_model_cats<-glmnet::cv.glmnet(x=jd_cats_dfm_train,
                                 y=jd_cats_train$Category,
                                 family="multinomial")

plot(jd_model_cats)

# With type="class", you can get a single predicted label for each document
cats_predict_label<-predict(jd_model_cats,
                            newx = jd_cats_dfm_test,
                            type="class")[,1]

# raw accuracy
mean(cats_predict_label==jd_cats_test$Category)

# Confusion matrix - great for multinomials!

table(cats_predict_label,jd_cats_test$Category)

# easier to read in R
table(cats_predict_label,substr(jd_cats_test$Category,0,10))

# but really you should export to a spreadsheet
table(cats_predict_label,jd_cats_test$Category) %>%
  write.csv("cats_table.csv")

# type="response" produces a probability that each document is in each class
cats_predict<-predict(jd_model_cats,
                      newx = jd_cats_dfm_test,
                      type="response")[,,1] %>%
  round(4)

# this way you can set different thresholds for each label
# use the probabilities in a regression instead of the absolute labels, etc.

# returns a matrix - one row per document, one column per class
head(cats_predict)
dim(cats_predict)


######################################################################
# A topic model example
######################################################################

# shrink the focus on a single category for topic modeling
jd_small<- jobdesc %>%
  filter(Category=="Engineering Jobs" & ContractTime=="permanent" & ContractType!="part_time") %>%
  mutate(desc_wdct=str_count(FullDescription,"[[:alpha:]]+"),
         sentiment=FullDescription %>%
           sentiment_by() %>%
           pull(ave_sentiment)
  ) %>%
  rename(salary="SalaryNormalized")

# note - this makes the "random" split identical for all of us, so we get the same results
set.seed(2022)

train_split=sample(1:nrow(jd_small),3500)

jd_small_train<-jd_small[train_split,]
jd_small_test<-jd_small[-train_split,]

# First we need a dfm object (ngram matrix in a quanteda file format)
# Topic models are usually estimated with only unigrams, and without stopwords
jd_small_dfm_train<-TMEF_dfm(jd_small_train$FullDescription,ngrams=1)

jd_small_dfm_test<-TMEF_dfm(jd_small_test$FullDescription,ngrams=1) %>%
  dfm_match(colnames(jd_small_dfm_train))


# Train a 20-topic model
jd_topicMod20<-stm(jd_small_dfm_train,K=20)

# There are metrics you can use to choose the topic number.
# These are controversial... you are better off adjusting to taste
# This is how you would run that, though....
# Fist convert to stm format, then put the documents and vocab into searchK()

# jd_stm_format<-jd_small_dfm_train %>%
#   convert(to="stm")
# sk<-searchK(jd_stm_format$documents,
#             jd_stm_format$vocab,
#             K=c(10,20,30,40))
# plot(sk)

# Note - you can save topic models as RDS files, too!

saveRDS(jd_topicMod20,file="jd_topicMod20.RDS")


jd_topicMod20<-readRDS("jd_topicMod20.RDS")

topicNum=jd_topicMod20$settings$dim$K

# LDA will not name your topics for you! It's good to come up with some names on your own

topicNames<-paste0("Topic",1:topicNum)

# Most common topics, and most common words from each topic
plot(jd_topicMod20,type="summary",n = 7,xlim=c(0,.3),labeltype = "frex",
     topic.names = topicNames) 

# You can add names to the vector one at a time
topicNames[1]="Infrastructure "
topicNames[2]="Design "
topicNames[5]="Software "
topicNames[8]="CNC "
topicNames[9]="Building Maintenance "
topicNames[14]="Sales "
topicNames[16]="Project Manager "
topicNames[20]="Systems"

# We can also grab more words per topic
labelTopics(jd_topicMod20)

findThoughts(model=jd_topicMod20,
             texts=jd_small_train$FullDescription,
             topics=2,n = 5)

# We can even put them in a word cloud! If you fancy it

cloud(jd_topicMod20,14)

cloud(jd_topicMod20,5)

# Which topics correlate with one another?
plot(topicCorr(jd_topicMod20),
     vlabels=topicNames,
     vertex.size=20)

stmEffects<-estimateEffect(1:topicNum~salary,
                           jd_topicMod20,
                           meta= jd_small_train %>%
                             select(salary))


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
  labs(y="Correlation with Salary",x="Topic") +
  theme(panel.grid=element_blank(),
        axis.text=element_text(size=20))




# This contains the topic proportions for each document..
topic_prop_train<-jd_topicMod20$theta
dim(topic_prop_train)
colnames(topic_prop_train)<-topicNames

# We can use these topic proportions just like any other feature
jd_model_stm<-glmnet::cv.glmnet(x=topic_prop_train,
                                y=jd_small_train$salary)

# Note that we didn't give enough features... there is no U shape
plot(jd_model_stm)

topic_prop_test<-fitNewDocuments(jd_topicMod20,
                                 jd_small_dfm_test %>%
                                   convert(to="stm") %>%
                                   `$`(documents))

test_stm_predict<-predict(jd_model_stm,
                          newx = topic_prop_test$theta)[,1]

# Note the drop in performance, compared to the ngrams
acc_stm<-kendall_acc(jd_small_test$salary,test_stm_predict)

acc_stm

jd_model_dfm<-glmnet::cv.glmnet(x=jd_small_dfm_train,
                                y=jd_small_train$salary)

plot(jd_model_dfm)

test_dfm_predict<-predict(jd_model_dfm,
                          newx = jd_small_dfm_test)[,1]

acc_dfm<-kendall_acc(jd_small_test$salary,test_dfm_predict)

acc_dfm

# How is our sentiment benchmark doing?
acc_sentiment<-kendall_acc(jd_small_test$salary,jd_small_test$sentiment)

acc_sentiment

# A wordcount benchmark for good measure
acc_wdct<-kendall_acc(jd_small_test$salary,jd_small_test$desc_wdct)

acc_wdct


