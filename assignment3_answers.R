########################################################
#
# Machine Learning & Text Analysis for Social Science
#
#               Assignment 3 Answers
#
#
########################################################

# Run these every time
library(quanteda)
library(ggrepel)
library(textclean)
library(tidyverse)
library(glmnet)
library(pROC)
library(sentimentr) # new one.. for sentiment
library(stm) # new one... for topic models


source("TMEF_dfm.R")
source("kendall_acc.R")

######################################################################
# Load descriptions data
######################################################################

jd_small<-readRDS("data/jd_small.RDS")


train_split=sample(1:nrow(jd_small),8000)

jd_small_train<-jd_small[train_split,]
jd_small_test<-jd_small[-train_split,]


jd_small_dfm_train<-TMEF_dfm(jd_small_train$FullDescription,ngrams=1)

jd_small_dfm_test<-TMEF_dfm(jd_small_test$FullDescription,
                            ngrams=1,min.prop = 0) %>%
  dfm_match(colnames(jd_small_dfm_train))


jd_model<-glmnet::cv.glmnet(x=jd_small_dfm_train %>%
                               as.matrix(),
                             y=jd_small_train$SalaryNormalized)

plot(jd_model)

#### Interpret with a coefficient plot
plotDat<-jd_model %>%
  coef() %>%
  drop() %>%
  as.data.frame() %>%
  rownames_to_column(var = "ngram") %>%
  rename(score=".") %>%
  filter(score!=0 & ngram!="(Intercept)" & !is.na(score))  %>%
  # add ngram frequencies for plotting
  left_join(data.frame(ngram=colnames(jd_small_dfm_train),
                       freq=colMeans(jd_small_dfm_train)))

plotDat %>%
  mutate_at(vars(score,freq),~round(.,3))

plotDat %>%
  ggplot(aes(x=score,y=freq,label=ngram,color=score)) +
  scale_color_gradient(low="green",high="purple")+
  geom_vline(xintercept=0)+
  geom_point() +
  geom_label_repel(max.overlaps = 15,force = 6)+  
  scale_y_continuous(trans="log2",
                     breaks=c(.01,.05,.1,.2,.5,1,2,5))+
  theme_bw() +
  labs(x="Coefficient in Model",y="Uses per Description")+
  theme(legend.position = "none",
        axis.title=element_text(size=20),
        axis.text=element_text(size=16))


######################################################################
# topic model
######################################################################

# First we need unigram dfm object (ngram matrix in a quanteda file format)
jd_small_dfm_train<-TMEF_dfm(jd_small_train$FullDescription,ngrams=1)

jd_small_dfm_test<-TMEF_dfm(jd_small_test$FullDescription,ngrams=1,min.prop=0) %>%
  dfm_match(colnames(jd_small_dfm_train))

# Train a 20-topic model
jd_topicMod20<-stm(jd_small_dfm_train,K=20)

saveRDS(jd_topicMod20,file="jd_topicMod20.RDS")


jd_topicMod20<-readRDS("jd_topicMod20.RDS")

topicNum=jd_topicMod20$settings$dim$K

# LDA will not name your topics for you! It's good to come up with some names on your own

topicNames<-paste0("Topic",1:topicNum)

# Most common topics, and most common words from each topic
plot(jd_topicMod20,type="summary",n = 7,xlim=c(0,.3),labeltype = "frex",
     topic.names = topicNames) 

# You can add names to the vector one at a time
topicNames[2]="Software"
topicNames[4]="Nursing"
topicNames[7]="Teacher"
topicNames[9]="Sales"
topicNames[11]="Law"
topicNames[14]="Non-Profit"
topicNames[16]="Engineering"
topicNames[20]="Digital Marketing"

# We can also grab more words per topic
labelTopics(jd_topicMod20)

findThoughts(model=jd_topicMod20,
             texts=jd_small_train$FullDescription,
             topics=2,n = 5)

# We can even put them in a word cloud! If you fancy it

cloud(jd_topicMod20,14)

cloud(jd_topicMod20,5)


stmEffects<-estimateEffect(1:topicNum~SalaryNormalized,
                           jd_topicMod20,
                           meta= jd_small_train %>%
                             select(SalaryNormalized))


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
                                y=jd_small_train$SalaryNormalized)

# Note that we didn't give enough features... there is no U shape
plot(jd_model_stm)

topic_prop_test<-fitNewDocuments(jd_topicMod20,
                                 jd_small_dfm_test %>%
                                   convert(to="stm") %>%
                                   `$`(documents))

test_stm_predict<-predict(jd_model_stm,
                          newx = topic_prop_test$theta)[,1]

# Note the drop in performance, compared to the ngrams
acc_stm<-kendall_acc(jd_small_test$SalaryNormalized,test_stm_predict)

acc_stm

jd_model_dfm<-glmnet::cv.glmnet(x=jd_small_dfm_train,
                                y=jd_small_train$SalaryNormalized)

plot(jd_model_dfm)

test_dfm_predict<-predict(jd_model_dfm,
                          newx = jd_small_dfm_test)[,1]

acc_dfm<-kendall_acc(jd_small_test$SalaryNormalized,test_dfm_predict)

acc_dfm

############### Benchmarks

# Create benchmarks

jd_small_test <- jd_small_test %>%
  mutate(text_wdct=str_count(FullDescription,"[[:alpha:]]+"),
         model_random=sample(test_dfm_predict),
         sentiment=sentiment_by(FullDescription) %>%
           pull(ave_sentiment))

acc_wdct<-kendall_acc(jd_small_test$SalaryNormalized,
                      jd_small_test$text_wdct)

acc_wdct

acc_random<-kendall_acc(jd_small_test$SalaryNormalized,
                        jd_small_test$model_random)

acc_random


acc_sentiment<-kendall_acc(jd_small_test$SalaryNormalized,
                        jd_small_test$sentiment)

acc_sentiment


######################################################################
# multinomial classifier
######################################################################

# five most common categories
topcats<-names(rev(sort(table(jd_small$Category)))[1:5])

# let's grab some descriptions from different categories
jd_cats<- jd_small %>%
  filter(Category%in%topcats  & !is.na(Category) & ContractTime=="permanent" & ContractType=="full_time")

# note - this makes the "random" split identical for all of us, so we get the same results
set.seed(02138)

train_split=sample(1:nrow(jd_cats),4000)

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



