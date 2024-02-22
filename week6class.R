################################################
#
#     Text Mining for Economics & Finance
#
#          In-Class Activity - Week 6
#               
#
################################################


# Run these every time
library(quanteda)
library(ggrepel)
library(textclean)
library(tidyverse)
library(glmnet)
library(sentimentr)

source("vectorFunctions.R") # a new one!
source("TMEF_dfm.R")
source("kendall_acc.R")

############### Word Vectors

# The real word vector files are ~ 6GB - too big for dropbox! 
# This is a smaller version,
# containing only the 50,000 most common words
vecSmall<-readRDS("vecSmall.RDS")

# You can download the full version from here if you like
# https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip
#
# Big files need to be loaded using data.table - much faster
# library(data.table)
# vecFile<-data.table::fread("crawl-300d-2M.vec",
#                            quote="",header=F,col.names = c("word",paste0("vec",1:300)))

# remember: ALWAYS clear big files out of the workspace to reduce memory load before closing RStudio
#rm(vecSmall)
head(vecSmall)

# Word frequency file - to reweight common words
load("wfFile.RData")

# one column with words, and 300 with vector projections (uninterpretable!)
head(vecSmall[,1:20])

head(wfFile)

# Calculating similarity using bag of words doesn't know the difference between sad and happy!
bowSimCalc(x=c("I am very sad","I am very happy"),
           y="I am thrilled")

# However, processing the text as dense vectors allows the meaning to emerge. 
vecSimCalc(x=c("I am very sad","I am very happy"),
           y="I am thrilled",
           vecfile=vecSmall)

# word frequency weighting removes influence of most (globally) common words
vecSimCalc(x=c("I am very sad","I am very happy"),
           y="I am thrilled",
           vecfile=vecSmall,
           wffile=wfFile)

# PCAtrim removes influence of (locally) overlapping words
vecSimCalc(x=c("I am very sad","I am very happy"),
           y="I am thrilled",
           vecfile=vecSmall,
           wffile=wfFile,
           PCAtrim = 1)




set.seed(2022)

cfpb_small<-readRDS(file="cfpb_small.RDS") %>%
  slice(1:5000)

train_split=sample(1:nrow(cfpb_small),4000)

cfpb_small_train<-cfpb_small[train_split,]
cfpb_small_test<-cfpb_small[-train_split,]


cfpb_small_dfm_train<-TMEF_dfm(cfpb_small_train$narrative)
cfpb_small_dfm_test<-TMEF_dfm(cfpb_small_test$narrative,min.prop = 0) %>%
  dfm_match(colnames(cfpb_small_dfm_train))

#############################################
# project data to embedding space
vdat<-vecCheck(cfpb_small$narrative,
               vecSmall,
               wfFile,
               PCAtrim=1)


vdat_train<-vdat[train_split,]
vdat_test<-vdat[-train_split,]

#############################################
# Train a vector classifier

lasso_vec<-glmnet::cv.glmnet(x=vdat_train,
                             y=cfpb_small_train$disputed)

# notice two lines - one is at the minimum, the other is more conservative 
plot(lasso_vec)

# the default chooses the more conservative one, with fewer features
test_all_predict<-predict(lasso_vec,
                          newx = vdat_test)

kendall_acc(test_all_predict,cfpb_small_test$disputed)

# this is how you use the minimum one - usually it produces better accuracy
test_vec_predict<-predict(lasso_vec,newx = vdat_test,
                          s="lambda.min")

kendall_acc(test_vec_predict,cfpb_small_test$disputed)

#############################################
# vector embeddings + ngrams
combined_x_train=cbind(vdat_train,cfpb_small_dfm_train)
combined_x_test=cbind(vdat_test,cfpb_small_dfm_test)

lasso_all<-glmnet::cv.glmnet(x=combined_x_train,
                             y=cfpb_small_train$disputed)

plot(lasso_all)

test_all_predict<-predict(lasso_all,
                          newx = combined_x_test,
                          s="lambda.min")

kendall_acc(test_all_predict,cfpb_small_test$disputed)


#############################################
# ngrams alone
lasso_dfm<-glmnet::cv.glmnet(x=cfpb_small_dfm_train,
                             y=cfpb_small_train$disputed)

plot(lasso_dfm)

test_dfm_predict<-predict(lasso_dfm,newx = cfpb_small_dfm_test,
                          s="lambda.min")

kendall_acc(test_dfm_predict,cfpb_small_test$disputed)


########################################
# similarity calculation
########################################

table(cfpb_small_train$Company)

cfpb_small_train %>%
  filter(Company=="Sherloq Group, Inc") %>%
  pull(narrative)

which(cfpb_small_train$Company=="Sherloq Group, Inc")

target<-cfpb_small_train %>%
  filter(Company=="Sherloq Group, Inc") %>%
  pull(narrative)

sims<-vecSimCalc(x=cfpb_small_train$narrative,
                 y=target,
                 vecfile=vecSmall,
                 wffile = wfFile,
                 PCAtrim=1)

hist(sims)
max(sims)

 
cfpb_small_train %>%
  arrange(-sims) %>%
  slice(1:2) %>%
  pull(narrative)

cfpb_small_train$sims<-sims

######################################################################
# Distributed Dictionary
######################################################################

# extract dictionary as document
uncertainty_dict<-textdata::lexicon_loughran() %>%
  filter(sentiment=="uncertainty") %>%
  pull(word) %>%
  paste(collapse=" ")

# calculate similarities to dictionary "document"
lsims<-vecSimCalc(x=cfpb_small_train$narrative,
                  y=uncertainty_dict,
                  vecfile=vecSmall,
                  wffile = wfFile,
                  PCAtrim=1)

# estimate accuracy
kendall_acc(lsims,cfpb_small_train$disputed)

# add the similarity scores to the data.frame
cfpb_small_train$uncertain_sim<-lsims

cfpb_small_train %>%
  group_by(`Sub-issue`) %>%
  summarize(m=mean(uncertain_sim),
            se=sd(uncertain_sim)/sqrt(n())) %>%
  # reorder() re-orders the group names according to the mean values
  mutate(`Sub-issue`=reorder(`Sub-issue`,-m)) %>%
  ggplot(aes(x=`Sub-issue`,y=m,
             ymin=m-se,ymax=m+se)) +
  geom_point() +
  geom_errorbar() +
  theme_bw() +
  labs(y="Normalised Similarity with Uncertainty Dictionary") +
  coord_flip() # This line puts the long names on the left axis!


#############################################
# extract dictionary the normal way
#############################################

loughran_words<-textdata::lexicon_loughran()

uncertain_dict<-dictionary(list(
  loughran_uncertainty=loughran_words %>%
    filter(sentiment=="uncertainty") %>%
    pull(word)))

# Traditional dictionary approach using dfm_lookup()
cfpb_small_train_dicts<-cfpb_small_train %>%
  pull(narrative) %>%
  tokens() %>%
  dfm() %>%
  dfm_lookup(uncertain_dict) %>%
  convert(to="data.frame")

# Accuracy score using traditional dictionary
kendall_acc(cfpb_small_train_dicts$loughran_uncertainty,
            cfpb_small_train$disputed)

########################################################
# using L&M dictionary - with/without grammar awareness
########################################################

loughran_words<-textdata::lexicon_loughran()


cfpb_small_train_dicts<-cfpb_small_train %>%
pull(narrative) %>%
tokens() %>%
dfm() %>%
dfm_lookup(as.dictionary(loughran_words)) %>%
convert(to="data.frame") %>%
select(-doc_id)

# all the dictionaries are in there!
head(cfpb_small_train_dicts)

# usually you want to divide by the word count
cfpb_small_train_dicts<-cfpb_small_train_dicts %>%
  mutate_all(~./cfpb_small_train$narr_wdct)

cfpb_small_train_dicts<-cfpb_small_train_dicts %>%
  mutate(sentiment=positive-negative)

kendall_acc(-cfpb_small_train_dicts$sentiment,
            cfpb_small_train$disputed)

cfpb_small_train<-cfpb_small_train %>%
  mutate(LMsentiment=sentiment_by(narrative,
                                  polarity_dt=lexicon::hash_sentiment_loughran_mcdonald) %>%
pull(ave_sentiment))

kendall_acc(-cfpb_small_train$LMsentiment,
            cfpb_small_train$disputed)

# examples - 
c("this is a bad product","this is not a bad product") %>%
  sentiment_by(polarity_dt=lexicon::hash_sentiment_loughran_mcdonald) 

c("this is a bad product","this is not a bad product") %>%
  tokens() %>%
  dfm() %>%
  dfm_lookup(as.dictionary(loughran_words))

# ALWAYS clear big files out of the workspace to reduce memory load before closing RStudio
rm(vecSmall,wfFile)

