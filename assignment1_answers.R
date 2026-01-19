################################################
#
# Text Mining for Economics & Finance
#
#              Assignment 1 Answers
#

#
################################################

library(quanteda)
library(ggrepel)
library(textclean)
library(tidyverse)
library(glmnet)

###############################################################
###############################################################

# Here  I am creating a function that saves all of our defaults in one place
TMEF_dfm<-function(text,
                   ngrams=1:2,
                   stop.words=TRUE,
                   min.prop=.01){
  if(!is.character(text)){                # First, we check our input is correct
    stop("Must input character vector")
  }
  drop_list=""
  if(stop.words) drop_list=stopwords("en") #uses stop.words argument to adjust what is dropped
  
  text_data<-text %>%
    replace_contraction() %>%
    tokens(remove_numbers=TRUE,
           remove_punct = TRUE) %>%
    tokens_wordstem() %>%
    tokens_select(pattern = drop_list, 
                  selection = "remove") %>%
    tokens_ngrams(ngrams) %>%
    dfm() %>%
    dfm_trim(min_docfreq = min.prop,docfreq_type="prop")
  return(text_data)
}

###############################################################
# Load data
###############################################################

# Review data
review_dat<-readRDS("review_dat.RDS")

# Business data
businesses<-readRDS("businessset.RDS")
# First thing - check variables

names(review_dat)

names(businesses)

businesses<-businesses %>%
  # remove the ones we don't need
  filter(business_id%in%review_dat$business_id) %>%
  # One variable name overlaps, so we rename one
  rename(average_stars="stars") %>%
  # convert to numeric 
  mutate(price=as.numeric(RestaurantsPriceRange2))


# We want to use reviews to predict price data, but price is in businesses, not review_dat

# To move the business data over to the review data, we use left_join

review_dat <- review_dat %>%
  left_join(businesses,
            by="business_id")

# First, we need to split the data into training and testing samples
train_split=sample(1:nrow(review_dat),round(nrow(review_dat)/2))

# create our prediction variables
dfm3<-TMEF_dfm(review_dat$text,ngrams=1) %>%
  convert(to="data.frame") %>%
  select(-doc_id)

###############################################################
# FIRST, THE GENDER MODEL
###############################################################

trainX<-dfm3 %>%
  slice(train_split) %>%
  as.matrix()

trainY<-review_dat %>%
  slice(train_split) %>%
  pull(male)

testX<-dfm3 %>% 
  slice(-train_split) %>%
  as.matrix()

testY<-review_dat %>%
  slice(-train_split) %>%
  pull(male)

# Put training data into LASSO model (note - glmnet requires a matrix)

lasso_model<-cv.glmnet(x=trainX,y=trainY)

# generate predictions for test data
test_predict<-predict(lasso_model,newx = testX)[,1]

# split the predictions in two, using the median

# note - the target Y variable is 1/0 so we have to convert to 1/0, not 2/1
test_predict_binary=ifelse(test_predict>median(test_predict),
                           1,
                           0) 

# calculate accuracy

round(100*mean(test_predict_binary==testY),3)

#####################
# Build a plot
#####################

# extract coefficients
plotCoefs<-lasso_model %>%
  coef() %>%
  drop() %>%
  as.data.frame() %>%
  rownames_to_column(var = "ngram") %>%
  rename(score=".") %>%
  filter(score!=0 & ngram!="(Intercept)" & !is.na(score))  

# merge frequencies
plotDat<-plotCoefs %>%
  left_join(data.frame(ngram=colnames(trainX),
                       freq=colMeans(trainX))) %>%
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
  scale_x_continuous(limits = c(-.2,.25),
                     breaks = seq(-.2,.2,.1)) +
  scale_y_continuous(trans="log2",
                     breaks=c(.01,.05,.1,.2,.5,1,2,5))+
  theme_bw() +
  labs(x="Coefficient in Gender Model",y="Uses per Review")+
  theme(legend.position = "none",
        axis.title=element_text(size=20),
        axis.text=element_text(size=16))

###############################################################
# NEXT, THE STAR RATING MODEL
###############################################################

# it's five levels, not two! So let's convert to a binary split 
hist(review_dat$stars)


trainX<-dfm3 %>%
  slice(train_split) %>%
  as.matrix()

trainY<-review_dat %>%
  slice(train_split) %>%
  pull(stars)

testX<-dfm3 %>% 
  slice(-train_split) %>%
  as.matrix()

testY<-review_dat %>%
  slice(-train_split) %>%
  pull(stars)

# Put training data into LASSO model (note - glmnet requires a matrix)

lasso_model<-cv.glmnet(x=trainX,y=trainY)

# generate predictions for test data
test_predict<-predict(lasso_model,newx = testX)[,1]

# split the predictions in two, using the median

# note - the target Y variable is 4/2 so we have to convert to 4/2, not 2/1
test_predict_binary=ifelse(test_predict>median(test_predict),
                           4,
                           2) 

# calculate accuracy

round(100*mean(test_predict_binary==testY),3)

#####################
# Build a plot
#####################

# extract coefficients
plotCoefs<-lasso_model %>%
  coef() %>%
  drop() %>%
  as.data.frame() %>%
  rownames_to_column(var = "ngram") %>%
  rename(score=".") %>%
  filter(score!=0 & ngram!="(Intercept)" & !is.na(score))  

# merge frequencies
plotDat<-plotCoefs %>%
  left_join(data.frame(ngram=colnames(trainX),
                       freq=colMeans(trainX))) %>%
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
  scale_x_continuous(#limits = c(-.2,.1),
                     breaks = seq(-.8,.2,.1)) +
  scale_y_continuous(trans="log2",
                     breaks=c(.01,.05,.1,.2,.5,1,2,5))+
  theme_bw() +
  labs(x="Coefficient in Star Rating Model",y="Uses per Review")+
  theme(legend.position = "none",
        axis.title=element_text(size=20),
        axis.text=element_text(size=16))
