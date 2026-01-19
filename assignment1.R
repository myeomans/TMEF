################################################
#
# Text Mining for Economics & Finance
#
#                    Activity 1
#
#
################################################

# Run these once, if you haven't installed them before
#install.packages("tidyverse")
# install.packages("quanteda")
# install.packages("textclean")
# install.packages("ggrepel")
# install.packages("glmnet")

# Run these every time
library(tidyverse) # useful for almost everything
library(quanteda) # text analysis workhorse
library(textclean) # extra pre-processing
library(ggrepel) # for plots
library(glmnet) # Our estimation model



# read in data
week1_liz<-read.csv("week1_liz.csv")
week1_boris<-read.csv("week1_boris.csv")
week1_rishi<-read.csv("week1_rishi.csv")

# make sure the column names are correct
names(week1_liz)
names(week1_boris)
names(week1_rishi)

#make sure guesses are all filled in
table(week1_liz$stars_liz,useNA="ifany")
table(week1_liz$price_liz,useNA="ifany")
table(week1_liz$gender_liz,useNA="ifany")

table(week1_boris$stars_boris,useNA="ifany")
table(week1_boris$price_boris,useNA="ifany")
table(week1_boris$gender_boris,useNA="ifany")

table(week1_rishi$stars_rishi,useNA="ifany")
table(week1_rishi$price_rishi,useNA="ifany")
table(week1_rishi$gender_rishi,useNA="ifany")

# left join combines two datasets, using the ID columns in the "by" argument

week1<-left_join(week1_liz,
                 week1_boris,
                 by=c("review_id","text")) 

# note - both the ID and the text columns are identical in these two datasets. 
# you don't want to duplicate the text column when you join!


names(week1)

names(left_join(week1_liz,
                week1_boris,
                by=c("review_id")))


# You could also drop the text column before joining:

week1<-left_join(week1_liz,
                 week1_boris %>%
                   select(-text),
                 by="review_id") 

# a quick word on tidyverse - the %>% is called "pipe"
# it takes the finished object from the current line
# and inserts it as the first argument to the function on the next line

# so, these two commands are identical"
week1_boris %>%
  select(-text)

select(week1_boris, -text)

# as are these:

names(left_join(week1_liz,
                week1_boris,
                by=c("review_id","text")))

left_join(week1_liz,
          week1_boris,
          by=c("review_id","text")) %>%
  names()

# Okay, let's get back to work. We want to join all three datasets

week1_humans<-left_join(week1_liz,
                        week1_boris %>%
                          select(-text),
                        by=c("review_id")) %>%
  left_join(week1_rishi %>%
              select(-text),
            by=c("review_id"))

names(week1_humans)

# let's also look at the first few rows:

week1_humans %>%
  head()

# hard to read text columns... the "tibble" format from tidyverse has a cleaner print-out

week1_humans %>%
  as_tibble() %>%
  head()

# STOP - now create your own week1_humans dataset, using all of your groups' annotations

######################################################################

# First, we want to see how consistent the raters are

# we use starts_with(). a "tidy select" command, to grab all the stars ratings
week1_humans %>%
  select(starts_with("stars"))

# we will take the data.frame that contains the star ratings and compute a correlation table
# to make the display look good, we'll round the correlations to three decimal places
week1_humans %>%
  select(starts_with("stars")) %>%
  cor() %>%
  round(3)

# remember, this command is the same as (but is easier to read than):

round(cor(week1_humans[,c("stars_liz","stars_boris","stars_rishi")]),3)

# STOP - now try the same thing for price and for gender

######################################################################


week1_humans %>%
  select(starts_with("price")) %>%
  cor() %>%
  round(3)

week1_humans %>%
  select(starts_with("gender")) %>%
  cor() %>%
  round(3)

# gender is a character variable! We need to convert to numeric

# mutate is the function that creates new variables and changes old ones

week1_humans<- week1_humans %>%
  mutate(newcol=3)

week1_humans %>%
  head()

# if the variable exists already, you will overwrite it

week1_humans<- week1_humans %>%
  mutate(newcol=4)

week1_humans %>%
  head()

#Let's get rid of the new column

week1_humans<- week1_humans %>%
  select(-newcol)

week1_humans %>%
  names()

# If we want to apply an identical mutation to multiple columns, we can use "mutate_at"

week1_humans %>%
  select(starts_with("stars")) %>%
  head()

week1_humans %>%
  mutate_at(vars(starts_with("stars")), ~.+1) %>%
  head()

# Note: the ~ creates a "lambda function" - it is a quick way to create a short function
# In this case, the . represents the input to the function.
# It's shorthand for the following

addone<-function(x){
  x=x+1
  return(x)
}

week1_humans %>%
  mutate_at(vars(starts_with("stars")), addone) %>%
  head()

# We can also input the variable names into mutate_at directly, like this:
week1_humans %>%
  mutate_at(c("stars_liz","stars_rishi","stars_boris"),~.+1) %>%
  head()

# we could also select our columns first and then mutate_humans
week1_humans %>%
  select(starts_with("stars")) %>%
  mutate_all(~ .+1) %>%
  head()

# Getting back to our gender problem... we want to transform gender to numeric

# We can create a lambda function that returns TRUE/FALSE values
week1_humans %>%
  mutate_at(vars(starts_with("gender")),~(.=="male")) %>%
  head()

# You can convert TRUE/FALSE to binary by multiplying by 1
week1_humans %>%
  mutate_at(vars(starts_with("gender")),~1*(.=="male")) %>%
  head()

# For more complex transformations you can use ifelse

week1_humans %>%
  mutate_at(vars(starts_with("gender")),~ifelse(.=="male",1,0)) %>%
  head()

week1_humans %>%
  mutate_at(vars(starts_with("gender")),~ifelse(.=="male","one","zero")) %>%
  head()

# Let's convert to numeric binary and save it

week1_humans<-week1_humans %>%
  mutate_at(vars(starts_with("gender")),~ifelse(.=="male",1,0))

# Now we can compute our correlation

week1_humans %>%
  select(starts_with("gender")) %>%
  cor() %>%
  round(3)



# Correlations give us "pairwise" summary statistics. What if we want to 
# summarize more than two columns? We calculate "Cronbach's alpha"

# we need the psych package - run this installation once
# install.packages("psych")

# load the package every time you open R
library(psych)

week1_humans %>%
  select(starts_with("gender")) %>%
  alpha()

# The raw alpha is what we want... but this doesn't work

week1_humans %>%
  select(starts_with("gender")) %>%
  alpha()$total

# We could do this, and use the function version of $ which looks like `$`()

week1_humans %>%
  select(starts_with("gender")) %>%
  alpha() %>%
  `$`(total)

# More commonly, we use with()
# with() allows us to refer to the input inside the parens using a .
week1_humans %>%
  select(starts_with("gender")) %>%
  alpha() %>%
  with(.$total)

# This alpha is our "consistency" - how correlated are the annotators with each other?

# Note: this is not a measure of "validity" - how correlated are the annotators with the truth?

# What if all the annotators make the same mistake? high consistency, low validity

############################################################################

# Before we join this to the correct answers, we need one more new concept - pivoting to long format

week1_long <- week1_humans %>%
  pivot_longer(-c(text,review_id),names_to="question",values_to="guess")

dim(week1_long) # 288 = 9x32 = 9 guesses for each of 32 texts

# the "question" column contains two bits of info - the question and the annotator
head(week1_long)

# Lets split them into two separate columns
week1_long <- week1_long %>%
  separate(question,into=c("metric","annotator"),sep="_")

head(week1_long)


# checking that everything worked - we have 32 observations 
# for every combination of metric and annotator
week1_long %>%
  with(table(metric,annotator))

# Note we are using that with() function again! It's equivalent to:

table(week1_long$metric,week1_long$annotator)

############################################################################

# Let's bring in our correct answers

week1_answers=read.csv("week1_answers.csv")

head(week1_answers)

# Let's do the same pivot as before

week1_answers_long <- week1_answers %>%
  pivot_longer(-c(text,review_id),names_to="metric",values_to="answer")

# check to make sure the metric names match
table(week1_answers_long$metric)

# As before, let's left_join them
week1_all <- left_join(week1_long,
                       week1_answers_long %>%
                         select(-text),
                       by=c("review_id","metric"))

# Calculating accuracy here is easy: does the guess equal the answer?

week1_all <- week1_all %>%
  mutate(correct=1*(guess==answer))

# This tells us the average accuracy
mean(week1_all$correct)

# But we want to calculate accuracy separately for each metric/annotator

# We do this using group_by() and summarize()

week1_all %>%
  group_by(metric) %>%
  summarize(acc=mean(correct))

week1_all %>%
  group_by(annotator) %>%
  summarize(acc=mean(correct))

# Let save this set of results as an object
# Also, maybe we want standard errors? our formula for binary data is p*(1-p)/sqrt(n)
# In summarize() we can get the number of rows using n()

acc_report<-week1_all %>%
  group_by(annotator) %>%
  summarize(acc=mean(correct),
            se=sqrt(mean(correct)*(1-mean(correct))/n()))

print(acc_report)

# One last thing - we want percentages so let's multiple everything by 100

acc_report <- acc_report %>%
  mutate_at(c("acc","se"),~.*100)

print(acc_report)


# Tables are fun, but graphs are even more fun! Let's use ggplot
# We will work with ggplot a lot! 

# First we create a plot using ggplot() that contains aes() - short for "aesthetic"
# aes() will let us assign data columns to different aspects of the plot
# We then add layers to the plot with functions (note: we chain with +, not %>%)

acc_report %>% 
  ggplot(aes(x=annotator,color=annotator,
             y=acc,ymin=acc-se,ymax=acc+se)) +
  geom_hline(yintercept=50) +              # Adds baseline at 50%
  geom_point() +                           # adds points to the plot
  geom_errorbar(width=.4) +                # adds error bars
  labs(x="Annotator Name",                 # changes axis labels
       y="Accuracy") +                     
  theme_bw() +                             # changes the color scheme
  theme(axis.text=element_text(size=20),
        axis.title=element_text(size=24),
        panel.grid = element_blank(),
        legend.position = "none") # other design options


##### Let's do this all again, but separate the metrics as well

acc_report<-week1_all %>%
  group_by(annotator,metric) %>%
  summarize(acc=mean(correct),
            se=sqrt(mean(correct)*(1-mean(correct))/n())) %>%
  mutate_at(c("acc","se"),~.*100)

print(acc_report)

acc_report %>% 
  ggplot(aes(x=annotator,color=annotator,
             y=acc,ymin=acc-se,ymax=acc+se)) +
  geom_hline(yintercept=50) +              # Adds baseline at 50%
  facet_wrap(~metric) +                    # splits plot into separate "facets"
  geom_point() +                           # adds points to the plot
  geom_errorbar(width=.4) +                # adds error bars
  labs(x="Annotator Name",                 # changes axis labels
       y="Accuracy") +                     
  theme_bw() +                             # changes the color scheme
  theme(axis.text=element_text(size=20),
        axis.title=element_text(size=24),
        panel.grid = element_blank(),
        strip.text=element_text(size=24),
        strip.background = element_rect(fill="white"),
        legend.position = "none")          # other design options

# Finally, let's save this plot

ggsave("week1.png",dpi=200,width=20,height=10)

#####################################################################################
#####################################################################################

# Part 2 - let's build an NLP model

######### Simple bag of words

testDocs<-c("This is a test sentence.", 
            "I am providing another sentence to test this.",
            "This isn't a sentence",
            "This is a test document. It has 2 sentences")

# First we need to split up the sentences into "tokens" - (usually words)

testDocs %>%
  tokens()

# We then count how often each token occurs in each document 
# This produces a "document feature matrix" (or document term matrix)
# One row for each doc, one column for each feature
testDocs %>%
  tokens() %>%
  dfm()

# We can also combine adjoining words into "bigrams"

testDocs %>%
  tokens() %>%
  tokens_ngrams(2) %>%
  dfm()

# often people combine multiple token lengths together, as ngrams
testDocs %>%
  tokens() %>%
  tokens_ngrams(1:2) %>%
  dfm()

# Many different ways to tokenize - see the help file for options

?tokens

# We can stem words

testDocs %>%
  tokens(remove_punct=TRUE) %>%
  tokens_wordstem()

# we can remove punctuation
testDocs %>%
  tokens(remove_punct=TRUE) %>%
  tokens_ngrams(1:2)

# we can remove numbers
testDocs %>%
  tokens(remove_numbers=TRUE) %>%
  tokens_ngrams(1:2)

# contractions are done with a function from textclean
testDocs %>%
  replace_contraction() %>%
  tokens()


# dfm converts everything to lower case by default, but we can turn this off
testDocs %>%
  tokens() %>%
  dfm()

testDocs %>%
  tokens() %>%
  dfm(tolower=FALSE)

# we can also remove "stop words"
testDocs %>%
  tokens() %>%
  tokens_select(pattern = stopwords("en"), 
                selection = "remove") %>%
  tokens_ngrams(1:2)

# This is the built-in quanteda stopword list
stopwords("en")

# we can create our own custom list if we like
testDocs %>%
  tokens() %>%
  tokens_select(pattern = c("a","is","the"), 
                selection = "remove") %>%
  tokens_ngrams(1:2)


# Instead of removing common words, we can downweight them, using tfidf

dox<-c("This is a sentence.",
       "this is also a sentence.",
       "here is a rare word",
       "here is another word.",
       "and other sentences")

# Without tfidf, all words are given the same weight
dox %>%
  tokens(remove_punct= TRUE) %>%
  dfm() %>%
  convert(to="data.frame") %>%
  select(-doc_id) %>%
  round(2)

# Here, rare words are given more weight
dox %>%
  tokens(remove_punct= TRUE) %>%
  dfm() %>%
  dfm_tfidf() %>%
  convert(to="data.frame") %>%
  select(-doc_id) %>%
  round(2)

# We can also remove words that are too rare to learn anything about

dox %>%
  tokens(remove_punct= TRUE) %>%
  dfm() %>%
  dfm_trim(min_docfreq = 2) %>%
  convert(to="data.frame") %>%
  select(-doc_id) %>%
  round(2)

# Usually we do this by proportion of words

dox %>%
  tokens(remove_punct= TRUE) %>%
  dfm() %>%
  dfm_trim(min_docfreq = .25,docfreq_type="prop") %>%
  convert(to="data.frame") %>%
  select(-doc_id) %>%
  round(2)

# Typically the cut-off gets set around 1% of documents

# Here  I am creating a function that saves all of our defaults in one place
TMEF_dfm<-function(text,
                  ngrams=1:2,
                  stop.words=TRUE,
                  min.prop=.01){
  if(!is.character(text)){                # First, we check our input is correct
    stop("Must input character vector")
  }
  drop_list=""
  if(stop.words) drop_list=stopwords("en") #uses stop.words arugment to adjust what is dropped
  
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

TMEF_dfm(dox)

# we can easily modify the defaults of our custom arguments
TMEF_dfm(dox, ngrams=2)

TMEF_dfm(dox, stop.words = FALSE)

TMEF_dfm(dox, min.prop=.25)

# Note... this is a bit rudimentary
# If you prefer, you can use a more robust function I wrote for a different package
# install.packages("doc2concrete")
library(doc2concrete)

ngramTokens(dox)

######### New data - restaurant reviews

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

names(review_dat)

# Calculate a 1-gram feature count matrix for the review data, with no dropped words
dfm1<-TMEF_dfm(review_dat$text,
              ngrams=1,
              min.prop=0,
              stop.words = FALSE)

dim(dfm1) # >10k ngrams! Too many

# most common words - obvious
sort(colMeans(dfm1),decreasing=TRUE)[1:20]

# least common words
sort(colMeans(dfm1))[1:20]

######## Ok, let's build a model to predict price!

# First, let's look at our price data

table(review_dat$price)

# Let's only use 1-grams for now
dfm3<-TMEF_dfm(review_dat$text,ngrams=1) %>%
  convert(to="data.frame") %>%
  select(-doc_id)

# Lots of words
dim(dfm3)

#  Most common words in 1- and 2-price reviews... lots of the same words!
sort(colMeans(dfm3[review_dat$price==2,]),decreasing=T)[1:20]

sort(colMeans(dfm3[review_dat$price==1,]),decreasing=T)[1:20]

# What we really care about is - does the presence of a word predict price?

# A simple start - correlate each word with star rating

correlations<-dfm3 %>%
  summarise_all(~round(cor(.,review_dat$price),3)) %>%
  unlist()

# Ten lowest associations
sort(correlations)[1:10]

# Ten highest associations
rev(sort(correlations))[1:10]

# note - same as:
sort(correlations,decreasing=TRUE)[1:10]

# As we said in class we are not often interested in the effects of individual words
# Instead, we care more about how all the words perform as a class

# To do this, we will use the cv.glmnet() function to build a model

# First, we need to split the data into training and testing samples
train_split=sample(1:nrow(review_dat),round(nrow(review_dat)/2))

length(train_split)

# create our prediction variables
dfm3<-TMEF_dfm(review_dat$text,ngrams=1) %>%
  convert(to="data.frame") %>%
  select(-doc_id)


trainX<-dfm3 %>%
  slice(train_split) %>%
  as.matrix()

trainY<-review_dat %>%
  slice(train_split) %>%
  pull(price)

testX<-dfm3 %>% 
  slice(-train_split) %>%
  as.matrix()

testY<-review_dat %>%
  slice(-train_split) %>%
  pull(price)

# Put training data into LASSO model (note - glmnet requires a matrix)

lasso_model<-cv.glmnet(x=trainX,y=trainY)

# let's plot the cross-validation curve to see if it's finding any signal
plot(lasso_model)

# generate predictions for test data
test_predict<-predict(lasso_model,newx = testX)[,1]

# Note that while the true answers are binary, the predictions are continuous
# Always check these distributions!!
hist(testY)
hist(test_predict)

# For now, let's just split the predictions in two, using the median

test_predict_binary=ifelse(test_predict>median(test_predict),
                           2,
                           1)
hist(test_predict_binary)

# quick plot of the split to make sure it looks right
plot(x=test_predict,y=test_predict_binary)


# This should have the same values as testY
hist(test_predict_binary)

# and we can calculate accuracy from that

round(100*mean(test_predict_binary==testY),3)

#### What is in the model? We can extract the coefficients

# lots of zeros
lasso_model %>%
  coef() %>%
  drop()

# let's get this in a data frame
lasso_model %>%
  coef() %>%
  drop() %>%
  as.data.frame() %>%
  rownames_to_column(var = "ngram") %>%
  rename(score=".")

# just the top
lasso_model %>%
  coef() %>%
  drop() %>%
  as.data.frame() %>%
  rownames_to_column(var = "ngram") %>%
  rename(score=".") %>%
  head(20)

# drop zeros, and save
plotCoefs<-lasso_model %>%
  coef() %>%
  drop() %>%
  as.data.frame() %>%
  rownames_to_column(var = "ngram") %>%
  rename(score=".") %>%
  filter(score!=0 & ngram!="(Intercept)" & !is.na(score))  

plotCoefs

# create a similar data frame with ngram frequencies
plotFreqs<-data.frame(ngram=colnames(trainX),
                      freq=colMeans(trainX))


# combine data, round for easy reading
plotDat<-plotCoefs %>%
  left_join(plotFreqs) %>%
  mutate_at(vars(score,freq),~round(.,3))

head(plotDat)

# here's our first plot, with minimal customization
plotDat %>%
  ggplot(aes(x=score,y=freq,label=ngram,color=score)) +
  geom_point()

# Problems:
# Bad axis labels
# no point labels
# I don't like the default grey background
# legend is redundant

plotDat %>%
  ggplot(aes(x=score,y=freq,label=ngram,color=score)) +
  geom_point() +
  geom_label() +
  theme_bw() +
  labs(x="Coefficient in Model",y="Uses per Review")+
  theme(legend.position = "none")

# More problems:
# wasted space in Y axis
# lots of overlapping labels
# small axis labels
# i don't like the default colors

# colors we can set manually

plotDat %>%
  ggplot(aes(x=score,y=freq,label=ngram,color=score)) +
  scale_color_gradient2(low="blue",
                        mid = "grey",
                        high="green",
                        midpoint = 0)+
  geom_point() +
  geom_label_repel()+  
  scale_y_continuous(trans="log2",
                     breaks=c(.01,.05,.1,.2,.5,1,2,5))+
  theme_bw() +
  labs(x="Coefficient in Model",y="Uses per Review")+
  theme(legend.position = "none",
        axis.title=element_text(size=20),
        axis.text=element_text(size=16))

# let's get more words on the plot
# also make the X axis clearer
# use darker colors

plotDat %>%
  ggplot(aes(x=score,y=freq,label=ngram,color=score)) +
  scale_color_gradient2(low="navyblue",
                        mid = "grey",
                        high="forestgreen",
                        midpoint = 0)+
  geom_vline(xintercept=0)+
  geom_point() +
  geom_label_repel(max.overlaps = 15)+  
  scale_x_continuous(limits = c(-.2,.1),
                     breaks = seq(-.2,.2,.05)) +
  scale_y_continuous(trans="log2",
                     breaks=c(.01,.05,.1,.2,.5,1,2,5))+
  theme_bw() +
  labs(x="Coefficient in Model",y="Uses per Review")+
  theme(legend.position = "none",
        axis.title=element_text(size=20),
        axis.text=element_text(size=16))




