################################################
#
# Text Mining for Economics & Finance
#
#              Assignment 6
#
#
################################################

library(quanteda)
library(ggrepel)
library(textclean)
library(tidyverse)
library(glmnet)
library(gemini.R)
library(tidyllm)

source("TMEF_dfm.R")
source("kendall_acc.R")


# filter to only look at calls from fiscal quarter 3, just to start
ecMain<-readRDS("data/earningsDat_3Y.RDS") %>%
  filter(FQ==3)

# filter the Q&A to match the filtering on the main dataset
ecQA<-readRDS("data/earningsQandA_3Y.RDS")%>%
  mutate(wordcount=str_count(text,"[[:alpha:]]+")) %>%
  filter(callID%in%ecMain$callID)

ecQA %>%
  filter(callID=="100521") %>%
  as_tibble() %>%
  head(30) 

# how many questions per call are there?
ecQA %>%
  filter(asker==1) %>%
  group_by(callID) %>%
  summarize(qcount=n()) %>%
  ggplot(aes(x=qcount)) +
  geom_histogram()

# how many questions per questioner are there?
ecQA %>%
  filter(asker==1) %>%
  group_by(callID,askerID) %>%
  summarize(qcount=n()) %>%
  ggplot(aes(x=qcount)) +
  geom_histogram()


# how many questioners per call are there?
ecQA %>%
  filter(asker==1) %>%
  group_by(callID) %>%
  summarize(qcount=n_distinct(askerID)) %>%
  ggplot(aes(x=qcount)) +
  geom_histogram()


# This variable indicates the order of askers in each call
ecQA %>%
  filter(asker==1 & followup == 0) %>%
  with(table(as.numeric(askerID)))

# question has the order of questions
ecQA %>%
  with(table(question,asker))

####################################################################################
# calculate a turn-level feature and add it to the conversation-level data
####################################################################################

# Average word count of first questions in the Q&A data and join to the main data
ecMain_merged <- ecMain %>%
  left_join(ecQA %>%
              filter(asker==1 & question==1) %>%
              group_by(callID) %>%
              summarize_at(vars(wordcount),sum) %>%
              rename(wordcount_q1="wordcount"))

# compare average answer word count to earnings per share
kendall_acc(ecMain_merged$wordcount_q1,
            ecMain_merged$EPS_actual)

# Average word count of first answers in the Q&A data and join to the main data
ecMain_merged <- ecMain %>%
  left_join(ecQA %>%
              filter(asker==0 & question==1) %>%
              group_by(callID) %>%
              summarize_at(vars(wordcount),sum) %>%
              rename(wordcount_a1="wordcount")
  ) %>%
  mutate(wordcount_a1=replace_na(wordcount_a1,0))

# compare average answer word count to earnings per share
kendall_acc(ecMain_merged$wordcount_a1,
            ecMain_merged$EPS_actual)


############################################################
# Create a question-answer pair dataset
############################################################

QApairs=ecQA %>%
  filter(callID=="101111") %>%
  mutate(qnum=question,
         asker=ifelse(asker==1,"question","answer")) %>%
  group_by(callID,qnum,asker) %>%
  summarize(text=paste(text,collapse=" ")) %>%
  pivot_wider(names_from="asker",values_from="text")

head(QApairs)

############################################################
# Use Gemini to annotate data
############################################################

revs<-read.csv("data/week1_answers.csv")

source("geminiAPI.R")
gemini.R::setAPI(gKey)
gKey="AIzaSyC9N1-dDm5NWZq68dukPAH5LySm9lFIZRw"
revs$stars_guess<-NA
revs$price_guess<-NA
revs$gender_guess<-NA


stars_task<-"I am giving you a restaurant review. 
  I want you to read it and guess if the writer liked or hated the restaurant.. 
If they liked it, respond with a '4'. If they hated it, respond with a '2'.
Only respond with the number 4 or 2, don't write anything else. 

Here is the review:  "


price_task<-"I am giving you a restaurant review. 
  I want you to read it and guess if the restaurant is cheap or expensive. 
If it is cheap, respond with a '1'. If it is expensive, respond with a '2'.
Only respond with the number 1 or 2, don't write anything else. 
Here is the review:  "

gender_task<-"I am giving you a restaurant review. 
  I want you to read it and guess if the writer is male or female. 
Answer only with the word 'male' or 'female' and nothing else. 
Here is the review:  "

for(x in 1:nrow(revs)){
  s_prompt=paste(stars_task,revs[x,]$text)
  xx<-gemini.R::gemini_chat(s_prompt,
                            model = "2.0-flash-lite")
  revs[x,]$stars_guess<-xx$outputs
  p_prompt=paste(price_task,revs[x,]$text)
  xx<-gemini.R::gemini_chat(p_prompt,
                            model = "2.0-flash-lite")
  revs[x,]$price_guess<-xx$outputs
  
  g_prompt=paste(gender_task,revs[x,]$text)
  xx<-gemini.R::gemini_chat(g_prompt,temperature=0,
                            
                            model = "2.0-flash-lite")
  revs[x,]$gender_guess<-xx$outputs
  print(x)
}

revs$stars_guess<-gsub("\n","",revs$stars_guess)
revs$price_guess<-gsub("\n","",revs$price_guess)
revs$gender_guess<-gsub("\n","",revs$gender_guess)

revs$gender_guess<-1*(revs$gender_guess=="male")

revs %>%
  with(table(gender,gender_guess))

revs %>%
  with(table(price,price_guess))

revs %>%
  with(table(stars,stars_guess))

############################################################

############################################################


library(tidyllm)
Sys.setenv(GOOGLE_API_KEY = gKey)

rev_dat<-readRDS("data/review_dat.RDS")

businesses<-readRDS("data/businessset.RDS")

rev_dat <- rev_dat %>%
  left_join(businesses %>%
              mutate(price=as.numeric(RestaurantsPriceRange2)) %>%
              select(price,business_id),
            by="business_id")


embed_corpus<-function(texts){
  batches<-sort(sample(1:(length(texts)/50),
                       length(texts),
                       replace=T))
  
  embed_data<-matrix(nrow=length(texts),
                     ncol=3072)
  tpb<-txtProgressBar(0,max(batches))
  for(x in 1:max(batches)){
    .rows<-which(batches==x)
    api_call<-gemini_embedding(
      .input=texts[.rows],
      .model = "gemini-embedding-001",
      .truncate = TRUE,
      .timeout = 120,
      .dry_run = FALSE,
      .max_tries = 3
    )
    for(y in 1:length(.rows)){
      embed_data[.rows[y],]<-unlist(api_call$embeddings[y])
    }
    setTxtProgressBar(tpb,x)
  }
  return(embed_data)
}

# Run on review text
rev_ec<-embed_corpus(rev_dat$text)


# ngrams as comparison
rev_dfm<-TMEF_dfm(rev_dat$text,ngrams=1) %>%
  convert(to="data.frame") %>%
  select(-doc_id)

# training samples
train_split=sample(1:nrow(rev_dat),
                   round(nrow(rev_dat)/2))

trainX_ec<-rev_ec[train_split,]

trainX_df<-rev_dfm %>%
  slice(train_split) %>%
  as.matrix()

trainY<-rev_dat %>%
  slice(train_split) %>%
  pull(price)

# testing samples
testX_df<-rev_dfm %>% 
  slice(-train_split) %>%
  as.matrix()

testX_ec<-rev_ec[-train_split,]

testY<-rev_dat %>%
  slice(-train_split) %>%
  pull(price)

# Build models
lasso_df<-cv.glmnet(x=trainX_df,y=trainY)
lasso_ec<-cv.glmnet(x=trainX_ec,y=trainY)

plot(lasso_df)
plot(lasso_ec)

# predict on new data
pred_df<-predict(lasso_df,testX_df)[,1]
pred_ec<-predict(lasso_ec,testX_ec)[,1]

kendall_acc(pred_df,testY)

kendall_acc(pred_ec,testY)

# use ngrams to explain bigger model
cor(pred_df,pred_ec)

full_output<-predict(lasso_ec,rev_ec)[,1]

interpreter<-cv.glmnet(x=as.matrix(rev_dfm),
                       y=full_output)

plot(interpreter)



# extract coefficients
plotCoefs<-interpreter %>%
  coef() %>%
  drop() %>%
  as.data.frame() %>%
  rownames_to_column(var = "ngram") %>%
  rename(score=".") %>%
  filter(score!=0 & ngram!="(Intercept)" & !is.na(score))  

# merge frequencies
plotDat<-plotCoefs %>%
  left_join(data.frame(ngram=colnames(rev_dfm),
                       freq=colMeans(rev_dfm))) %>%
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
  labs(x="Coefficient in Interpreter Model",y="Uses per Review")+
  theme(legend.position = "none",
        axis.title=element_text(size=20),
        axis.text=element_text(size=16))

