################################################
#
#     Text Mining for Economics & Finance
#
#          In-Class Activity - Week 6
#               All the answers
#
################################################



# Run these every time
library(quanteda)
library(ggrepel)
library(textclean)
library(tidyverse)
library(glmnet)


source("vectorFunctions.R")
source("TMEF_dfm.R")

source("kendall_acc.R")

############### Word Vectors

# The real word vector files are ~ 6GB - too big! This is a smaller version,
# containing only the 50,000 most common words
vecSmall<-readRDS("vecSmall.RDS")

# Word frequency file - to reweight common words
load("wfFile.RData")

earningsCalls<-readRDS(file="earningsDat.RDS") %>%
  mutate(EPS_surprise=(EPS_actual-EPS_consens)/EPS_consens) %>%
  filter(FY%in%c("2010","2011"))


table(earningsCalls$FY)

train_split=which(earningsCalls$FY=="2010")

ec_train<-earningsCalls[train_split,]
ec_test<-earningsCalls[-train_split,]

ec_dfm_train<-TMEF_dfm(ec_train$opening_speech,ngrams=1)

ec_dfm_test<-TMEF_dfm(ec_test$opening_speech,ngrams=1) %>%
  dfm_match(colnames(ec_dfm_train))


ec_model<-glmnet::cv.glmnet(x=ec_dfm_train %>%
                              as.matrix(),
                            y=ec_train$EPS_actual)
plot(ec_model)

#### Interpret with a coefficient plot
ec_model %>%
  coef(s="lambda.min") %>%
  drop() %>%
  as.data.frame() %>%
  rownames_to_column(var = "ngram") %>%
  rename(score=".") %>%
  filter(score!=0 & ngram!="(Intercept)" & !is.na(score))  %>%
  # add ngram frequencies for plotting
  left_join(data.frame(ngram=colnames(ec_dfm_train),
                       freq=colMeans(ec_dfm_train))) %>%
  ggplot(aes(x=score,y=freq,label=ngram,color=score)) +
  scale_color_gradient(low="blue",high="red")+
  geom_vline(xintercept=0)+
  geom_point() +
  geom_label_repel(max.overlaps = 40,force = 6)+
  scale_y_continuous(trans="log2",
                     breaks=c(.01,.05,.1,.2,.5,1,2,5))+
  theme_bw() +
  labs(x="Coefficient in Model",y="Uses per Speech")+
  theme(legend.position = "none",
        axis.title=element_text(size=20),
        axis.text=element_text(size=16))


#### Evaluate Accuracy
test_ngram_predict<-predict(ec_model,
                            s="lambda.min",
                            newx = ec_dfm_test %>%
                              as.matrix())[,1]
acc_ngram<-kendall_acc(ec_test$EPS_actual,test_ngram_predict)

acc_ngram

###########################

# vectors

vdat_train<-vecCheck(ec_train$opening_speech,vecSmall,wfFile)

vdat_test<-vecCheck(ec_test$opening_speech,vecSmall,wfFile)

lasso_vec<-glmnet::cv.glmnet(x=vdat_train,
                             y=ec_train$EPS_actual)

test_vec_predict<-predict(lasso_vec,newx = vdat_test,
                          s="lambda.min")

kendall_acc(test_vec_predict,ec_test$EPS_actual)

# Other benchmarks

ec_test <- ec_test %>%
  mutate(speech_sent=syuzhet::get_sentiment(opening_speech),
         speech_wdct=str_count(opening_speech,"[[:alpha:]]+"))


acc_wdct<-kendall_acc(ec_test$EPS_actual,ec_test$speech_wdct)

acc_sent<-kendall_acc(ec_test$EPS_actual,ec_test$speech_sent)

acc_wdct
acc_sent


###################################################
# Q3 examples
###################################################

ec_test <- ec_test %>%
  mutate(vec_pred=test_vec_predict,
         ngram_pred=test_ngram_predict,
         vec_error=abs(vec_pred-EPS_actual),
         ngram_error=abs(ngram_pred-EPS_actual))
        
# find the range for the predictions 
ec_test %>%
  ggplot(aes(x=vec_error,y=ngram_error)) +
  geom_point() +
  scale_x_continuous(trans='sqrt') +
  scale_y_continuous(trans='sqrt') +
  theme_bw()

# 
ec_test %>%
  filter(vec_error<.25 & ngram_error > 1) %>%
  arrange(EPS_actual) %>%
  slice(1) %>%
  pull(opening_speech)

ec_test %>%
  filter(vec_error<.25 & ngram_error > 1) %>%
  arrange(-EPS_actual) %>%
  slice(1) %>%
  pull(opening_speech)

###################################################
# Q4 dictionaries
###################################################

positive_dict<-textdata::lexicon_loughran() %>%
  filter(sentiment=="uncertainty") %>%
  pull(word) %>%
  paste(collapse=" ")

pos_dict<-list(uc=textdata::lexicon_loughran() %>%
                 filter(sentiment=="positive") %>%
                 pull(word)) %>%
  dictionary()

pos_dict_bow<-ec_test$opening_speech %>%
  tokens() %>%
  dfm() %>%
  dfm_lookup(pos_dict) %>%
  convert(to="matrix") %>%
  apply(2, function(x) x/ec_test$speech_wdct)


pos_sims<-vecSimCalc(x=ec_test$opening_speech,
                  y=uncertain_dict,
                  vecfile=vecSmall,
                  wffile = wfFile,
                  PCAtrim = 1)

acc_possims<-kendall_acc(ec_test$EPS_actual,pos_sims)

acc_possims

acc_posbow<-kendall_acc(ec_test$EPS_actual,pos_dict_bow)

acc_posbow

####################################
# Q5 - Similarity
####################################

earningsCalls %>%
  filter(FY == c(2010,2011)) %>%
  with(table(FY,FQ))

table(earningsCalls$FY)

table(table(earningsCalls$IBES_ID))

# list of companies that have all 8 quarters in the data
allQs<-earningsCalls %>%
  group_by(IBES_ID) %>%
  summarize(n=n()) %>%
  filter(n==8) %>% 
  pull(IBES_ID)

repSet<-earningsCalls %>%
  filter(IBES_ID%in% allQs)

# matching opening speeches
oneSet<-repSet %>%
  filter(FY=="2010" & FQ==1) %>%
  select(IBES_ID,first_speech="opening_speech")

# merge back into data
repSet<-repSet %>%
  left_join(oneSet)


repSet$vecSim=NA
tpb<-txtProgressBar(0,nrow(repSet))
for (z in 1:nrow(repSet)){ 
  repSet[z,]$vecSim=vecSimCalc(x=repSet[z,]$opening_speech,
                               y=repSet[z,]$first_speech,
                               vecfile=vecSmall,
                               wffile=wfFile)
  setTxtProgressBar(tpb,z)
}

repSet %>%
  group_by(FY,FQ) %>%
  summarize(m=mean(vecSim),
            se=sd(vecSim)/sqrt(n())) %>%
  ggplot(aes(x=FQ,y=m,ymin=m-se,ymax=m+se)) +
  theme_bw() +
  geom_point() +
  geom_errorbar()+
  facet_wrap(~FY)


# An alternate approach - only if you *really* want to use PCAtrim

repSet$vecSim=NA
tpb<-txtProgressBar(0,length(unique(repSet$IBES_ID)))
z=0
for (ID in unique(repSet$IBES_ID)){ 
  ID_rows=which(repSet$IBES_ID==ID)
  
  repSet[ID_rows,]$vecSim=vecSimCalc(x=repSet[ID_rows,]$opening_speech,
                                     # we only need one because the first speech is the same for everyone
                                     y=repSet[ID_rows[1],]$first_speech,
                                     vecfile=vecSmall,
                                     wffile=wfFile,
                                     PCAtrim=1)
  setTxtProgressBar(tpb,z)
}

repSet %>%
  group_by(FY,FQ) %>%
  summarize(m=mean(vecSim),
            se=sd(vecSim)/sqrt(n())) %>%
  ggplot(aes(x=FQ,y=m,ymin=m-se,ymax=m+se)) +
  theme_bw() +
  geom_point() +
  geom_errorbar()+
  facet_wrap(~FY)

