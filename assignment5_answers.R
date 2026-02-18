################################################
#
# Text Mining for Economics & Finance
#
#              Assignment 5 Answers
#

#
################################################


library(quanteda)
library(ggrepel)
library(textclean)
library(tidyverse)
library(glmnet)
library(spacyr) # a new one!
library(politeness) # another new one! And a package I wrote :)
library(semgram) # also new - for motifs

source("MLTASS_dfm.R")
source("kendall_acc.R")

###################################
set.seed(02138)

# load data
rmp<-read_csv("data/rmp_data.csv") %>%
  mutate(year=str_sub(post_date,-4,-1)) %>%
  mutate(wordcount=str_count(comments,"[[:alpha:]]+")) %>%
  filter(wordcount>10 & wordcount < 100)

# non-random training split
rmp_train<-rmp %>%
  filter(year<2012)
rmp_test<-rmp %>%
  filter(year>=2012)

# Politeness

rmp_train_polite<-politeness(rmp_train$comments,parser="spacy")
rmp_test_polite<-politeness(rmp_test$comments,parser="spacy")


rmp_train_dfm<-MLTASS_dfm(rmp_train$comments)
rmp_test_dfm<-MLTASS_dfm(rmp_test$comments,min.prop = 0) %>%
  dfm_match(colnames(rmp_train_dfm))


Cmodel_dfm<-cv.glmnet(x=rmp_train_dfm,
                      y=rmp_train$student_star)

Cmodel_polite<-cv.glmnet(x=as.matrix(rmp_train_polite),
                         y=rmp_train$student_star)

Cmodel_both<-cv.glmnet(x=cbind(as.matrix(rmp_train_polite),rmp_train_dfm),
                         y=rmp_train$student_star)


#### Evaluate Accuracy
dfm_predict<-predict(Cmodel_dfm,
                     newx = as.matrix(rmp_test_dfm))[,1]

polite_predict<-predict(Cmodel_polite,
                     newx = as.matrix(rmp_test_polite))[,1]

both_predict<-predict(Cmodel_both,
                        newx = cbind(as.matrix(rmp_test_polite),rmp_test_dfm))[,1]

kendall_acc(rmp_test$student_star,dfm_predict)

kendall_acc(rmp_test$student_star,polite_predict)

kendall_acc(rmp_test$student_star,both_predict)

featurePlot(rmp_test_polite,
            rmp_test$student_star,
            split_levels=c("Low","High"),
            split_name = "Star Rating",
            middle_out = .05
            )



# extract coefficients
plotCoefs<-Cmodel_both %>%
  coef() %>%
  drop() %>%
  as.data.frame() %>%
  rownames_to_column(var = "ngram") %>%
  rename(score=".") %>%
  filter(score!=0 & ngram!="(Intercept)" & !is.na(score))  

both_train<-cbind(as.matrix(rmp_train_polite),rmp_train_dfm)

# merge frequencies
plotDat<-plotCoefs %>%
  left_join(data.frame(ngram=colnames(both_train),
                       freq=colMeans(both_train))) %>%
  mutate_at(vars(score,freq),~round(.,3)) %>%
  # use case_when to 
  mutate(color=case_when(
    ngram%in%colnames(rmp_train_polite) ~ "polite",
    score<0 ~ "negative",
    score>0 ~ "positive")) %>%
  mutate(ngram=ifelse(ngram%in%colnames(rmp_train_polite),ngram,""))

# pipe into ggplot
plotDat %>%
  ggplot(aes(x=score,y=freq,label=ngram,color=color)) +
  geom_vline(xintercept=0)+
  geom_point() +
  geom_label_repel(max.overlaps = 15)+  
  scale_color_manual(values=c("red","black","blue")) +
  scale_y_continuous(trans="log2",
                     breaks=c(.01,.05,.1,.2,.5,1,2,5))+
  theme_bw() +
  labs(x="Coefficient in Cons Model",y="Uses per Review")+
  theme(legend.position = "none",
        axis.title=element_text(size=20),
        axis.text=element_text(size=16))

###################
# spacy parse
#############
rmp_sp<-spacy_parse(rmp_test$comments,
                      nounphrase = T,
                      lemma = T,
                      dependency = T,
                      pos = T,
                      tag=T)

# extract motifs

rmp_em<-extract_motifs(tokens = rmp_sp,
               parse_multi_token_entities = T,
               entities = c("he","she"),
               add_sentence = T,
               markup = T)

actions<-rmp_em$actions %>%
  group_by(action,Entity) %>%
  summarize(n=n()) %>%
  pivot_wider(names_from="Entity",values_from="n") %>%
  mutate_all(~replace_na(.,0)) %>%
  arrange(-he)

head(actions,10)

# calculate gender tilt

chars<-rmp_em$characterizations %>%
  group_by(characterization,Entity) %>%
  summarize(n=n()) %>%
  pivot_wider(names_from="Entity",values_from="n") %>%
  mutate_all(~replace_na(.,0)) %>%
  mutate(total=he+she,
    tilt=he/(he+she)) %>%
  filter(total>10) %>%
  arrange(-tilt)


chars
  

