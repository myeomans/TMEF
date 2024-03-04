################################################
#
#     Text Mining for Economics & Finance
#
#          In-Class Activity - Week 7
#               All the answers
#
################################################



# Run these every time
library(quanteda)
library(ggrepel)
library(textclean)
library(tidyverse)
library(glmnet)
library(spacyr) 
library(politeness)


source("TMEF_dfm.R")
source("kendall_acc.R")

ecMain<-readRDS("earningsDat_3Y.RDS") %>%
  filter(FY==2011)

ecQA<-readRDS("earningsQandA_3Y.RDS")%>%
  mutate(wordcount=str_count(text,"[[:alpha:]]+")) %>%
  filter(callID%in%ecMain$callID)


################################################
# Question 1
################################################

ecQA %>%
  filter(asker==1) %>%
  group_by(callID,askerID) %>%
  summarize(qcount=n()) %>%
  group_by(askerID) %>%
  summarize(m=mean(qcount),
            se=sd(qcount)/sqrt(n())) %>%
  mutate(askerID=as.numeric(askerID)) %>%
  filter(askerID<20) %>%
  ggplot(aes(x=askerID,y=m,
             ymin=m-se,ymax=m+se)) +
  geom_point() +
  geom_errorbar() +
  theme_bw() +
  labs(x="Asker Order",y="Number of Questions Asked")


################################################
# Question 2
################################################

# Join turn-level data (from the first ten questions) to the call-level data
ecMain_merged <- ecMain %>%
  # combine answer text as a single document and merge
  left_join(ecQA %>%
              filter(asker==0 & question <=10) %>%
              group_by(callID) %>%
              summarize(answertext=paste(text,collapse=" "),
                        answer_wdct=str_count(answertext,"[[:alpha:]]+"))) %>%
  # combine question text as a single document and merge
  left_join(ecQA %>%
              filter(asker==1 & question <=10) %>%
              group_by(callID) %>%
              summarize(questiontext=paste(text,collapse=" "),
                        question_wdct=str_count(questiontext,"[[:alpha:]]+")))

# Questions First

ecMain_dfm_q<-TMEF_dfm(ecMain_merged$questiontext)

EPSmodel_q<-cv.glmnet(x=as.matrix(ecMain_dfm_q),
                    y=(ecMain$EPS_actual))

plot(EPSmodel_q)

#### Interpret with a coefficient plot
plotDat<-EPSmodel_q %>%
  coef(s="lambda.min") %>%
  drop() %>%
  as.data.frame() %>%
  rownames_to_column(var = "ngram") %>%
  rename(score=".") %>%
  filter(score!=0 & ngram!="(Intercept)" & !is.na(score))  %>%
  # add ngram frequencies for plotting
  left_join(data.frame(ngram=colnames(ecMain_dfm_q),
                       freq=colMeans(ecMain_dfm_q)))

plotDat %>%
  ggplot(aes(x=score,y=freq,label=ngram,color=score)) +
  scale_color_gradient(low="blue",high="red")+
  geom_vline(xintercept=0)+
  geom_point() +
  geom_label_repel(max.overlaps = 20,force = 6)+  
  scale_y_continuous(trans="log2",
                     breaks=c(.01,.05,.1,.2,.5,1,2,5))+
  theme_bw() +
  labs(x="Coefficient in Model",y="Uses per Question")+
  theme(legend.position = "none",
        axis.title=element_text(size=20),
        axis.text=element_text(size=16))

########## Answers next

ecMain_dfm_a<-TMEF_dfm(ecMain_merged$answertext)


EPSmodel_a<-cv.glmnet(x=as.matrix(ecMain_dfm_a),
                      y=(ecMain$EPS_actual))

plot(EPSmodel_a)

#### Interpret with a coefficient plot
plotDat<-EPSmodel_a %>%
  coef(s="lambda.min") %>%
  drop() %>%
  as.data.frame() %>%
  rownames_to_column(var = "ngram") %>%
  rename(score=".") %>%
  filter(score!=0 & ngram!="(Intercept)" & !is.na(score))  %>%
  # add ngram frequencies for plotting
  left_join(data.frame(ngram=colnames(ecMain_dfm_q),
                       freq=colMeans(ecMain_dfm_q)))

plotDat %>%
  ggplot(aes(x=score,y=freq,label=ngram,color=score)) +
  scale_color_gradient(low="blue",high="red")+
  geom_vline(xintercept=0)+
  geom_point() +
  geom_label_repel(max.overlaps = 20,force = 6)+  
  scale_y_continuous(trans="log2",
                     breaks=c(.01,.05,.1,.2,.5,1,2,5))+
  theme_bw() +
  labs(x="Coefficient in Model",y="Uses per Answer")+
  theme(legend.position = "none",
        axis.title=element_text(size=20),
        axis.text=element_text(size=16))

################################################
# Question 3
################################################

ecMain_test<-readRDS("earningsDat_3Y.RDS") %>%
  filter(FY==2012)

ecQA_test<-readRDS("earningsQandA_3Y.RDS")%>%
  mutate(wordcount=str_count(text,"[[:alpha:]]+")) %>%
  filter(callID%in%ecMain_test$callID)

ecMain_test_merged <- ecMain_test %>%
  # combine answer text as a single document and merge
  left_join(ecQA_test %>%
              filter(asker==0 & question <=10) %>%
              group_by(callID) %>%
              summarize(answertext=paste(text,collapse=" "),
                        answer_wdct=str_count(answertext,"[[:alpha:]]+"))) %>%
  # combine question text as a single document and merge
  left_join(ecQA_test %>%
              filter(asker==1 & question <=10) %>%
              group_by(callID) %>%
              summarize(questiontext=paste(text,collapse=" "),
                        question_wdct=str_count(questiontext,"[[:alpha:]]+")))

# Questions first

ecMain_test_dfm_q<-TMEF_dfm(ecMain_test_merged$questiontext,
                             min.prop=0) %>%
  dfm_match(colnames(ecMain_dfm_q)) %>%
  as.matrix()

test_EPS_predict_q<-predict(EPSmodel_q,
                            newx=ecMain_test_dfm_q,
                            s="lambda.min")

kendall_acc(test_EPS_predict_q,
            ecMain_test$EPS_actual)


# Answers next

ecMain_test_dfm_a<-TMEF_dfm(ecMain_test_merged$answertext,
                             min.prop=0) %>%
  dfm_match(colnames(ecMain_dfm_a)) %>%
  as.matrix()

test_EPS_predict_a<-predict(EPSmodel_a,
                            newx=ecMain_test_dfm_a,
                            s="lambda.min")

kendall_acc(test_EPS_predict_a,
            ecMain_test$EPS_actual)



################################################
# Question 4
################################################

# Join call-level data to the turn-level data
ecQA_merged<-ecQA %>%
  left_join(ecMain %>%
              select(callID,FY,FQ,EPS_actual,EPS_consens)) %>%
  mutate(first_quarter=1*(FQ==1)) %>%
  filter(asker==0)




ecQA_dfmx<-as.matrix(TMEF_dfm(ecQA_merged$text))

quartermodel<-cv.glmnet(x=ecQA_dfmx,
                        y=ecQA_merged$first_quarter)


#### Interpret with a coefficient plot
plotDat<-quartermodel %>%
  coef(s="lambda.min") %>%
  drop() %>%
  as.data.frame() %>%
  rownames_to_column(var = "ngram") %>%
  rename(score=".") %>%
  filter(score!=0 & ngram!="(Intercept)" & !is.na(score))  %>%
  # add ngram frequencies for plotting
  left_join(data.frame(ngram=colnames(ecQA_dfmx),
                       freq=colMeans(ecQA_dfmx)))

plotDat %>%
  ggplot(aes(x=score,y=freq,label=ngram,color=score)) +
  scale_color_gradient(low="blue",high="red")+
  geom_vline(xintercept=0)+
  geom_point() +
  geom_label_repel(max.overlaps = 20,force = 6)+  
  scale_y_continuous(trans="log2",
                     breaks=c(.01,.05,.1,.2,.5,1,2,5))+
  theme_bw() +
  labs(x="Coefficient in Model",y="Uses per Question")+
  theme(legend.position = "none",
        axis.title=element_text(size=20),
        axis.text=element_text(size=16))

################################################
# Question 5
################################################

table(ecMain$IBES_ID)

# FIX
ecMain_tiny <- ecMain %>%
  filter(IBES_ID%in%c("FSC","TMO","ONNN"))

ecQA_tiny <- ecQA %>%
  filter(callID%in%ecMain_tiny$callID)


spacyr::spacy_initialize()
ecQA_tiny_sp<-spacy_parse(ecQA_tiny$text,
                          nounphrase = TRUE,
                          lemma = FALSE,
                          dependency = FALSE,
                          pos = FALSE,
                          tag=FALSE)

head(ecQA_tiny_sp,20)

ecQA_ner<-spacy_extract_entity(ecQA_tiny$text)

ecQA_ner %>%
  filter(ent_type=="GPE") %>%
  with(rev(sort(table(text)))[1:10])

ecQA_ner <- ecQA_ner %>%
  uncount(length) %>%
  group_by(doc_id,start_id) %>%
  mutate(doc_token_id=start_id+0:(n()-1),
         first=1*(start_id==doc_token_id)) %>%
  ungroup() %>%
  mutate(text=str_replace_all(text," ","_")) %>%
  select(doc_id,ner_text="text",first,doc_token_id) 

ecQA_sp_ner <- ecQA_tiny_sp %>%
  group_by(doc_id) %>%
  # annoying that the nounphrase counts doc tokens, not sentence tokens
  # but we do what we must
  mutate(doc_token_id=1:n()) %>%
  ungroup()%>%
  left_join(ecQA_ner) %>%
  filter(is.na(ner_text)|first==1) %>%
  mutate(ner_token=ifelse(is.na(ner_text),token,ner_text)) %>%
  select(-first,-ner_text)

# generate a dfm from this

ecQA_ner_docs<-ecQA_sp_ner %>%
  group_by(doc_id) %>%
  summarize(text=paste(ner_token, collapse=" ")) %>%
  mutate(doc_id=as.numeric(str_replace_all(doc_id,"text",""))) %>%
  arrange(doc_id)# %>%
  left_join(ecQA_ner_docs)

# extract all the common noun phrases
Q1_phrases<-ecQA_ner_docs %>%
  bind_cols(ecQA_tiny %>%
              select(-text)) %>%
  left_join(ecMain_tiny) %>%
  filter(FQ==1) %>%
  pull(text) %>%
  tokens() %>%
  dfm() %>%
  convert(to="data.frame") %>%
  select(contains("_"),-doc_id) %>%
  colMeans() %>%
  sort(decreasing = T) %>%
  names()

Q1_phrases[1:50]


Q4_phrases<-ecQA_ner_docs %>%
  bind_cols(ecQA_tiny %>%
              select(-text)) %>%
  left_join(ecMain_tiny) %>%
  filter(FQ==4) %>%
  pull(text) %>%
  tokens() %>%
  dfm() %>%
  convert(to="data.frame") %>%
  select(contains("_"),-doc_id) %>%
  colMeans() %>%
  sort(decreasing = T) %>%
  names()

Q4_phrases[1:50]

