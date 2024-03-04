################################################
#
#     Text Mining for Economics & Finance
#
#          In-Class Activity - Week 7
#           
#
################################################



# Run these every time
library(quanteda)
library(ggrepel)
library(textclean)
library(tidyverse)
library(glmnet)
library(spacyr) # a new one!
library(politeness) # another new one! And a package I wrote :)

source("TMEF_dfm.R")
source("kendall_acc.R")

# filter to only look at calls from fiscal quarter 3
ecMain<-readRDS("earningsDat_3Y.RDS") %>%
  filter(FQ==3)

# filter the Q&A to match the filtering on the main dataset
ecQA<-readRDS("earningsQandA_3Y.RDS")%>%
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


# This is how you would normally produce the politeness features....

# Very slow...
# ecQA_polite<-politeness(ecQA$text,parser="spacy")

# Not quite as slow, due to parallel processing 
# doMC is for Macs - Windows users should try the doParallel package
# doMC::registerDoMC(10)
# ecQA_polite<-politeness(ecQA$text,parser="spacy",num_mc_cores = 8)
# 
# # Either way, we save the data because it takes a long time to generate
# saveRDS(ecQA_polite,"ecQA_polite.RDS")

# Here's a pre-saved version you can use instead
ecQA_polite<-readRDS("ecQA_polite.RDS")

# dialogue acts from each text
head(ecQA_polite)

# This produces a politeness plot to compare the text of questions and answers
politenessPlot(ecQA_polite,
               ecQA$asker,
               split_levels = c("Answerer","Questioner"),
               middle_out = .01) 

# Add the politeness count columns to the main Q&A dataset
ecQA_pol <- bind_cols(ecQA,ecQA_polite)


################################################
#     an introduction to some spacy features
################################################
spacy_install()


ecMain_tiny <- ecMain %>%
  slice(1:20)

ecQA_tiny <- ecQA %>%
  filter(callID%in%ecMain_tiny$callID)


spacyr::spacy_initialize()

ecQA_tiny_sp<-spacy_parse(ecQA_tiny$text,
                     nounphrase = T,
                     lemma = T,
                     dependency = T,
                     pos = T,
                     tag=T)

head(ecQA_tiny_sp,20)

##################################################
# Use lemmas instead of stems!
##################################################

# recreate documents from the lemmas
ecQA_lemma_docs<-ecQA_tiny_sp %>%
  group_by(doc_id) %>%
  summarize(text=paste(lemma, collapse=" ")) %>%
  mutate(doc_id=as.numeric(str_replace_all(doc_id,"text",""))) %>%
  arrange(doc_id)

#extract lemmas as words from the document
lemmas<-ecQA_lemma_docs$text %>%
  tokens() %>%
  tokens_select(pattern = stopwords("en"), 
                selection = "remove") %>%
  dfm() %>%
  colMeans() %>%
  sort(decreasing=TRUE) %>%
  names()

# the normal approach of stemming
stems<-TMEF_dfm(ecQA_lemma_docs$text) %>%
  colMeans() %>%
  sort(decreasing=TRUE) %>%
  names()

#lots of shortened non-words
stems[!stems%in%lemmas][1:100]

#this makes sense at least
lemmas[!lemmas%in%stems][1:100]

##################################################
# Using POS tags to disambiguate words
##################################################

# words with two senses
two_senses<-ecQA_tiny_sp %>%
  group_by(token,pos) %>%
  summarize(pos_ct=n()) %>%
  left_join(ecQA_tiny_sp %>%
              group_by(token) %>%
              summarize(all_ct=n())) %>%
  mutate(pos_ratio=pos_ct/all_ct) %>%
  filter(all_ct>100) %>%
  filter(pos_ratio>.2 & pos_ratio<.8) %>%
  as.data.frame()

# a few examples of words with multiple POS
two_senses %>%
  head(100)

ecQA_sp_tagged <- ecQA_tiny_sp %>%
  left_join(two_senses %>%
              mutate(token_tag=paste0(token,"_",pos)) %>%
              select(token,pos,token_tag)) %>%
  mutate(tagged_tokens=ifelse(is.na(token_tag),token,token_tag))

# create a dfm from this
ecQA_tagged_docs<-ecQA_sp_tagged %>%
  group_by(doc_id) %>%
  summarize(text=paste(tagged_tokens, collapse=" ")) %>%
  mutate(doc_id=as.numeric(str_replace_all(doc_id,"text",""))) %>%
  arrange(doc_id)

TMEF_dfm(ecQA_tagged_docs$text) %>%
  colnames() %>%
  sort()


##################################################
# named entity recognition
##################################################

ecQA_ner<-spacy_extract_entity(ecQA_tiny$text)

ecQA_ner %>%
  filter(ent_type=="GPE") %>%
  with(rev(sort(table(text))))

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
  select(-pos,-tag,-head_token_id,-first,-dep_rel,-nounphrase,-ner_text)

# generate a dfm from this

ecQA_ner_docs<-ecQA_sp_ner %>%
  group_by(doc_id) %>%
  summarize(text=paste(ner_token, collapse=" ")) %>%
  mutate(doc_id=as.numeric(str_replace_all(doc_id,"text",""))) %>%
  arrange(doc_id)

# extract all the common noun phrases
phrases<-TMEF_dfm(ecQA_ner_docs$text,
                  min.prop = .001) %>%
  as.data.frame() %>%
  select(contains("_"),-doc_id) %>%
  colMeans() %>%
  sort(decreasing = T) %>%
  names()

phrases[1:100]

rm(ecQA_pol,ecQA_polite,
   ecQA_sp_ner,ecQA_lemma_docs,
   ecQA_sp_tagged)

