################################################
#
#     Text Mining for Economics & Finance
#
#          In-Class Activity - Week 7
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

source("TMEF_dfm.R")
source("kendall_acc.R")

################################################
################################################
#     an introduction to some spacy features
################################################


###################################
set.seed(2022)

rev_med<-readRDS("rev_med.RDS")

train_split=sample(1:nrow(rev_med),1000)

rev_med_train<-rev_med[train_split,]
rev_med_test<-rev_med[-train_split,]



spacy_install()


spacyr::spacy_initialize()


# Politeness

rev_med_train_polite<-politeness(rev_med_train$text,parser="spacy")

politenessPlot(rev_med_train_polite,
               rev_med_train$stars,
               middle_out = .05) +
  theme(panel.grid = element_blank())


rev_tiny <- rev_med %>%
  slice(1:200)



rev_tiny_sp<-spacy_parse(rev_tiny$text,
                         nounphrase = T,
                         lemma = T,
                         dependency = T,
                         pos = T,
                         tag=T)

head(rev_tiny_sp,20)

##################################################
# Use lemmas instead of stems!
##################################################

# recreate documents from the lemmas
rev_lemma_docs<-rev_tiny_sp %>%
  group_by(doc_id) %>%
  summarize(text=paste(lemma, collapse=" ")) %>%
  mutate(doc_id=as.numeric(str_replace_all(doc_id,"text",""))) %>%
  arrange(doc_id)

#extract lemmas as words from the document
lemmas<-rev_lemma_docs$text %>%
  tokens() %>%
  tokens_select(pattern = stopwords("en"), 
                selection = "remove") %>%
  dfm() %>%
  colMeans() %>%
  sort(decreasing=TRUE) %>%
  names()

# the normal approach of stemming
stems<-TMEF_dfm(rev_lemma_docs$text) %>%
  colMeans() %>%
  sort(decreasing=TRUE) %>%
  names()

#lots of shortened non-words
stems[!stems%in%lemmas][1:100]

#this makes sense at least
lemmas[!lemmas%in%stems][1:100]

##################################################
# Negation Scoping
##################################################

# requires spacyr and sentimentr
negation_scoper<-function(text,lemmas=T){
  text_p<-spacy_parse(text,lemma =lemmas,entity = F,pos=F)
  text_p<-text_p %>%
    mutate(negation=1*(token%in%c(lexicon::hash_valence_shifters %>%
                                    filter(y==1) %>%
                                    pull(x))),
           clause_end=1*grepl("^[,.:;!?]$",token)) %>%
    group_by(sentence_id) %>%
    mutate(clause_count=cumsum(clause_end)) %>%
    group_by(doc_id,sentence_id,clause_count) %>%
    mutate(negated=cumsum(negation)-negation) %>%
    ungroup()
  if(lemmas){
    text<-text_p %>%
      mutate(lemma_neg=ifelse(negated==0,lemma,
                              paste0(lemma,"_NEG"))) %>%
      mutate(doc_id=as.numeric(gsub("text","",doc_id))) %>% 
      arrange(doc_id) %>%
      group_by(doc_id) %>%
      summarize(text=paste(lemma_neg,collapse=" ")) %>%
      pull(text)
  } else{
    text<-text_p  %>%
      mutate(token_neg=ifelse(negated==0,token,
                              paste0(token,"_NEG"))) %>%
      mutate(doc_id=as.numeric(gsub("text","",doc_id))) %>% 
      arrange(doc_id) %>%
      mutate(token_neg=ifelse(grepl("'",token),token_neg,paste0(" ",token_neg))) %>%
      group_by(doc_id) %>%
      summarize(text=paste(token_neg,collapse="")) %>%
      pull(text)
  }
  return(text)
}

negation_scoper(c("the food is not very good for the price",
                  "another test of the scoper: it's not too bad, is it?"),
                lemmas=F)


##################################################
# Using POS tags to disambiguate words
##################################################

# words with two senses
two_senses<-rev_tiny_sp %>%
  group_by(token,pos) %>%
  summarize(pos_ct=n()) %>%
  left_join(rev_tiny_sp %>%
              group_by(token) %>%
              summarize(all_ct=n())) %>%
  mutate(pos_ratio=pos_ct/all_ct) %>%
  filter(all_ct>5) %>%
  filter(pos_ratio>.2 & pos_ratio<.8) %>%
  as.data.frame()

# a few examples of words with multiple POS
two_senses

rev_sp_tagged <- rev_tiny_sp %>%
  left_join(two_senses %>%
              mutate(token_tag=paste0(token,"_",pos)) %>%
              select(token,pos,token_tag)) %>%
  mutate(tagged_tokens=ifelse(is.na(token_tag),token,token_tag))

# create a dfm from this
rev_tagged_docs<-rev_sp_tagged %>%
  group_by(doc_id) %>%
  summarize(text=paste(tagged_tokens, collapse=" ")) %>%
  mutate(doc_id=as.numeric(str_replace_all(doc_id,"text",""))) %>%
  arrange(doc_id)

TASSL_dfm(rev_tagged_docs$text) %>%
  colnames() %>%
  sort()


##################################################
# named entity recognition
##################################################

rev_ner<-spacy_extract_entity(rev_tiny$text)

rev_ner %>%
  filter(ent_type=="GPE") %>%
  with(rev(sort(table(text))))

rev_ner <- rev_ner %>%
  uncount(length) %>%
  group_by(doc_id,start_id) %>%
  mutate(doc_token_id=start_id+0:(n()-1),
         first=1*(start_id==doc_token_id)) %>%
  ungroup() %>%
  mutate(text=str_replace_all(text," ","_")) %>%
  select(doc_id,ner_text="text",first,doc_token_id) 

rev_sp_ner <- rev_tiny_sp %>%
  group_by(doc_id) %>%
  # annoying that the nounphrase counts doc tokens, not sentence tokens
  # but we do what we must
  mutate(doc_token_id=1:n()) %>%
  ungroup()%>%
  left_join(rev_ner) %>%
  filter(is.na(ner_text)|first==1) %>%
  mutate(ner_token=ifelse(is.na(ner_text),token,ner_text)) %>%
  select(-pos,-tag,-head_token_id,-first,-dep_rel,-nounphrase,-ner_text)

# generate a dfm from this

rev_ner_docs<-rev_sp_ner %>%
  group_by(doc_id) %>%
  summarize(text=paste(ner_token, collapse=" ")) %>%
  mutate(doc_id=as.numeric(str_replace_all(doc_id,"text",""))) %>%
  arrange(doc_id)

# extract all the common noun phrases
phrases<-TASSL_dfm(rev_ner_docs$text,
                   min.prop = .001) %>%
  convert(to="data.frame") %>%
  select(contains("_"),-doc_id) %>%
  colMeans() %>%
  sort(decreasing = T) %>%
  names()

phrases[1:100]

rm(rev_sp_ner,rev_lemma_docs,
   rev_sp_tagged)

##################################################
# Extracting motifs 
##################################################

extract_motifs(tokens = rev_tiny_sp,
               parse_multi_token_entities = T,
               entities = c("server","waiter","waitress"),
               add_sentence = T,
               markup = T)



extract_motifs(tokens = rev_tiny_sp,
               parse_multi_token_entities = T,
               entities = c("husband","wife","spouse","partner"),
               add_sentence = T,
               markup = T)


extract_motifs(tokens = rev_tiny_sp,
               parse_multi_token_entities = T,
               entities = c("Toronto"),
               add_sentence = T,
               markup = T)


#######################################################
#######################################################
#######################################################
#     THIS WEEK'S DATA
#######################################################
#######################################################
#######################################################


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
featurePlot(ecQA_polite,
               ecQA$asker,
               split_levels = c("Answerer","Questioner"),
               middle_out = .01) 

# Add the politeness count columns to the main Q&A dataset
ecQA_pol <- bind_cols(ecQA,ecQA_polite)


