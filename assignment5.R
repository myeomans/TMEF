################################################
#
# Text Mining for Economics & Finance
#
#              Assignment 5
#
#
################################################

library(quanteda)
library(ggrepel)
library(textclean)
library(tidyverse)
library(glmnet)
library(sentimentr)
library(spacyr) # a new one!
library(politeness) # another new one! And a package I wrote :)
library(semgram) # also new - for motifs

source("MLTASS_dfm.R")
source("kendall_acc.R")

################################################
################################################
#     an introduction to some spacy features
################################################

<<<<<<< HEAD
#spacyr::spacy_install()
=======
>>>>>>> eb9ffddabe2bb94fb542326aca40eedad793335f

###################################
set.seed(02138)

rev_med<-readRDS("data/rev_med.RDS")

train_split=sample(1:nrow(rev_med),1000)

rev_med_train<-rev_med[train_split,]
rev_med_test<-rev_med[-train_split,]


spacyr::spacy_initialize()


# Politeness

rev_med_train_polite<-politeness(rev_med_train$text,parser="spacy")

featurePlot(rev_med_train_polite,
            rev_med_train$stars,
            split_levels = c("Low","High"),
            split_name = "Stars",
            middle_out = .05) +
  theme(panel.grid = element_blank())


p_model<-cv.glmnet(x=as.matrix(rev_med_train_polite),
                   y=rev_med_train$stars)

rev_med_test_polite<-politeness(rev_med_test$text,parser="spacy")

test_predict<-predict(p_model,
                      newx = rev_med_test_polite)[,1]

acc_dfm<-kendall_acc(rev_med_test$stars,test_predict)

acc_dfm


rev_tiny <- rev_med %>%
  slice(1:500)



rev_tiny_sp<-spacy_parse(rev_tiny$text,
                         nounphrase = T,
                         lemma = T,
                         dependency = T,
                         pos = T,
                         tag=T)

head(rev_tiny_sp,20)

# Save the output of slow-executing code!!
saveRDS(rev_tiny_sp,"rev_tiny_sp.RDS")

rev_tiny_sp<-readRDS("rev_tiny_sp.RDS")

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
stems<-MLTASS_dfm(rev_lemma_docs$text) %>%
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

MLTASS_dfm(rev_tagged_docs$text) %>%
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
phrases<-MLTASS_dfm(rev_ner_docs$text,
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


partners<-extract_motifs(tokens = rev_tiny_sp,
               parse_multi_token_entities = T,
               entities = c("husband","wife"),
               add_sentence = T,
               markup = T)


head(partners$actions)

partners$actions %>%
  group_by(action,Entity) %>%
  summarize(n=n()) %>%
  pivot_wider(names_from="Entity",values_from="n") %>%
  mutate_all(~replace_na(.,0)) %>%
  mutate(total=husband+wife,
         tilt=husband/(husband+wife)) %>%
  arrange(-tilt)


