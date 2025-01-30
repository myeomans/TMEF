
library(tidyverse)
library(ggrepel)
library(ggridges)
##############################################
# Pre-course survey data
##############################################

sd<-read.csv("surveydat.csv")

saveRDS(sd,file="surveyDat.RDS")

# Load the data, calculate word counts
surveys<-readRDS("surveyDat.RDS") %>%
  mutate(why_join_wdct=str_count(why_join,"[[:alpha:]]+"),
         future_plans_wdct=str_count(future_plans,"[[:alpha:]]+"),
         group_project_wdct=str_count(group_project,"[[:alpha:]]+")
  ) 


# ggplot has easy histograms
surveys %>%
  ggplot(aes(x=stay_in_class)) +
  geom_histogram()

# most people wanted to stay! that's nice.

# let's change the number ratings to words
# create a new variable called "stay" 
# and create the mapping with factor labels

surveys <- surveys %>%
  mutate(stay=factor(stay_in_class,ordered=TRUE,
                     levels=4:8,
                     labels=c("not sure",
                              "slightly sure",
                              "somewhat sure",
                              "very sure",
                              "extremely sure")
                     )) 

# since they are categories now, we can use a bar plot
surveys %>%
  ggplot(aes(x=stay)) +
  geom_bar()

# we can see there are empty cells now! 
# Let's filter them out of the data

surveys <- surveys %>%
  filter(!is.na(stay_in_class))

# Let's make them different colors, flip the axis and add axis labels
surveys %>%
  ggplot(aes(x=stay,fill=stay)) +
  geom_bar() +
  coord_flip() +
  labs(x="Will you stay in the course?",
       y="Number of Students") 

# let's put the question as the title so it is horizontal,
# make the words bigger, and get rid of the legend

surveys %>%
  ggplot(aes(x=stay,fill=stay)) +
  geom_bar() +
  coord_flip() +
  ggtitle("Will you stay in the course?") +
  labs(x="",
       y="Number of Students") +
  theme(legend.position = "none",
        title =element_text(size=20),
        axis.text=element_text(size=15),
        axis.title=element_text(size=20))

# instead of a screenshot, we save it to a file
ggsave("stay in course.png")

####################################

# is this variable correlated with whether they know RStudio?

surveys %>%
  ggplot(aes(x=stay_in_class,y=tools_RStudio)) +
  geom_point()

# the points overlap, so we use jitter

surveys %>%
  ggplot(aes(x=jitter(stay_in_class),y=jitter(tools_RStudio))) +
  geom_point()

# is there a relationship? let's fit a trend line
surveys %>%
  ggplot(aes(x=jitter(stay_in_class),y=jitter(tools_RStudio))) +
  geom_point() +
  geom_smooth()

# we don't have enough points for a curve, let's do a straight line
surveys %>%
  ggplot(aes(x=jitter(stay_in_class),y=jitter(tools_RStudio))) +
  geom_point() +
  geom_smooth(method="lm")


# Let's change the color, axis labels, and the background
surveys %>%
  ggplot(aes(x=jitter(stay_in_class),y=jitter(tools_RStudio))) +
  geom_point() +
  geom_smooth(method="lm",
              color="purple",
              fill="orange") +
  theme_bw() +
  labs(x="Will you stay in class?",
       y="Do you have RStudio experience?")
  
# bigger axis titles, and get rid of the grid
surveys %>%
  ggplot(aes(x=jitter(stay_in_class),y=jitter(tools_RStudio))) +
  geom_point() +
  geom_smooth(method="lm",
              color="purple",
              fill="orange") +
  theme_bw() +
  labs(x="Will you stay in class?",
       y="Do you have RStudio experience?") +
  theme(panel.grid=element_blank(),
        axis.title=element_text(size=15))

ggsave("Rstudio.png")


########################################
# let's look at what topics people know about

surveys %>%
  select(starts_with("know_"))

# we need to convert from wide to long format
surveylong<-surveys %>%
  select(ResponseId,starts_with("know_")) %>%
  pivot_longer(-ResponseId,
               values_to="knowledge",
               names_to="topic") 

surveylong

# remove the "know_" from the topic names
surveylong<-surveylong%>%
  mutate(topic=gsub("know_","",topic,fixed=T))

surveylong

# some of these names are bad
unique(surveylong$topic)

#let's fix them with case_when()

surveylong<- surveylong %>%
  mutate(topic_named=case_when(
    # if the part left of the ~ is TRUE, use the right part as the value
    topic == "crossval"~ "cross-validation", 
    topic == "supervised"~ "supervised learning", 
    topic == "dialogact"~ "dialogue acts", 
    topic == "topics"~ "topic models", 
    TRUE ~ topic # Finally, if none of the above are true, use this value
  ))


# not that helpful... no colors or labels and too much overlap
surveylong %>%
  ggplot(aes(x=knowledge,group=topic_named)) +
  geom_density(alpha=.2)

# better
surveylong %>%
  ggplot(aes(x=knowledge,y=topic_named)) +
  geom_density_ridges()

#needs colors, axis titles, etc.
surveylong %>%
  ggplot(aes(x=knowledge,y=topic_named,
             fill=topic_named)) +
  geom_density_ridges() +
  theme_bw() +
  labs(x="Topic Knowledge",
       y="") +
  theme(legend.position="none",
        panel.grid=element_blank(),
        axis.text = element_text(size=20),
        axis.title = element_text(size=24))

##### Let's put them in order

# first, calculate the order from low to high
topic_order=surveylong %>%
  group_by(topic_named) %>%
  summarize(tOrder=mean(knowledge))

# ugh, there are NAs again!!
head(topic_order)

topic_order=surveylong %>%
  filter(!is.na(knowledge)) %>%
  group_by(topic) %>%
  summarize(tOrder=mean(knowledge))


surveylong %>%
  # merge order data
  left_join(topic_order) %>%
  # use the order column to re-order topicID as a factor
  mutate(topic_named=fct_reorder(topic_named,tOrder)) %>%
  ggplot(aes(x=knowledge,fill=topic_named,
             y=topic_named)) +
  geom_density_ridges() +
  theme_bw() +
  theme(legend.position="none")

ggsave("topic_knowledge.png")

#################################################################

reviews<-readRDS("rev_med.RDS") %>%
  mutate(wordcount=str_count(text,"[[:alpha:]]+"))

# let's plot the effect of word count on star rating

reviews %>%
  ggplot(aes(x=wordcount,y=stars)) +
  geom_point() +
  geom_smooth()

# looks weird, let's flip the axes

reviews %>%
  ggplot(aes(x=stars,y=wordcount)) +
  geom_point() +
  geom_smooth(method="lm")

# still weird... let's do a separate point for each star rating


reviews %>%
  group_by(stars) %>%
  summarize(avg=mean(wordcount),
            se=sd(wordcount)/sqrt(n())) %>%
  ggplot(aes(x=stars,y=avg,
             ymin=avg-se,ymax=avg+se)) +
  geom_point() +
  geom_errorbar()

# maybe the effect is different for different price levels?

reviews %>%
  group_by(stars,price) %>%
  summarize(avg=mean(wordcount),
            se=sd(wordcount)/sqrt(n())) %>%
  ggplot(aes(x=stars,y=avg,
             ymin=avg-se,ymax=avg+se)) +
  geom_point() +
  geom_errorbar()

# what a mess! we can connect  points with colors and lines

reviews %>%
  group_by(stars,price) %>%
  summarize(avg=mean(wordcount),
            se=sd(wordcount)/sqrt(n())) %>%
  ggplot(aes(x=stars,y=avg,
             ymin=avg-se,ymax=avg+se,
             color=price,group=price)) +
  geom_point() +
  geom_errorbar() +
  geom_line()

# still messy... let's make price a factor and narrower error bars

reviews %>%
  group_by(stars,price) %>%
  summarize(avg=mean(wordcount),
            se=sd(wordcount)/sqrt(n())) %>%
  mutate(price=factor(price)) %>%
  ggplot(aes(x=stars,y=avg,
             ymin=avg-se,ymax=avg+se,
             color=price,group=price)) +
  geom_point() +
  geom_errorbar(width=.2) +
  geom_line()

# let's add a fourth variable -is the user experienced?

# it's continuous so let's break up into four groups

reviews <- reviews %>%
  mutate(user_exp=factor(ntile(user_review_count,3),
                         ordered=T,
                         levels=1:3,
                         labels=c("New Users","Some Experience","Most Experienced")))

# ntile creates equal groups from the distribution
reviews %>%
  group_by(user_exp) %>%
  summarize(mean(user_review_count))

# We can use different shapes but there are still too many lines
reviews %>%
  group_by(stars,price,user_exp) %>%
  summarize(avg=mean(wordcount),
            se=sd(wordcount)/sqrt(n())) %>%
  mutate(price=factor(price)) %>%
  ggplot(aes(x=stars,y=avg,
             ymin=avg-se,ymax=avg+se,
             color=price,group=price,
             shape=user_exp
              )) +
  geom_point() +
  geom_errorbar(width=.2) +
  geom_line()

#Instead, let's use facets!! Multiple plots at once :)

reviews %>%
  group_by(stars,price,user_exp) %>%
  summarize(avg=mean(wordcount),
            se=sd(wordcount)/sqrt(n())) %>%
  mutate(price=factor(price)) %>%
   ggplot(aes(x=stars,y=avg,
             ymin=avg-se,ymax=avg+se,
             color=price,group=price,
  )) +
  geom_point() +
  geom_errorbar(width=.2) +
  geom_line() +
  facet_wrap(~user_exp)

# let's clean up the labels fix the background, and move the legend

# not enough data for the top price so let's get rid of it

reviews %>%
  filter(price!=4) %>%
  group_by(stars,price,user_exp) %>%
  summarize(avg=mean(wordcount),
            se=sd(wordcount)/sqrt(n())) %>%
  mutate(price=factor(price)) %>%
  ggplot(aes(x=stars,y=avg,
             ymin=avg-se,ymax=avg+se,
             color=price,group=price,
  )) +
  geom_point() +
  geom_errorbar(width=.2) +
  geom_line() +
  theme_bw() +
  facet_wrap(~user_exp) +
  labs(x="Star Rating",y="Word Count",
       color="Price") +
  theme(axis.title=element_text(size=20),
        axis.text=element_text(size=14),
        strip.text=element_text(size=16),
        legend.title=element_text(size=20),
        legend.text=element_text(size=14),
        legend.position="top")

ggsave("review_experience.png")
