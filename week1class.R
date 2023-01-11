################################################
#
#     Text Mining for Economics and Finance
#
#          In-Class Activity - Week 1
#
#
################################################

# if you haven't already....
#install.packages("tidyverse")

library(tidyverse) 


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

# Let's convert to binary and save it

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

# First, we need to convert gender to numeric again

week1_answers<-week1_answers %>%
  mutate(genderTRUE=ifelse(genderTRUE=="male",1,0))

# Let's do the same pivot as before

week1_answers_long <- week1_answers %>%
  pivot_longer(-c(text,review_id),names_to="metric",values_to="answer")

# Note the metric names are not the same as above, so we can't join them yet
table(week1_answers_long$metric)

# we need to mutate that column to match the same labels 

# we could do some ifelse calls again....
week1_answers_long %>% 
  mutate(metric=ifelse(metric=="genderTRUE","gender",
                       ifelse(metric=="priceTRUE","price",
                              "stars"))) %>%
  head()

# For multiple ifelse branches, we can use case_when:
week1_answers_long<-week1_answers_long %>% 
  mutate(metric=case_when(
    metric=="genderTRUE" ~ "gender",
    metric=="priceTRUE" ~ "price",
    metric=="starsTRUE" ~ "stars"))

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

# That's all the programming for today! 
# Complete the rest of the assignment with your group and submit the results
