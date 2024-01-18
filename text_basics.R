
library(tidyverse)

######################################################################
# Some basic text manipulation
######################################################################


# paste() joins text - default setting is to put a space in the middle

paste("x","y")

# paste0 changes the default to put nothing in the middle

paste0("x","y")

# paste handles vectors - if they are the same length, one item from the first 
# argument will be pasted to one from the second, and so on down the line

paste0(c("x","y","z"),
       c("a","b"))

# If the second argument is only one item long, 
# it is pasted to every item in the first argument

paste0(c("x","y"),
       c("a"))

# If they are unequal lengths you can get strange behavior - not recommended!
paste0(c("x","y","z"),
       c("a","b"))

##############################

toolong="this string is good but it is too long"

# This counts characters
toolong %>%
  str_length()

# This counts words - the funny thing in the bracket is a "regular expression"
# REs are a bit advanced for this class.. For now just memorize this one
# It basically means "the number of blocks of consecutive alphabetical characters"
toolong %>%
  str_count("[[:alpha:]]+")


# str_sub() helps us chop up a strings
toolong %>%
  str_sub(0,19)

# negatives count from the end
toolong %>%
  str_sub(20,-1)

toolong %>%
  str_sub(-9,-1)

##############################

# str_replace substitutes one character string for another


c("abcx","defxx","ghiy") %>%
  str_replace("x","y")

c("abc*","defx*","ghiy") %>%
  str_replace("*","y")

# what happened? Again, this function uses "regular expressions"

# If you just want the literal characters, you need to "escape" them with \\
c("abc*","defx*","ghiy") %>%
  str_replace("\\*","y")

# We can delete character strings by replacing them with an empty string

c("abc","def","rabbit") %>%
  str_replace("b","")

# Notice it only replaces the first instance... to replace all, we usd a different function

c("abc","def","rabbit") %>%
  str_replace_all("b","")


c("before","abetting","in my bed") %>%
  str_replace_all("be","")


##############################

# str_detect() tells us whether a string is contained within another

c("teases","taxes","xylophone") %>%
  str_detect("x")

# It is case sensitive
c("Before","abetting","in my bed") %>%
  str_detect("be")

# We can convert to lowercase first
c("Before","abetting","in my bed") %>%
  str_to_lower() %>%
  str_detect("be")


# We can also get counts using str_count()

c("Before","abetting","in my bed") %>%
  str_count("be")

c("Before","abetting","in my bed") %>%
  str_count("e")

c("Before","abetting","in my bed") %>%
  str_count("t")
