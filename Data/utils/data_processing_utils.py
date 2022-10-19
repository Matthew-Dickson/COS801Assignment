import numpy as np
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import string
import re


#Print statistics
def get_stats(var):
    print("Min:", np.min(var))
    print("Max:", np.max(var))
    print("Mean:", np.mean(var))
    print("Median", np.median(var))
    print("1st percentile", np.percentile(var, 1))
    print("95th percentile", np.percentile(var, 95))
    print("99th percentile", np.percentile(var, 99))
    print("99.5th Percentile", np.percentile(var, 99.5))
    print("99.9th Percentile", np.percentile(var, 99.9))


#Plot sns displot
def plot_snsplot(array,title,xlabel,ylabel,save_file_name):
    sns.distplot(array, kde = False, bins = 70, color = 'blue').set_title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(0, 100)
    plt.savefig(save_file_name)


#Gets word count, character count and average length of texts
def getTextMetaInformation(data,label):
    word_count = []
    char_count = []

    text = list(data[label].values)
    for i in range(len(text)):
        word_count.append(len(text[i].split()))
        char_count.append(len(text[i]))

    # Convert lists to numpy arrays
    word_count = np.array(word_count)
    char_count = np.array(char_count)

    # Calculate average word lengths
    ave_length = np.array(char_count)/np.array(word_count)
    return word_count, char_count, ave_length


#Get outlier indexes 
def get_outlier_indexes(count,size,type):
    if(type=="less"):
        outliers_indexes = np.where(count < size)
    elif(type=="greater"):
        outliers_indexes = np.where(count > size)

    return outliers_indexes


#print unique character counts
def print_unique_character_counts(text):
    text_string = ''

    for i in range(len(text)):
        text_string += text[i].lower()

    # Get character frequencies
    char_cnt = Counter(text_string)
    print(char_cnt)
    print(len(char_cnt))
    return char_cnt


# Find all texts containing unusual characters
def find_character_list_count(text,character_list):
    matching_text = []
    for i in range(len(text)):
        for j in text[i]:
            if j in character_list:
                matching_text.append(i)
        
    matching_text = list(set(matching_text))
    print('There are', str(len(matching_text)), 'texts containing the characters provided.')
    return matching_text



# Remove invalid characters from text
def remove_character_from_text(text,remove_characters):
    for char in remove_characters:
            text = [excerpt.replace(char, '') for excerpt in text]
    return text


#Remove blocks of white space
def remove_block_of_white_space(text):
    new_text = []
    for excerpt in text:
        while "  " in excerpt:
            excerpt = excerpt.replace("  "," ")
        new_text.append(excerpt)
    print(len(text))
    return new_text



#Count text with particular string
def count_text_with_string(text,string_value):
    count= text.str.count(string_value).sum()
    return count


#Remove punctuation
def remove_punctuation(text):
    new_text = []
    for i in range(len(text)):
        new = text[i].lower()
        new = new.translate(str.maketrans('','', string.punctuation))
        new = new.replace('“', '').replace('”', '')
        new_text.append(new)
    print(len(new_text))

    return new_text


#Make text lower case
def make_text_lower(text):
    new_text = []
    for i in range(len(text)):
        new = text[i].lower()
        new_text.append(new)
    print(len(new_text))
    return new_text


#Remove text with substrings
def remove_text_with_substring(data,text,substring):
    df = data[text.str.contains(substring) == False]
    return df


#Removes urls from string
def remove_url(txt):
    return " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", txt).split())
