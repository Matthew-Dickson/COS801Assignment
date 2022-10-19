import textblob
from textblob import TextBlob

# functions for separating the POS Tags
def tagCC(text):
    blob = TextBlob(text)
    count=0
    for (word,tag) in blob.tags:
        if tag == 'CC':
            count += 1
    return count

def tagCD(text):
    blob = TextBlob(text)
    count=0
    for (word,tag) in blob.tags:
        if tag == 'CD':
            count += 1
    return count

def tagDT(text):
    blob = TextBlob(text)
    count=0
    for (word,tag) in blob.tags:
        if tag == 'DT':
            count += 1
    return count

def tagEX(text):
    blob = TextBlob(text)
    count=0
    for (word,tag) in blob.tags:
        if tag == 'EX':
            count += 1
    return count        



def tagFW(text):
    blob = TextBlob(text)
    count=0
    for (word,tag) in blob.tags:
        if tag == 'FW':
            count += 1
    return count

def tagIN(text):
    blob = TextBlob(text)
    count=0
    for (word,tag) in blob.tags:
        if tag == 'IN':
            count += 1
    return count

def tagJJ(text):
    blob = TextBlob(text)
    count=0
    for (word,tag) in blob.tags:
        if tag == 'JJ':
            count += 1
    return count    


def tagJJR(text):
    blob = TextBlob(text)
    count=0
    for (word,tag) in blob.tags:
        if tag == 'JJR':
            count += 1
    return count    

def tagJJS(text):
    blob = TextBlob(text)
    count=0
    for (word,tag) in blob.tags:
        if tag == 'JJS':
            count += 1
    return count    

def tagLS(text):
    blob = TextBlob(text)
    count=0
    for (word,tag) in blob.tags:
        if tag == 'LS':
            count += 1
    return count   

def tagMD(text):
    blob = TextBlob(text)
    count=0
    for (word,tag) in blob.tags:
        if tag == 'MD':
            count += 1
    return count   

def tagNN(text):
    blob = TextBlob(text)
    count=0
    for (word,tag) in blob.tags:
        if tag == 'NN':
            count += 1
    return count  

def tagNNS(text):
    blob = TextBlob(text)
    count=0
    for (word,tag) in blob.tags:
        if tag == 'NNS':
            count += 1
    return count   

def tagNNP(text):
    blob = TextBlob(text)
    count=0
    for (word,tag) in blob.tags:
        if tag == 'NNP':
            count += 1
    return count   

def tagNNPS(text):
    blob = TextBlob(text)
    count=0
    for (word,tag) in blob.tags:
        if tag == 'NNPS':
            count += 1
    return count   


def tagPDT(text):
    blob = TextBlob(text)
    count=0
    for (word,tag) in blob.tags:
        if tag == 'PDT':
            count += 1
    return count  

def tagPOS(text):
    blob = TextBlob(text)
    count=0
    for (word,tag) in blob.tags:
        if tag == 'POS':
            count += 1
    return count  


def tagPRP(text):
    blob = TextBlob(text)
    count=0
    for (word,tag) in blob.tags:
        if tag == 'PRP':
            count += 1
    return count  

def tagPRP2(text):
    blob = TextBlob(text)
    count=0
    for (word,tag) in blob.tags:
        if tag == 'PRP$':
            count += 1
    return count 


def tagRB(text):
    blob = TextBlob(text)
    count=0
    for (word,tag) in blob.tags:
        if tag == 'RB':
            count += 1
    return count 

def tagRBR(text):
    blob = TextBlob(text)
    count=0
    for (word,tag) in blob.tags:
        if tag == 'RBR':
            count += 1
    return count 

def tagRBS(text):
    blob = TextBlob(text)
    count=0
    for (word,tag) in blob.tags:
        if tag == 'RBS':
            count += 1
    return count 

def tagRP(text):
    blob = TextBlob(text)
    count=0
    for (word,tag) in blob.tags:
        if tag == 'RP':
            count += 1
    return count 


def tagSYM(text):
    blob = TextBlob(text)
    count=0
    for (word,tag) in blob.tags:
        if tag == 'SYM':
            count += 1
    return count 

def tagTO(text):
    blob = TextBlob(text)
    count=0
    for (word,tag) in blob.tags:
        if tag == 'TO':
            count += 1
    return count 


def tagUH(text):
    blob = TextBlob(text)
    count=0
    for (word,tag) in blob.tags:
        if tag == 'UH':
            count += 1
    return count 


def tagVB(text):
    blob = TextBlob(text)
    count=0
    for (word,tag) in blob.tags:
        if tag == 'VB':
            count += 1
    return count 


def tagVBD(text):
    blob = TextBlob(text)
    count=0
    for (word,tag) in blob.tags:
        if tag == 'VBD':
            count += 1
    return count 


def tagVBG(text):
    blob = TextBlob(text)
    count=0
    for (word,tag) in blob.tags:
        if tag == 'VBG':
            count += 1
    return count 



def tagVBN(text):
    blob = TextBlob(text)
    count=0
    for (word,tag) in blob.tags:
        if tag == 'VBN':
            count += 1
    return count 

def tagVBP(text):
    blob = TextBlob(text)
    count=0
    for (word,tag) in blob.tags:
        if tag == 'VBP':
            count += 1
    return count 


def tagVBZ(text):
    blob = TextBlob(text)
    count=0
    for (word,tag) in blob.tags:
        if tag == 'VBZ':
            count += 1
    return count 

def tagWDT(text):
    blob = TextBlob(text)
    count=0
    for (word,tag) in blob.tags:
        if tag == 'WDT':
            count += 1
    return count 

def tagWP(text):
    blob = TextBlob(text)
    count=0
    for (word,tag) in blob.tags:
        if tag == 'WP':
            count += 1
    return count 


def tagWP2(text):
    blob = TextBlob(text)
    count=0
    for (word,tag) in blob.tags:
        if tag == 'WP$':
            count += 1
    return count 

def tagWRB(text):
    blob = TextBlob(text)
    count=0
    for (word,tag) in blob.tags:
        if tag == 'WRB':
            count += 1
    return count 


def tagComma(text):
    blob = TextBlob(text)
    count=0
    for (word,tag) in blob.tags:
        if tag == ',':
            count += 1
    return count 

def tagColon(text):
    blob = TextBlob(text)
    count=0
    for (word,tag) in blob.tags:
        if tag == ':':
            count += 1
    return count 

def tagEplision(text):
    blob = TextBlob(text)
    count=0
    for (word,tag) in blob.tags:
        if tag == '...':
            count += 1
    return count 

def tagSemiColon(text):
    blob = TextBlob(text)
    count=0
    for (word,tag) in blob.tags:
        if tag == ';':
            count += 1
    return count 


def tagQuestionMark(text):
    blob = TextBlob(text)
    count=0
    for (word,tag) in blob.tags:
        if tag == '?':
            count += 1
    return count 

def tagExclamation(text):
    blob = TextBlob(text)
    count=0
    for (word,tag) in blob.tags:
        if tag == '!':
            count += 1
    return count 


def tagPeroid(text):
    blob = TextBlob(text)
    count=0
    for (word,tag) in blob.tags:
        if tag == '.':
            count += 1
    return count 

def tagPeroid(text):
    blob = TextBlob(text)
    count=0
    for (word,tag) in blob.tags:
        if tag == '$':
            count += 1
    return count 
