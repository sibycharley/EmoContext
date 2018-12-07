import sys
sys.path.append("C:\\Anaconda3\\envs\\tensorflow\\python35.zip")
sys.path.append("C:\\Anaconda3\\envs\\tensorflow\\DLLs")
sys.path.append("C:\\Anaconda3\\envs\\tensorflow\\lib") 
sys.path.append("C:\\Anaconda3\\envs\\tensorflow") 
sys.path.append("C:\\Anaconda3\\envs\\tensorflow\\lib\\site-packages")
sys.path.append("C:\\Anaconda3\\envs\\tensorflow\\lib\\site-packages\\setuptools-27.2.0-py3.5.egg")
print(sys.path)

import numpy as np
import pandas as pd
import emoji
import re

#Read training data
data = pd.read_table('../Data/train.txt')

#Regex
special_regex = re.compile('[-"@_+=#$%^&*()<>/\|}{~:;,]+')
question_regex = re.compile('[?+]')
ex_regex = re.compile('[!]+')
dot_regex = re.compile('[.]+')
num_regex = re.compile('[^[0-9]+$]')
apho_rege = re.compile('[\']')
alphanum_regex = [r'\w+']

#Unknown chars
unk_pattern = re.compile("["
        "\\U0001F9D0"
        "\\U000FE339"
        "\\U000023EA"
        "\\U00000E3F"
        "\\U000020B9"
        "\\U00002211"
        "\\U00002267"
        "\\U00002207"
        "\\U00002248"
        "\\U00002284"
        "\\U000000BF"
        "\\U000000AC"
        "\\U000000B0 - \\U000000B6"
        "\\U00002022"
        "\\U00000296"
        "\\U000003C9"
        "\\U000000D7"
        "]+", flags=re.UNICODE)

#Replace Special chars
for i, row in data.iterrows():
    data.set_value(i, 'turn1', special_regex.sub(r' SPLCHAR ', row['turn1']))
    data.set_value(i, 'turn2', special_regex.sub(r' SPLCHAR ', row['turn2']))
    data.set_value(i, 'turn3', special_regex.sub(r' SPLCHAR ', row['turn3']))

#Replace Question chars
for i, row in data.iterrows():
    data.set_value(i, 'turn1', question_regex.sub(r' QCHAR ', row['turn1']))
    data.set_value(i, 'turn2', question_regex.sub(r' QCHAR ', row['turn2']))
    data.set_value(i, 'turn3', question_regex.sub(r' QCHAR ', row['turn3']))

#Replace Exlamatory chars
for i, row in data.iterrows():
    data.set_value(i, 'turn1', ex_regex.sub(r' EXCHAR ', row['turn1']))
    data.set_value(i, 'turn2', ex_regex.sub(r' EXCHAR ', row['turn2']))
    data.set_value(i, 'turn3', ex_regex.sub(r' EXCHAR ', row['turn3']))

#Replace Dots
for i, row in data.iterrows():
    data.set_value(i, 'turn1', dot_regex.sub(r' DCHAR ', row['turn1']))
    data.set_value(i, 'turn2', dot_regex.sub(r' DCHAR ', row['turn2']))
    data.set_value(i, 'turn3', dot_regex.sub(r' DCHAR ', row['turn3']))

#Emoji Classification
happy = pd.read_csv('../Data/Emoji/happy.csv')
sad = pd.read_csv('../Data/Emoji/sad.csv')
angry = pd.read_csv('../Data/Emoji/angry.csv')
other = pd.read_csv('../Data/Emoji/other.csv')

#Alpha-numeric methods
def hasdigit(word):
    return any(c for c in word if c.isdigit())

def hasalpha(word):
    return any(c for c in word if c.isalpha())

def hasalnum(word):
    return hasdigit(word) and hasalpha(word)

happylst = []
sadlst = []
angrylst = []
otherlst = []

for i, row in happy.iterrows():
    happylst.append(row[0])

for i, row in sad.iterrows():
    sadlst.append(row[0])

for i, row in angry.iterrows():
    angrylst.append(row[0])

for i, row in other.iterrows():
    otherlst.append(row[0])

#Apply emoji classification on Turn 1 data
for i, row in data.iterrows():
    tempStr = []
    for s in row.turn1.split():
        for word in s.split():
            tempWord = []
            for char in word:
                if char in emoji.UNICODE_EMOJI:
                    char = 'U+{:X}'.format(ord(char))
                    if char in happylst:
                        char = 'HAPPY'
                    elif char in sadlst:
                        char = 'SAD'
                    elif char in angrylst:
                        char = 'ANGRY'
                    elif char in otherlst:
                        char = 'OTHER'
                    tempWord.append(' ')
                    tempWord.append(char)
                    tempWord.append(' ')
                else:
                    tempWord.append(char)
        strWord = ''.join(i for i in tempWord)
        tempStr.append(strWord)
    strFinal = ' '.join(w for w in tempStr)
    data.set_value(i, 'turn1', strFinal)

#Apply emoji classification on Turn 2 data
for i, row in data.iterrows():
    tempStr = []
    for s in row.turn2.split():
        for word in s.split():
            tempWord = []
            for char in word:
                if char in emoji.UNICODE_EMOJI:
                    char = 'U+{:X}'.format(ord(char))
                    if char in happylst:
                        char = 'HAPPY'
                    elif char in sadlst:
                        char = 'SAD'
                    elif char in angrylst:
                        char = 'ANGRY'
                    elif char in otherlst:
                        char = 'OTHER'
                    tempWord.append(' ')
                    tempWord.append(char)
                    tempWord.append(' ')
                else:
                    tempWord.append(char)
        strWord = ''.join(i for i in tempWord)
        tempStr.append(strWord)
    strFinal = ' '.join(w for w in tempStr)
    data.set_value(i, 'turn2', strFinal)

#Apply emoji classification on Turn 3 data
for i, row in data.iterrows():
    tempStr = []
    for s in row.turn3.split():
        for word in s.split():
            tempWord = []
            for char in word:
                if char in emoji.UNICODE_EMOJI:
                    char = 'U+{:X}'.format(ord(char))
                    if char in happylst:
                        char = 'HAPPY'
                    elif char in sadlst:
                        char = 'SAD'
                    elif char in angrylst:
                        char = 'ANGRY'
                    elif char in otherlst:
                        char = 'OTHER'
                    tempWord.append(' ')
                    tempWord.append(char)
                    tempWord.append(' ')
                else:
                    tempWord.append(char)
        strWord = ''.join(i for i in tempWord)
        tempStr.append(strWord)
    strFinal = ' '.join(w for w in tempStr)
    data.set_value(i, 'turn3', strFinal)


#Replace Numbers from text
for i, row in data.iterrows():
    temp1 = []
    temp2 = []
    temp3 = []
    for s in row.turn1.split():
        if hasalnum(s) or hasalpha(s):
            t = s
        else:
            t = re.sub('\d+', 'NUM', s)
        temp1.append(str(t))
    strturn1 = ' '.join(word for word in temp1)
    data.set_value(i, 'turn1', strturn1)
    
    for s in row.turn2.split():
        if hasalnum(s) or hasalpha(s):
            t = s
        else:
            t = re.sub('\d+', 'NUM', s)
        temp2.append(str(t))
    strturn2 = ' '.join(word for word in temp2)
    data.set_value(i, 'turn2', strturn2)
    
    for s in row.turn3.split():
        if hasalnum(s) or hasalpha(s):
            t = s
        else:
            t = re.sub('\d+', 'NUM', s)
        temp3.append(str(t))
    strturn3 = ' '.join(word for word in temp3)
    data.set_value(i, 'turn3', strturn3)

#Some more data cleaning
for i, row in data.iterrows():
    temp1 = []
    temp2 = []
    temp3 = []
    for s in row.turn1.split():
        t = re.sub('[\\\\]', '', s)
        u = re.sub('[\[\]]', '', t)
        v = re.sub('[.]', '', u)
        x = re.sub('[-]', '', v)
        temp1.append(str(x))
    strturn1 = ' '.join(word for word in temp1)
    data.set_value(i, 'turn1', strturn1)
    
    for s in row.turn2.split():
        t = re.sub('[\\\\]', '', s)
        u = re.sub('[\[\]]', '', t)
        v = re.sub('[.]', '', u)
        x = re.sub('[-]', '', v)
        temp2.append(str(x))
    strturn2 = ' '.join(word for word in temp2)
    data.set_value(i, 'turn2', strturn2)
    
    for s in row.turn3.split():
        t = re.sub('[\\\\]', '', s)
        u = re.sub('[\[\]]', '', t)
        v = re.sub('[.]', '', u)
        x = re.sub('[-]', '', v)
        temp3.append(str(x))
    strturn3 = ' '.join(word for word in temp3)
    data.set_value(i, 'turn3', strturn3)


#remove unknowns
for i, row in data.iterrows():
    temp1 = []
    temp2 = []
    temp3 = []
    for s in row.turn1.split():
        x = unk_pattern.sub(r'', s)
        temp1.append(str(x))
    strturn1 = ' '.join(word for word in temp1)
    data.set_value(i, 'turn1', strturn1)
    
    for s in row.turn2.split():
        x = unk_pattern.sub(r'', s)
        temp2.append(str(x))
    strturn2 = ' '.join(word for word in temp2)
    data.set_value(i, 'turn2', strturn2)
    
    for s in row.turn3.split():
        x = unk_pattern.sub(r'', s)
        temp3.append(str(x))
    strturn3 = ' '.join(word for word in temp3)
    data.set_value(i, 'turn3', strturn3)


#Remove apostrophe
for i, row in data.iterrows():
    temp1 = []
    temp2 = []
    temp3 = []
    for s in row.turn1.split():
        x = apho_rege.sub(r'', s)
        temp1.append(str(x))
    strturn1 = ' '.join(word for word in temp1)
    data.set_value(i, 'turn1', strturn1)
    
    for s in row.turn2.split():
        x = apho_rege.sub(r'', s)
        temp2.append(str(x))
    strturn2 = ' '.join(word for word in temp2)
    data.set_value(i, 'turn2', strturn2)
    
    for s in row.turn3.split():
        x = apho_rege.sub(r'', s)
        temp3.append(str(x))
    strturn3 = ' '.join(word for word in temp3)
    data.set_value(i, 'turn3', strturn3)

#Save to CSV/Text file
turn1_data =  data['turn1']
turn2_data =  data['turn2']
turn3_data =  data['turn3']

turn1_data.to_csv('./Clean/T1_v3.csv', sep=',', encoding='utf-8')
turn2_data.to_csv('./Clean/T2_v3.csv', sep=',', encoding='utf-8')
turn3_data.to_csv('./Clean/T3_v3.csv', sep=',', encoding='utf-8')