import enchant
dict_en = enchant.Dict('en_US')



def create_word_list(tokens):
    # Create List to house tokens that are in the dictionary
    word_list   = []
    for token in tokens:
        if len(token) > 0 and dict_en.check(token) is True:
            # Append words to word_list
            word_list.append(token)

    # Create text as continous string
    text        = ' '.join(word_list)

    # Return text
    return text




def get_continuous_list_tokens(list_of_texts):

    list_tokens = []

    for text in list_of_texts:
        text_tokenized = text.split(' ')

        for token in text_tokenized:
            list_tokens.append(token)

    return list_tokens

def get_max_len_of_list_txts(list_txt_tokenized):
    '''
    Input:      List of tokenized text
    Output:     Int representing num longest text
    Process:    Replace txt with len of text
    '''
    num_tokens_list = []

    for txt in list_txt_tokenized:
        num_tokens_list.append(len(txt))

    return max(num_tokens_list), min(num_tokens_list), sum(num_tokens_list)/len(num_tokens_list)


def cap_len_txt(list_txt_tokenized, max_len):
    '''
    Input:      List of tokenized text, max_len as int
    Output:     Same list w/ max num of tokens per txt
    Process :   Cut each list of txt and append to new list
    '''
    capped_list_txt = []

    for txt in list_txt_tokenized:
        capped_list_txt.append(txt[: max_len])

    return capped_list_txt



def get_txt_vocab_dict(txt_tokenized):
    '''
    Input:      List of tokenized text
    Output:     Dictionary of unique tokens
    '''
    vocab   = {}
   
    # Create Set of Tokens From List of All Tokens
    set_words = list(set(txt_tokenized)) 
    words_enumerated = enumerate(set_words)
    

    # Build Vocabulary Dictionary w/ value = num
    for word in words_enumerated:
        vocab[word[1]] = word[0]

    return vocab

def enumerate_tokens(txt_tokenized, translation_dictionary):
    
    new_list_txt_tokenized = []

    # Iterate List of tokens
    for token in txt_tokenized:
        # Look up in dictionary and append int to temp_list
        new_list_txt_tokenized.append(translation_dictionary[token])

    # Return New List w/ Lists of Text w/ Tokens Enumerated
    return new_list_txt_tokenized




        

















