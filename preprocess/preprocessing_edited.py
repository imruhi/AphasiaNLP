import pandas as pd
import re
import contractions
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


# TODO: (th) them -> them or -> th them
special_characters = ['(.)', '[/]', '[//]', '‡', 'xxx', '+< ', '„', '+', '"" /..""', '+"/.', '+"', '+/?', '+//.',
                        '+//?', '[]', '<>', '_', '-', '^', ':', 'www .', '*PAR', '+/', '@o', '<', '>',
                        '//..', '//', '/..', '/', '"', 'ʌ', '..?', '0.', '0 .', '"" /.', ')', '(', "@u", "@si", "@k",
                        "@n", "$n", "$co", "$adj", "$on", "$v", "@l", 'æ', 'é', 'ð', 'ü', 'ŋ', 'ɑ', 'ɒ', 'ɔ', 'ə',
                        'ɚ', 'ɛ', 'ɜ', 'ɝ', 'ɡ', 'ɪ', 'ɹ', 'ɾ', 'ʃ', 'ʊ', 'ʒ', 'ʔ', 'ʤ', 'ʧ', 'ː', '˞', '͡', 'θ', "@q",
                        "@sspa", "@i", "@wp", "@sjpn", "@sdeu", "@p", "@sfra"]

def contains_whitespace(string):
    """
    Check if string contains whitespace.
    :param string: input string.
    :return: Returns true if string contains a space.
    """
    for letter in string:
        if letter == " ":
            return True
    return False


def return_ending_punctuation(input_sentence):
    """
    returns special char of ending sentence ?, !, or . given a sentence.
    :param input_sentence: string
    :return: string containing special char.
    """
    matches = ["?", "!", "."]

    for symbol in input_sentence[-3:]:
        if symbol in matches:
            return str(symbol)


def make_sentences_df(input_dataset):
    """
    Converts incoherent lines into sentences (based on sentence ending e.g: ?.!)
    (incoherent lines are lines that should belong together, but have been split previously)
    :return: dataframe with sentences.
    """
    df = input_dataset
    df = df.dropna(subset=['preprocessed_text'])
    df = df.drop(columns=['line_information', 'speaker_status', 'line_number', 'utterance_count'])
    line_merge_number = []
    i = 0
    for sentence in df['preprocessed_text']:
        matches = ["?", "!", "."]
        if any([x in sentence for x in matches]):
            line_merge_number.append(i)
            i += 1
        else:
            line_merge_number.append(i)

    df['line_merge_number'] = line_merge_number

    df2 = pd.DataFrame(columns=df.columns)
    for number in df['line_merge_number'].unique():
        tobe_merged_data = df.loc[(df['line_merge_number'] == number)]
        if len(tobe_merged_data) <= 1:
            df2 = pd.concat([df2,tobe_merged_data], ignore_index=True)
        else:
            original_text_merged = ''.join(tobe_merged_data['text'].to_list())
            processed_text_merged = ''.join(tobe_merged_data['preprocessed_text'].to_list())
            if processed_text_merged[:-1] == 0:
                new_row = {'scenario': tobe_merged_data['scenario'].values[0],
                           'text': original_text_merged,
                           'source_file': tobe_merged_data['source_file'].values[0],
                           'preprocessed_text': processed_text_merged,
                           'line_merge_number': None
                           }
                df2 = df2.append(new_row, ignore_index=True)

    df2 = df2.drop(columns=['text', 'line_merge_number'])
    return df2


def preprocess_line(utterance, mask_pauses, remove_repetitions, remove_masks):
    """
    Cleans text using regular expressions and custom functions.
    Also creates masks for filled and unfilled pauses.
    :param mask_pauses: Remove unfilled pauses with <mask> (if set true)
    :param remove_repetitions: Removes all masks (set true for healthy speech)

    :return:
    """
    expanded_words = []
    # for word in utterance.split():
        # using contractions.fix --> he's --> he is
        # x = contractions.fix(word)
        # EDITED: make sure @you is not included since @u means babbling
        # expanded_words.append(x.replace("@you", "@u"))
    # utterance = ' '.join(expanded_words)

    # ADDED: if there are any utterances (@u) replace them with what is in the [: ] after the @u
    # later on everything between [...] is deleted
    pattern1 = r"@u \[: (.*?)]"
    pattern2 = r"\b\w*@u\w*\b"

    news = re.findall(pattern1, utterance)
    olds = re.findall(pattern2, utterance)

    if len(news) == len(olds):
        for new, old in zip(news, olds):
            if "x@n" in new:
                utterance = utterance.replace(old, "")
            else:
                utterance = utterance.replace(old, new)


    # +... Trailing off pause (speaker forgets about what is about to say)
    unfilled_pauses = ["(..)", "(...)", "+..."]
    for pause in unfilled_pauses:
        utterance = utterance.replace(pause, "UNFILLEDPAUSE")

    # Replace filler pauses with FILLERPAUSE
    filler_pauses = ["&-um", "&-uh", "&-er", "&-mm" "&-eh", "&-like", "&-youknow", "&-hm", "&-sighs"]
    for pause in filler_pauses:
        utterance = utterance.replace(pause, "FILLERPAUSE")

    # Remove all actions: (e.g. &=points:picture)
    r = re.findall(r"\W\W\w+\W\w+", utterance)
    for regex in r:
        if not contains_whitespace(regex):
            utterance = utterance.replace(regex, "")

    # Remove all unicode errors   \W\d+\w\d+\W
    r = re.findall(r"\W\d+\w\d+\W", utterance)
    for regex in r:
        if not contains_whitespace(regex):
            utterance = utterance.replace(regex, "")

    # Remove all characters that start with 2 special chars and have text succeeding it:
    r = re.findall(r"\W\W\w+", utterance)
    for regex in r:
        # if regex[0] != " " and regex[1] != " ":
        if not contains_whitespace(regex):
            # This prevents second and first letter being a space to be removed.
            # e.g. : ") Cinderella" removes the word Cinderella too, which we dont want.
            utterance = utterance.replace(regex, "")

    # Remove anything between [ and ]
    r = re.findall(r"\[(.*?)\]", utterance)
    for regex in r:
        utterance = utterance.replace(regex, "")

    # Remove anything between < and >
    r = re.findall(r"\<(.*?)\>", utterance)
    for regex in r:
        utterance = utterance.replace(regex, "")

    # Remove anything &+
    r = re.findall(r"\&+[a-zA-Z]+", utterance)
    for regex in r:
        utterance = utterance.replace(regex, "")

    # Remove remaining special chars
    for special_character in special_characters:

        # EDITED: for words such as you_know and of_course
        if special_character == '_':
            utterance = utterance.replace(special_character, " ")

        utterance = utterance.replace(special_character, "")

    utterance = re.sub(' +', ' ', utterance)

    # Remove special characters from starting sentence
    remove_startswiths = [" ", ",", "!", ".", "?", "."]
    for start_string in remove_startswiths:
        if utterance.startswith(start_string):
            utterance = utterance[1:]

    # Replace all pauses with <mask> (if set true)
    if mask_pauses:
        utterance = utterance.replace("UNFILLEDPAUSE", "<mask>")
        utterance = utterance.replace("FILLERPAUSE", "<mask>")

    # Removes all masks (set true for healthy speech)
    if remove_masks:
        utterance = utterance.replace("<mask>", "")

    # Removes stuttering and bigram stuttering
    # if remove_repetitions:
    #    utterance = remove_all_repetitions(utterance)

    utterance = re.sub(' +', ' ', utterance) # Remove final whitespaces (e.g. double space)
    # EDITED: remove trailing whitespaces before punctuation
    utterance = re.sub(r'\s+([?.!"])', r'\1', utterance)

    # Remove anything between [ and ]
    r = re.findall(r"\[(.*?)\]", utterance)
    for regex in r:
        utterance = utterance.replace(regex, "")

    # remove extra spaces
    utterance = utterance.lstrip().rstrip().lower().replace("[]", "").replace("]", "")
    utterance = re.sub(' +|\t', ' ', utterance)
    utterance = re.sub(' ,', ',', utterance)
    return utterance


def preprocess_dataset(input_dataset_filename, mask_pauses, remove_repetitions, remove_masks):
    df = pd.read_csv(input_dataset_filename, encoding='utf8')
    df = df.dropna()

    speaker_status = []  # Add speaker status / 1 for interviewer, 2 for participant
    current_status = 0
    for line_information in df['line_information']:
        if line_information == "*INV" or line_information == "*IN1" or line_information == "*IN2":
            current_status = 1
        elif line_information == "*PAR":
            current_status = 2

        speaker_status.append(current_status)
    df["speaker_status"] = speaker_status
    df = df.loc[(df['speaker_status'] == 2)]

    # Not selected scenarios: "Cinderella_Intro", "Sandwich_Intro", "Important_Event_Intro", "Repetition", "BNT",
    # "Sandwich_Other","Sandwich_Picture", "VNT"
    selected_scenarios = ["Important_Event", "Speech", "Stroke", "Cinderella", "Sandwich", "Window", "Cat", "Umbrella",
                          "Flood", "Cinderella_Intro", "Sandwich_Intro", "Important_Event_Intro", "Repetition", "BNT",
                          "Sandwich_Other", "Sandwich_Picture", "VNT"]

    df = df[df.scenario.isin(selected_scenarios)]
    df = df.loc[(df['line_information'] == "*PAR")] # get only participants

    preprocessed_text = []
    for text in df['text']:
        preprocessed_line = preprocess_line(text, mask_pauses, remove_repetitions, remove_masks)
        preprocessed_text.append(preprocessed_line)

    df['original_text'] = df['text']
    df['preprocessed_text'] = preprocessed_text
    return df


if __name__ == "__main__":

    preprocessed_df = preprocess_dataset("data_broca.csv", True, False, True)
    df = make_sentences_df(preprocessed_df)
    df.to_csv("test1.csv")

    preprocessed_df = preprocess_dataset("data_control.csv", True, False, True)
    df = make_sentences_df(preprocessed_df)
    df.to_csv("test2.csv")