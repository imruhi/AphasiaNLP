import pandas as pd
import re
import warnings
import contractions
from get_repetition import get_single_repetitions
warnings.simplefilter(action='ignore', category=FutureWarning)

special_characters = ['(.)', '[/]', '[//]', '‡', 'xxx', '+< ', '„', '+', '"" /..""', '+"/.', '+"', '+/?', '+//.',
                      '+//?', '[]', '<>', '_', '-', '^', ':', 'www .', '*PAR', '+/', '@o', '<', '>',
                      '//..', '//', '/..', '/', '"', 'ʌ', '..?', '0.', '0 .', '"" /.', ')', '(', "@u", "@si", "@k",
                      "@n", "$n", "$co", "$adj", "$on", "$v", "@l", 'æ', 'é', 'ð', 'ü', 'ŋ', 'ɑ', 'ɒ', 'ɔ', 'ə',
                      'ɚ', 'ɛ', 'ɜ', 'ɝ', 'ɡ', 'ɪ', 'ɹ', 'ɾ', 'ʃ', 'ʊ', 'ʒ', 'ʔ', 'ʤ', 'ʧ', 'ː', '˞', '͡', 'θ', "@q",
                      "@sspa", "@i", "@wp", "@sjpn", "@sdeu", "@p", "@sfra", "&", "⌉", "⌈", "{l=@", "}l=@",
                      "{l=%", "}l=%"]
ipa = ['æ', 'é', 'ð', 'ü', 'ŋ', 'ɑ', 'ɒ', 'ɔ', 'ə', 'ɚ', 'ɛ', 'ɜ', 'ɝ', 'ɡ', 'ɪ', 'ɹ', 'ɾ', 'ʃ', 'ʊ', 'ʒ', 'ʔ', 'ʤ',
       'ʧ', 'ː', '˞', '͡', 'θ', '{n', '}n', '+/', "⌋", "⌊", "⌉", "⌈"]

repetition_df = pd.DataFrame({"word": [], "count": [], "file": []})

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
    df = df.drop(columns=["speaker"])
    # df = df.drop(columns=['line_information', 'speaker_status', 'line_number', 'utterance_count'])
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
            df2 = pd.concat([df2, tobe_merged_data], ignore_index=True)
            # print(df2["repetition"])
        else:
            original_text_merged = ''.join(tobe_merged_data['text'].to_list())
            processed_text_merged = ''.join(tobe_merged_data['preprocessed_text'].to_list())
            if processed_text_merged[:-1] == 0:
                new_row = {'scenario': tobe_merged_data['scenario'].values[0],
                           'text': original_text_merged,
                           'source_file': tobe_merged_data['source_file'].values[0],
                           'preprocessed_text': processed_text_merged,
                           # 'repetition': tobe_merged_data["repetition"],
                           'line_merge_number': None
                           }
                df2 = df2.append(new_row, ignore_index=True)
                # print(tobe_merged_data["repetition"])

    df2 = df2.drop(columns=['text', 'line_merge_number'])

    return df2


def preprocess_line(utterance, mask_pauses, remove_repetitions, remove_masks, get_repetition, source_file):
    """
    Cleans text using regular expressions and custom functions.
    Also creates masks for filled and unfilled pauses.
    :param mask_pauses: Remove unfilled pauses with <mask> (if set true)
    :param remove_repetitions: Removes all masks (set true for healthy speech)

    :return:
    """

    expanded_words = []
    # print("1: ", utterance)

    for word in utterance.split():
        # using contractions.fix --> he's --> he is
        x = contractions.fix(word)
        # EDITED: make sure @you is not included since @u means babbling
        x = x.replace("@you", "@u")
        x = x.replace("YOU:m", "um")
        x = x.replace("you:m", "um")
        x = x.replace("youm", "um")
        x = x.replace("YOUm", "um")
        expanded_words.append(x)

    utterance = ' '.join(expanded_words)

    utterance = re.sub(r"{l=[[A-z]+\b|}l=[A-z]+\b", "", utterance)
    # Remove IPA
    for special_character in ipa:
        utterance = utterance.replace(special_character, "")

    # pattern1 = r"@u *\[: (.*?)]"
    # pattern2 = r"\b\w*@u\w*\b"
    # news = re.findall(pattern1, utterance)
    # olds = re.findall(pattern2, utterance)
    # if len(news) == len(olds):
    #     for new, old in zip(news, olds):
    #         if "x@n" in new:
    #             utterance = utterance.replace(old, "")
    #         else:
    #             utterance = utterance.replace(old, new)

    # +... Trailing off pause (speaker forgets about what is about to say)
    unfilled_pauses = ["(..)", "(...)", "+...", "(.)"]

    for pause in unfilled_pauses:
        utterance = utterance.replace(pause, "UNFILLEDPAUSE")
    # print("2: ", utterance)
    # Replace filler pauses with FILLERPAUSE
    filler_pauses = ["&-um", "&-uh", "&-er", "&-mm" "&-eh", "&-like", "&-youknow", "&-hm", "&-sighs",
                     "Mhm ", "mhm ", "um ", "Hm-m ", "hm-m ", "hm ", "Um ", "uh ", "Uh ", "Mm ", "mm ", "unhunh ",
                     "Unhunh ", "Hunh-unh ", "hunh-unh ", "Unh-unh ", "unh-unh "]

    for pause in filler_pauses:
        utterance = utterance.replace(pause, "FILLERPAUSE")
    # print("3: ", utterance)

    # Remove all actions: (e.g. &=points:picture)
    utterance = re.sub(r"\&\=[a-zA-Z:_0-9]+", "", utterance)

    # remove before- (eg c- )
    # utterance = re.sub(r"[A-Za-z]- ", " ", utterance)

    # Remove all unicode errors   \W\d+\w\d+\W
    utterance = re.sub(r"\W\d+\w\d+\W", "", utterance)

    # Remove anything between [ and ]
    utterance = re.sub(r"\[.*?\]", "", utterance)

    # Remove  < and > for repetition [/] and retracing [//]
    utterance = re.sub(r"\<|\>", "", utterance)

    # Remove anything &+ or &* (eg &*INV)
    utterance = re.sub(r"\&\+[a-zA-Z]+", "", utterance)
    utterance = re.sub(r"\&\*INV:[a-zA-Z_]+", "", utterance)
    utterance = re.sub(r"\&\=[a-zA-Z_]+", "", utterance)

    # print("4: ", utterance)

    # Remove remaining special chars
    for special_character in special_characters:
        # EDITED: for words such as you_know and of_course
        if special_character == '_' or special_character == "-":
            utterance = utterance.replace(special_character, " ")
        utterance = utterance.replace(special_character, "")

    utterance = re.sub(' +', ' ', utterance)
    # print("5: ", utterance)

    # Remove special characters from starting sentence
    remove_startswiths = [" ", ",", "!", ".", "?", "."]
    for start_string in remove_startswiths:
        if utterance.startswith(start_string):
            utterance = utterance[1:]
    # print("6: ", utterance)

    # Replace all pauses with <mask> (if set true)
    if mask_pauses:
        utterance = utterance.replace("UNFILLEDPAUSE", "<mask>")
        utterance = utterance.replace("FILLERPAUSE", "<mask>")

    # Removes all masks (set true for healthy speech)
    if remove_masks:
        utterance = utterance.replace("<mask>", "")
    # print("7: ", utterance)

    # remove digits
    utterance = re.sub(r"[0-9]+", "", utterance)
    # print("8: ", utterance)

    # Remove final whitespaces (e.g. double space)
    utterance = re.sub(' +', ' ', utterance)
    # Remove trailing whitespaces before punctuation
    utterance = re.sub(r'\s+([?.!"])', r'\1', utterance)
    # print("9: ", utterance)

    # remove extra spaces
    utterance = utterance.lstrip().rstrip().lower().replace("[]", "").replace("]", "")
    utterance = re.sub(' +|\t', ' ', utterance)
    utterance = re.sub(' ,', ',', utterance)
    # print("10: ", utterance)

    # ADDED
    repetition = None
    if get_repetition:
        repetition = get_single_repetitions(utterance)
        if repetition is not None:
            repetition = pd.DataFrame.from_dict(repetition, orient='index').reset_index()
            repetition = repetition.rename(columns={'index': 'word', 0: 'count'})
            repetition["file"] = source_file

            global repetition_df
            repetition_df = pd.concat([repetition_df, repetition])

    return utterance, repetition


def preprocess_dataset(input_dataset_filename, mask_pauses, remove_repetitions, remove_masks, get_repetition):
    df = pd.read_csv(input_dataset_filename, encoding='utf8')
    df = df.dropna()
    # repetition = pd.DataFrame()
    # speaker_status = []  # Add speaker status / 1 for interviewer, 2 for participant
    # current_status = 0
    # for line_information in df['line_information']:
    #     if line_information == "*INV" or line_information == "*IN1" or line_information == "*IN2":
    #         current_status = 1
    #     elif line_information == "*PAR":
    #         current_status = 2
    #
    #     speaker_status.append(current_status)
    # df["speaker_status"] = speaker_status
    # df = df.loc[(df['speaker_status'] == 2)]
    #
    # # Not selected scenarios: "Cinderella_Intro", "Sandwich_Intro", "Important_Event_Intro", "Repetition", "BNT",
    # # "Sandwich_Other","Sandwich_Picture", "VNT"
    # selected_scenarios = ["Important_Event", "Speech", "Stroke", "Cinderella", "Sandwich", "Window", "Cat", "Umbrella",
    #                       "Flood", "Cinderella_Intro", "Sandwich_Intro", "Important_Event_Intro", "Repetition", "BNT",
    #                       "Sandwich_Other", "Sandwich_Picture", "VNT"]
    #
    # df = df[df.scenario.isin(selected_scenarios)]
    # df = df.loc[(df['line_information'] == "*PAR")]  # get only participants

    preprocessed_text = []

    for index, row in df.iterrows():
        preprocessed_line, rep = preprocess_line(row["text"], mask_pauses, remove_repetitions, remove_masks,
                                                 get_repetition, row["source_file"])
        # print(row["text"])
        # print(preprocessed_line)
        # repetition = pd.concat([repetition, rep])
        # repetition = repetition.reset_index(drop=True)
        preprocessed_text.append(preprocessed_line)
    # print(repetition)
    df['original_text'] = df['text']
    df['preprocessed_text'] = preprocessed_text

    # df['repetition'] = repetition
    return df


if __name__ == "__main__":
    # preprocessed_df = preprocess_dataset("data/data_broca.csv", True, False,
    #                                      True, True)
    # df = make_sentences_df(preprocessed_df)
    # df.to_csv("data/preprocessed_broca.csv")
    # print(repetition_df)
    # repetition_df.to_csv("data/repetition_broca.csv")

    preprocessed_df = preprocess_dataset("../linguistic_model/data/spoken corpus/data_test.csv", True, False,
                                         True, False)
    # df = make_sentences_df(preprocessed_df)
    preprocessed_df.to_csv("../linguistic_model/data/spoken corpus/preprocessed_test.csv")

    # preprocessed_df = preprocess_dataset("data/data_control.csv", True, False, True)
    # df = make_sentences_df(preprocessed_df)
    # df.to_csv("data/preprocessed_control.csv")

    # preprocessed_df = preprocess_dataset("data/data_control.csv", False, False, False)
    # df = make_sentences_df(preprocessed_df)
    # df.to_csv("data/preprocessed_control_pause.csv")
    # print("Control done")

    # preprocessed_df = preprocess_dataset("data/data_anomic.csv", False, False, False)
    # df = make_sentences_df(preprocessed_df)
    # df.to_csv("data/preprocessed_anomic_pause.csv")
    # print("Anomic done")
    #
    # preprocessed_df = preprocess_dataset("data/data_conduction.csv", False, False, False)
    # df = make_sentences_df(preprocessed_df)
    # df.to_csv("data/preprocessed_conduction_pause.csv")
    # print("Conduction done")
    #
    # preprocessed_df = preprocess_dataset("data/data_transsensory.csv", False, False, False)
    # df = make_sentences_df(preprocessed_df)
    # df.to_csv("data/preprocessed_transsensory_pause.csv")
    # print("Transsensory done")
    #
    # preprocessed_df = preprocess_dataset("data/data_wernicke.csv", False, False, False)
    # df = make_sentences_df(preprocessed_df)
    # df.to_csv("data/preprocessed_wernicke_pause.csv")
    # print("Wernicke done")
