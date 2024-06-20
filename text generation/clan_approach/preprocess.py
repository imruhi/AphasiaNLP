import re
import string
import contractions

special_characters = ['(.)', '[/]', '[//]', '‡', 'xxx', '+< ', '„', '+', '"" /..""', '+"/.', '+"', '+/?', '+//.',
                      '+//?', '[]', '<>', '_', '-', '^', ':', 'www .', '*PAR', '+/', '@o', '<', '>',
                      '//..', '//', '/..', '/', '"', 'ʌ', '..?', '0.', '0 .', '"" /.', ')', '(', "@u", "@si", "@k",
                      "@n", "$n", "$co", "$adj", "$on", "$v", "@l", 'æ', 'é', 'ð', 'ü', 'ŋ', 'ɑ', 'ɒ', 'ɔ', 'ə',
                      'ɚ', 'ɛ', 'ɜ', 'ɝ', 'ɡ', 'ɪ', 'ɹ', 'ɾ', 'ʃ', 'ʊ', 'ʒ', 'ʔ', 'ʤ', 'ʧ', 'ː', '˞', '͡', 'θ', "@q",
                      "@sspa", "@i", "@wp", "@sjpn", "@sdeu", "@p", "@sfra", "&", "*"]
ipa = ['æ', 'é', 'ð', 'ü', 'ŋ', 'ɑ', 'ɒ', 'ɔ', 'ə', 'ɚ', 'ɛ', 'ɜ', 'ɝ', 'ɡ', 'ɪ', 'ɹ', 'ɾ', 'ʃ', 'ʊ', 'ʒ', 'ʔ', 'ʤ',
       'ʧ', 'ː', '˞', '͡', 'θ', ]


def preprocess(utterance):

    expanded_words = []
    for word in utterance.split():
        # using contractions.fix --> he's --> he is
        x = contractions.fix(word)
        # EDITED: make sure @you is not included since @u means babbling
        expanded_words.append(x.replace("@you", "@u"))
    utterance = ' '.join(expanded_words)

    # Remove IPA
    for special_character in ipa:
        utterance = utterance.replace(special_character, "")

    pattern1 = r"@u *\[: (.*?)]"
    pattern2 = r"\b\w*@u\w*\b"
    news = re.findall(pattern1, utterance)
    olds = re.findall(pattern2, utterance)
    if len(news) == len(olds):
        for new, old in zip(news, olds):
            if "x@n" in new:
                utterance = utterance.replace(old, "")
            else:
                utterance = utterance.replace(old, new)

    # Remove all actions: (e.g. &=points:picture)
    utterance = re.sub(r"\&\=[a-zA-Z:_0-9]+", "", utterance)

    # Remove all unicode errors   \W\d+\w\d+\W
    # utterance = re.sub(r"\W\d+\w\d+\W", "", utterance)

    # Remove anything between [ and ]
    utterance = re.sub(r"\[.*?\]", "", utterance)

    # Remove  < and > for repetition [/] and retracing [//]
    utterance = re.sub(r"\<|\>", "", utterance)

    # Remove anything &+ or &* (eg &*INV)
    utterance = re.sub(r"\&\+[a-zA-Z]+", "", utterance)
    utterance = re.sub(r"\&\*INV:[a-zA-Z_]+", "", utterance)

    # Remove remaining special chars
    for special_character in special_characters:
        # EDITED: for words such as you_know and of_course
        if special_character in ['_', '-']:
            # print("yes")
            utterance = utterance.replace(special_character, " ")
        else:
            utterance = utterance.replace(special_character, "")

    utterance = re.sub(' +', ' ', utterance)

    # Remove special characters from starting sentence
    remove_startswiths = [" ", ",", "!", ".", "?", ".", ";"]
    for start_string in remove_startswiths:
        if utterance.startswith(start_string):
            utterance = utterance[1:]

    # Remove final whitespaces (e.g. double space)
    utterance = re.sub(' +', ' ', utterance)
    # Remove trailing whitespaces before punctuation
    utterance = re.sub(r'\s+([?.!"])', r'\1', utterance)

    # remove extra spaces
    utterance = utterance.lstrip().rstrip().lower().replace("[]", "").replace("]", "")
    utterance = re.sub(' +|\t', ' ', utterance)
    utterance = re.sub(' ,', ',', utterance)

    return utterance


def postprocess(sent):
    if isinstance(sent, str):
        if len(sent.rstrip()) >= 1:
            # remove trailing 's
            sent = re.sub(r"^'s | 's", "", sent)
            # remove trailing .
            sent = re.sub(r"\.{2,}", "", sent)
            # remove trailing ,
            sent = re.sub(r"^,+ |,{2,}", "", sent)
            # remove trailing '
            sent = re.sub(r"^'+ +| +'+", "", sent)
            # remove ,.
            sent = re.sub(r",\.", ".", sent)
            # remove ;
            sent = re.sub(r";", "", sent)
            # remove trailing ! and ?
            sent = re.sub(r"(\!{2,})", "!", sent)
            sent = re.sub(r"^\!+ +", "", sent)
            sent = re.sub(r"^\?+ +", "", sent)
            sent = re.sub(r"(\?{2,})", "?", sent)
            # remove space before all punctuation
            sent = re.sub(r'\s([?.!"](?:\s|$))', r'\1', sent)

            # remove all punctuation
            punc = '''()-[]{};:'"\,<>@#$%^&*_~'''
            for a in sent:
                if a in punc:
                    sent = sent.replace(a, "")

            sent = sent.lower()
            if len(sent.split()) >= 1 and not all(x in string.punctuation for x in sent.split()):
                return " ".join(sent.split())

    return ""
