import pandas as pd
import os
import re
import pathlib


def cha_txt_to_dataframe(file_name):
    """
    Returns a dataframe that converts single .cha (.txt) file to a dataframe.
    :param file_name: .txt file containing (cha formatted).
    :return: pandas dataframe
    """
    f = open(file_name, "r", encoding="utf-8")
    lines = f.readlines()

    scenarios = []
    current_scenario = "N/A"
    line_type = []

    with open(file_name, 'r', encoding="utf-8") as file:
        x = file.read().rstrip()
        line_type = list(set(re.findall(r'\*[A-Z]+', x)))

    # print(line_type)
    # line_type = ["*INV", "*PAR", "%wor", "%mor", "%gra", "%exp", "*IN1", "*IN2"]
    current_line_type = "N/A"

    line_number = []
    text = []
    speaker = []
    current_line_type = "N/A"
    i = 0
    for line in lines:
        # if line[:3] == "@G:":  # Adding scenario information
        #     line = re.sub("\n", "", line)
        #     line = re.sub("\t", " ", line)
        #     current_scenario = line[3:].strip()
        # scenarios.append(current_scenario)

        # if line[:4] in line_type:
        #     if line[:4] != current_line_type:
        #         current_line_type = line[:4]
        #         i += 1
        # line_number.append(i)
        # print(line)
        next_line_type = re.findall(r'\*[A-Z0-9]+', line)
        if next_line_type:
            current_line_type = next_line_type[0]
        if current_line_type != "N/A":
            line = re.sub("\n", "", line)
            line = re.sub("\t", " ", line)
            line = re.sub(r"\*[A-Z0-9]+: ","", line)
            text.append(line)
            speaker.append(current_line_type)

    # columns = ['line_number', 'scenario', 'text', 'line_information', 'utterance_count']
    columns = ['text', 'speaker']
    df = pd.DataFrame(columns=columns)
    df['text'] = text
    df['speaker'] = speaker
    # df['line_number'], df['scenario'], df['text'] = line_number, scenarios, text
    # df = df.groupby(['line_number', 'scenario'])['text'].apply(' '.join).reset_index()
    # df['line_information'] = df['text'].astype(str).str[:4]
    # df['text'] = df['text'].str[6:]
    # df = df.loc[df['line_information'] != "@Beg"]
    # df = df.loc[df['line_information'] != "@G: "]
    # df = df.loc[df['line_information'] != "@UTF"]

    # utterance_number = []
    # utterance_count = 0

    # for info in df['line_information']:
    #     participants = ["*INV", "*PAR", "*IN1", "*IN2"]
    #     if info in participants:
    #         utterance_count += 1
    #
    #     utterance_number.append(utterance_count)
    #
    # df['utterance_count'] = utterance_number
    print(df)
    return df


def cha_txt_files_to_csv(data_dir, file_name, fnames=None):
    """
    Uses a directory containing .cha based .txt files to convert to .csv file.
    :param data_dir: Directory in which .txt files are located.
    :param file_name: .CSV file save data in.
    :param fnames: EDITED file names which we are interested in
    :return: Returns true if completed.
    """
    # columns = ['line_number', 'scenario', 'text', 'line_information', 'utterance_count', 'source_file']
    columns = ['text', 'speaker', 'source_file']
    df = pd.DataFrame(columns=columns)

    count = 0

    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            if fnames is None:  # EDITED if we are interested in all files
                dirname = os.path.basename(subdir)
                if str(data_dir + dirname + file).endswith(".cha"):
                    file_df = cha_txt_to_dataframe(data_dir + dirname + "\\" + file)
                    file_df['source_file'] = file

                    df = pd.concat([df, file_df], ignore_index=True)
                    # break
            elif file in fnames:  # EDITED to add files only we are interested in
                # print(data_dir + file)
                dirname = os.path.basename(subdir)
                # print(dirname)
                if str(data_dir + file).endswith(".cha"):
                    file_df = cha_txt_to_dataframe(data_dir + dirname + "\\" + file)
                    file_df['source_file'] = file
                    df = pd.concat([df, file_df], ignore_index=True)
                    # break
            # else:
            #     if str(data_dir + file).endswith(".cha"):
            #         os.remove(data_dir + file)

    if not df.empty:
        print("Saved: " + str(file_name) + " at " + str(pathlib.Path().resolve())+file_name)
        df.to_csv(file_name, index=False, encoding="utf-8")

    return True


if __name__ == "__main__":
    # data_dir3 = str(pathlib.Path().resolve()) + "\\data\\data_broca\\"
    # data_dir2 = str(pathlib.Path().resolve()) + "\\data\\data_control\\"
    # data_dir1 = str(pathlib.Path().resolve()) + "\\data\\data_pwa\\"
    data_dir4 = str(pathlib.Path().resolve().parent) + "\\linguistic_model\\data\\spoken corpus\\"
    # fnames = list(pd.read_csv('data/broca_fname.csv'))
    # print(len(fnames))
    # csv_filename1 = "data/data_broca.csv"
    # cha_txt_files_to_csv(data_dir3, csv_filename1, fnames)

    csv_filename1 = "../linguistic_model/data/spoken corpus/data_test.csv"
    cha_txt_files_to_csv(data_dir4, csv_filename1)

    # fnames = list(pd.read_csv('data/wernicke_fname.csv'))
    # print(len(fnames))
    # csv_filename1 = "data/data_wernicke.csv"
    # cha_txt_files_to_csv(data_dir1, csv_filename1, fnames)
    #
    # fnames = list(pd.read_csv('data/transsensory_fname.csv'))
    # print(len(fnames))
    # csv_filename1 = "data/data_transsensory.csv"
    # cha_txt_files_to_csv(data_dir1, csv_filename1, fnames)
    #
    # fnames = list(pd.read_csv('data/conduction_fname.csv'))
    # print(len(fnames))
    # csv_filename1 = "data/data_conduction.csv"
    # cha_txt_files_to_csv(data_dir1, csv_filename1, fnames)
    #
    # fnames = list(pd.read_csv('data/anomic_fname.csv'))
    # print(len(fnames))
    # csv_filename1 = "data/data_anomic.csv"
    # cha_txt_files_to_csv(data_dir1, csv_filename1, fnames)

    # csv_filename2 = "data/data_control.csv"
    # cha_txt_files_to_csv(data_dir2, csv_filename2)
