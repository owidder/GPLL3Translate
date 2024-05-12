import os
import re
import time
import openai
import requests
import json
import csv
from jinja2 import Environment, FileSystemLoader
import pandas as pd
import sys
import difflib


openai.api_key = os.getenv("OPENAI_API_KEY")
min_length = os.getenv("MIN_LENGTH")
max_sentence_no = int(os.getenv("MAX_SENTENCE_NO", sys.maxsize))

LANGUAGES = {
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "nl": "Dutch",
    "sv": "Swedish",
    "zh": "Chinese",
}


SOURCE_LANGUAGE = "English"
TARGET_LANGUAGE = "German"


def retrieve_sentence(sentence_number: int):
    response = requests.get(f"https://tatoeba.org/en/api_v0/sentence/{sentence_number}")
    js = json.loads(response.text)
    translations = [j for i in js["translations"] for j in i]
    print(translations)


def read_sentence_with_translations(sentence_number: int, source_language: str, target_language: str) -> (str, [(str, str)], str):
    response = requests.get(f"https://tatoeba.org/en/api_v0/sentence/{sentence_number}")
    js = json.loads(response.text)
    translations = [j for i in js["translations"] for j in i]
    translations_list = [(t["lang_name"], t["text"]) for t in filter(lambda l: l["lang_name"] not in [source_language, target_language], translations)]
    target_text_list = [t["text"] for t in filter(lambda l: l["lang_name"] == target_language, translations)]
    return (js["text"], translations_list, target_text_list[0] if len(target_text_list) > 0 else "")


def ask_chatgpt(system: str, user: str, model: str) -> str:
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system
                },
                {
                    "role": "user",
                    "content": user
                }
            ],
            temperature=0,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        answer = response["choices"][0]["message"]["content"]
    except openai.error.RateLimitError as e:
        print(e.json_body["error"]["message"])
        secnum_re = re.match(".*Please try again in ([0-9.]*)s\..*", e.json_body["error"]["message"])
        if secnum_re:
            secnum = float(secnum_re.groups()[0])
        else:
            secnum = 1
        print(f"wait for {secnum}")
        time.sleep(secnum)
        return ask_chatgpt(system, user, model)
    return answer


def get_translation(source_text: str, source_language: str, target_language: str, translations: [(str, str)]):
    system = (
        f"You are an expert in all languages. In the following you get an original {source_language} text {'and several translations into other languages' if len(translations) > 0 else ''}."
        f"Translate the original {source_language} text into {target_language}. Ensure that the translated text retains the original meaning, tone, and intent."
        f"The answer has to contain ONLY the translation itself. No explaining text. Otherwise the answer is NOT CORRECT"
    )
    user_lines = [f"Original {source_language}: \"{source_text}\""]
    if len(translations) > 0:
        user_lines.extend([f"{language} Translation: \"{translation}\"" for language, translation in translations])
    user = "\n".join(user_lines)
    print(f"get_translation for '{source_text}'")
    answer = ask_chatgpt(system, user, model="gpt-4")
    print(f"get_translation: answer={answer}")
    normalized_answer = answer.replace("\n", " ")
    return normalized_answer


def check_translation(source_text: str, source_language: str, target_text: str, target_language: str):
    system = (
        f"You are an expert in all languages. In the following you get an original {source_language} text and a translation in {target_language}."
        "Please decide whether both texts have the same meaning, tone and intent. If so, just answer with 'YES', if not, explain the difference and find a better translation."
    )
    user_lines = [f"Original {source_language}: \"{source_text}\"", f"{target_language} translation: \"{target_text}\""]
    user = "\n".join(user_lines)
    print(f"check_translation for '{source_text}' -> '{target_text}'")
    answer = ask_chatgpt(system, user, model="gpt-4")
    print(f"check_translation: answer={answer}")
    return answer


def read_tsv_file(filename: str, max_number_of_sentences = sys.maxsize):
    """
    Reads a TSV file and returns the data as a list of dictionaries.

    :param filename: The name of the TSV file to read.
    :return: A list of dictionaries, where each dictionary represents a row of data.
    """
    data = []

    df = pd.read_csv(filename, compression='infer', sep='\t',
                     names=["sentence_no", "language", "text", "author", "ts1", "ts2"])
    current_number_of_sentences = 0
    for index, row in df.iterrows():
        data.append(row.to_dict())
        current_number_of_sentences += 1
        if current_number_of_sentences >= max_number_of_sentences:
            break

    return data


def normalize(str):
    return str.replace('"', '')


def read_translated_sentences(filename) -> dict[int, dict]:
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as file:
            sentences: dict[int, dict] = {}
            fieldnames = [
                "sentence_no",
                "ne",
                "source_text",
                "with_translations",
                "without_translations",
                "t1",
                "with_translations_back",
                "without_translations_back",
                "check_with_translations",
                "check_without_translations",
            ]
            reader = csv.DictReader(file, delimiter='\t', fieldnames=fieldnames)
            for row in reader:
                sentences[int(row['sentence_no'])] = row
        return sentences
    else:
        return {}


def add_to_translated_sentences(filename: str, new_line: str):
    new_sentence_no = new_line.split("\t")[0]

    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for index, line in enumerate(lines):
        sentence_no = line.split("\t")[0]
        if sentence_no == new_sentence_no:
            lines[index] = new_line
            break
        elif index == len(lines)-1:
            lines.append(new_line)

    with open(filename, 'w', encoding='utf-8') as file:
        file.writelines(lines)
        file.flush()


def create_diff_html(textA: str, textB: str, plusminus = "+") -> str:
    diff_list = list(difflib.ndiff(textA.split(), textB.split()))
    html = ""
    for word in diff_list:
        if word[0] == plusminus:
            html += "<span style = 'color: red; font-weight: bold;'>{}</span>".format(word[1:])
        elif word[0] == "?" and len(word) > 1:
            pass
        elif word[0] != "+" and word[0] != "-":
            html += "<span style = 'color: black;'>{}</span>".format(word)

    return html


def translate_and_check() -> [str]:
    translation_lines = []
    out_name = f"./translations/eng_translations_detailed.{min_length}.tsv"
    translated_sentences = read_translated_sentences(filename=out_name)
    sentences = read_tsv_file(filename=f"./source_texts/eng_sentences_detailed.{min_length}.tsv", max_number_of_sentences=100)
    for sentence in sentences:
        sentence_no = sentence["sentence_no"]
        if sentence_no > max_sentence_no:
            break

        translated_sentence = translated_sentences.get(sentence_no, {})
        target_text = translated_sentence.get("t1", "")
        with_translations = translated_sentence.get("with_translations", "") or ""
        without_translations = translated_sentence.get("without_translations", "") or ""
        with_translations_back = translated_sentence.get("without_translations_back", "") or ""
        without_translations_back = translated_sentence.get("without_translations_back", "") or ""
        check_with_translations = translated_sentence.get("check_with_translations", "") or ""
        check_without_translations = translated_sentence.get("check_without_translations", "") or ""
        source_text = translated_sentence.get("source_text", "")
        translations = []

        if len(source_text) == 0 or len(target_text) == 0 or len(with_translations) == 0 or len(without_translations) == 0 or (not with_translations_back or len(with_translations_back) == 0):
            (source_text, translations, target_text) = read_sentence_with_translations(sentence_no, source_language="English", target_language="German")

        if len(with_translations) == 0:
            with_translations = normalize(get_translation(source_text=source_text, translations=translations, source_language="English", target_language="German"))

        if len(without_translations) == 0:
            without_translations = normalize(get_translation(source_text=source_text, translations=[], source_language="English", target_language="German"))

        if not with_translations_back or len(with_translations_back) == 0:
            with_translations_back = normalize(get_translation(source_text=with_translations, translations=translations, source_language="German", target_language="English"))

        with_translations_differs_from_without_translations = with_translations != without_translations
        if not without_translations_back or len(without_translations_back) == 0:
            if with_translations_differs_from_without_translations:
                without_translations_back = normalize(get_translation(source_text=without_translations, translations=[], source_language="German", target_language="English"))
            else:
                without_translations_back = with_translations_back

        if len(check_with_translations) == 0:
            check_with_translations = check_translation(source_text=source_text, source_language=SOURCE_LANGUAGE, target_language=TARGET_LANGUAGE, target_text=with_translations)

        if len(check_without_translations) == 0:
            if with_translations_differs_from_without_translations:
                check_without_translations = check_translation(source_text=source_text, source_language=SOURCE_LANGUAGE, target_language=TARGET_LANGUAGE, target_text=without_translations)
            else:
                check_without_translations = check_with_translations

        ne = 'e' if with_translations == without_translations else 'n'

        line = (
            f"{sentence_no}\t{ne}"
            f"\t{source_text}\t{with_translations}\t{without_translations}\t{target_text}"
            f"\t{with_translations_back}\t{without_translations_back}"
            f"\t{check_with_translations}\t{check_without_translations}"
            "\n"
        )
        add_to_translated_sentences(filename=out_name, new_line=line)
        translation_lines.append({
            "sentence_no": sentence_no,
            "ne": ne,
            "source_text": source_text,
            "with_translations": create_diff_html(textA=with_translations, textB=without_translations, plusminus="-"),
            "without_translations": create_diff_html(textA=with_translations, textB=without_translations),
            "with_vs_target_text": create_diff_html(textA=with_translations, textB=target_text),
            "without_vs_target_text": create_diff_html(textA=without_translations, textB=target_text),
            "with_translations_back": create_diff_html(textA=source_text, textB=with_translations_back),
            "without_translations_back": create_diff_html(textA=source_text, textB=without_translations_back),
            "check_with_translations": check_with_translations,
            "check_without_translations": check_without_translations,
        })

    return translation_lines


def create_table(translation_lines: [str]):
    headers = [
        "source text",
        "with translations",
        "without translations",
        "official vs. with translations",
        "official vs. without translations",
        "with translations back vs. source text",
        "without translations back vs. source text",
        "check with translations",
        "check without translations",
    ]

    env = Environment(loader=FileSystemLoader('./templates'))
    template = env.get_template('simple_table.html.j2')
    html = template.render(headers=headers, translation_lines=translation_lines)
    with open(f"./tables/eng_table_detailed.{min_length}.html", 'w') as f:
        f.write(html)


if __name__ == '__main__':
    translation_lines = translate_and_check()
    create_table(translation_lines=translation_lines)
