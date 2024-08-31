import difflib
import asyncio
import json
from collections import Counter

from openai import AsyncOpenAI
from functools import reduce
import openai
import re
import time
import os
from jinja2 import Environment, FileSystemLoader
from bs4 import BeautifulSoup


CERTIFIED_LANGUAGES = [
    "de",
    "ru",
    "en",
    "ar",
    "fr"
]

LANGUAGES = {
    "ar": "arabic",
    "bg": "bulgarian",
    "cs": "czech",
    "da": "danish",
    "de": "german",
    "el": "greek",
    "en": "english",
    "es": "spanish",
    "et": "estonian",
    "fi": "finnish",
    "fr": "french",
    "ga": "irish",
    "hr": "croation",
    "hu": "hungarian",
    "id": "indonesian",
    "it": "italian",
    "ja": "japanese",
    "ko": "korean",
    "lv": "latvian",
    "lt": "lithuanian",
    "mt": "maltese",
    "nl": "dutch",
    "pl": "polish",
    "ro": "romanian",
    "ru": "russian",
    "sk": "slovak",
    "sl": "slovenian",
    "sv": "swedish",
    "zh": "chinese",
}
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.iteragpt.iteratec.de/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "azure/gpt-4o")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gcp/gemini-1.5-pro")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "aws/claude-3-sonnet")
CLAUDE_MODEL_3_5 = os.getenv("CLAUDE_MODEL", "aws/claude-3.5-sonnet")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "azure/mistral-large")
LLAMA3_MODEL = os.getenv("LLAMA3_MODEL", "iteratec/Llama3.1-70B-Instruct")
INPUT_FILE = os.getenv("INPUT_FILE")
INPUT_FILE_DESCRIPTION = os.getenv("INPUT_FILE_DESCRIPTION")
RETRY_CHECK = (os.getenv("RETRY_CHECK", "0") == "1")
TARGET_LANGUAGE = os.getenv("TARGET_LANGUAGE", "")
SOURCE_LANGUAGE = os.getenv("SOURCE_LANGUAGE", "de")
SECOND_COMPARE_LANGUAGE = os.getenv("SECOND_COMPARE_LANGUAGE", "en")
REDO_CHATGPT = os.getenv("REDO_CHATGPT", "False").lower() == "true"


def create_td(text = "", diff: [str]=[], plusminus="+") -> str:
    td = "<td style='border-bottom:3px solid white;border-right:3px solid white;'>"

    if len(diff) > 0:
        for word in diff:
            if word[0] == plusminus:
                td += "<span style = 'color: red; font-weight: bold;'>{}</span>".format(word[1:])
            elif word[0] == "?" and len(word) > 1:
                pass
            elif word[0] != "+" and word[0] != "-":
                td += "<span style = 'color: black;'>{}</span>".format(word)
    else:
        td += text
    td += "</td>"
    return td


async def ask_model(system: str, user: str, model: str) -> str:
    try:
        client = AsyncOpenAI(base_url=OPENAI_BASE_URL)
        response = await client.chat.completions.create(
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
            max_tokens=1024,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        answer = response.choices[0].message.content
    except openai.RateLimitError as e:
        message = e.body["message"]
        print(message)
        secnum_re = re.match(".*Try again in ([0-9.]*) seconds.*", message)
        if secnum_re:
            secnum = float(secnum_re.groups()[0])
        else:
            secnum = 1
        print(f"wait for {secnum}")
        time.sleep(secnum)
        return await ask_model(system, user, model)
    except Exception as exc:
        print(f"Exception: {exc}")
        return ""
    return answer


ABOUT_SCIARA = """
In Climate Time Machine, experience a personal journey that focuses on you - discover how your lifestyle and choices impact you and society.
First, provide some information about your lifestyle, from mobility and heating to diet and consumption. If you are missing details, we can help you out with convenient estimates and average values.
Our system then calculates your individual CO2 footprint, but Climate Time Machine goes far beyond traditional CO2 calculators. In the simulation, your decisions represent those of many other people.
The emissions changes of all participants are globally projected, and scientific models provide realistic climate developments.
Based on these calculations, we will show you how your living environment could change over time. Experience trees dying or thriving, drying rivers or flooding. Recognize how the temperature is developing in your region and which cities are at risk of flooding. Discover how many hot nights are ahead and what impact this will have on crop yields.
"""


async def get_translation(
        source_text: str,
        source_language: str,
        target_language: str,
        model: str,
        file_content: str,
        file_description: str,
):
    system = (
        f"There is an application called 'Climate Time Machine' with the following description '{ABOUT_SCIARA}'"
        f"This application has a language file, with the following content: {file_content}"
        f"The description of this language file is: {file_description}"
        f"You are a language expert and an expert in climate change. In the following you get a {LANGUAGES[source_language]} text from this language file."
        f"Translate this {LANGUAGES[source_language]} text into {LANGUAGES[target_language]}. Ensure that the translated text retains the original meaning, tone, and intent."
        f"The answer has to contain ONLY the translation itself. No explaining text is allowed in the answer."
    )
    user_lines = [f"Original {LANGUAGES[source_language]}: \"{source_text}\""]
    user = "\n".join(user_lines)
    print(f"get_translation for '{source_text}'")
    answer = await ask_model(system, user, model=model)
    print(f"get_translation: answer={answer}")
    answer = answer.replace('"', '').replace("'", "") if (answer.startswith('"') or answer.startswith("'")) else answer
    return answer


async def check_translation(source_text: str, source_language: str, target_text: str, target_language: str, llm="gpt"):
    system = (
        f"There is an application called 'Climate Time Machine' with the following description '{ABOUT_SCIARA}'"
        f"You are a language expert and an expert in climate change. In the following you get an original {LANGUAGES[source_language]} text and a translation in {LANGUAGES[target_language]}."
        "Please decide whether both texts have the same meaning, tone and intent. If so, just answer with 'YES', if not, explain the difference and find a better translation."
    )
    user_lines = [f"Original {LANGUAGES[source_language]}: \"{source_text}\"", f"{LANGUAGES[target_language]} translation: \"{target_text}\""]
    user = "\n".join(user_lines)
    print(f"check_translation for '{source_text}' -> '{target_text}'")
    if llm == "gpt":
        answer = await ask_model(system, user, model=OPENAI_MODEL)
    elif llm == "gemini":
        answer = await ask_model(system, user, model=GEMINI_MODEL)
    else:
        raise Exception(f"unknown llm: {llm}")
    print(f"check_translation with {llm}: answer={answer}")
    return answer


def catch_single_digit(input_string):
    single_digit_pattern = r'\b\d\b'
    matches = re.findall(single_digit_pattern, input_string)
    return matches[0]


def find_most_common_strings(string_list: [str]):
    count = Counter(string_list)
    max_frequency = max(count.values())
    most_common_strings = [string for string, freq in count.items() if freq == max_frequency]
    return most_common_strings


async def compare_translations(
        source_text: str,
        source_language: str,
        translations: [str],
        target_language: str,
        model: str,
        file_content: str,
        file_description: str,
) -> str:
    system = (
        f"There is an application called 'Climate Time Machine' with the following description '{ABOUT_SCIARA}'"
        f"The description of this language file is: {file_description}"
        f"This application has a language file, with the following content: {file_content}"
        f"You are a language expert and an expert in climate change. In the following you get a {LANGUAGES[source_language]} text from this language file and {len(translations)} translations in {LANGUAGES[target_language]}."
        "Please decide which translation is the most accurate and the best for this application. Answer only with the number of the translation. Do NOT explain your choice!!! ONLY ONE NUMBER AS ANSWER!!!"
    )
    original_line = [f"Original {LANGUAGES[source_language]}: \"{source_text}\""]
    translation_lines = [f"translation {index+1}: \"{translation}\"" for index, translation in enumerate(translations)]
    user = "\n".join(original_line + translation_lines)
    print(f"compare_translations for '{source_text}'")
    answer = await ask_model(system=system, user=user, model=model)
    winning_index = int(catch_single_digit(answer))
    return translations[winning_index - 1]


def is_translation(path: str) -> bool:
    return path.split(".")[-1].lower() in LANGUAGES and not "imageRef" in path


def create_table(input_file: str, translation_lines: [str], source: str, target: str):
    headers = [
        "source text",
        "--- 1 ---",
        "--- 1 back ---",
        "--- 2 ---",
        "--- 2 back ---",
        "--- 3 ---",
        "--- 3 back ---",
        "--- 4 ---",
        "--- 4 back ---",
        "--- 5 ---",
        "--- 5 back ---",
        "Assess (OpenAI)",
        "Assess (Gemini)",
        "Assess (Claude)",
        "Assess (Mistral)",
        "Assess (Claude 3.5)",
    ]

    env = Environment(loader=FileSystemLoader('./templates'))
    template = env.get_template('compare_table.html.j2')
    html = template.render(headers=headers, translation_lines=translation_lines)
    with open(f"./tables/{input_file}_table.{source}_{target}.html", 'w') as f:
        f.write(html)


def create_diff_text(target: str, source: str, plusminus) -> str:
    diff_list = list(difflib.ndiff(a=source.split(), b=target.split()))
    parts: [str] = []
    for word in diff_list:
        if word[0] == plusminus:
            parts.append("<span style = 'color: red; font-weight: bold;'>{}</span>".format(word[1:]))
        elif word[0] == "?" and len(word) > 1:
            pass
        elif word[0] != "+" and word[0] != "-":
            parts.append("<span style = 'color: black;'>{}</span>".format(word))

    return " ".join(parts)


def create_diff_html(target: str, source: str, plusminus="+") -> str:
    parsed_source = BeautifulSoup(target, "html.parser")
    pasrsed_target = BeautifulSoup(source, "html.parser")
    source_nodes = list(parsed_source.descendants)
    target_nodes = list(pasrsed_target.descendants)
    if len(source_nodes) == len(target_nodes):
        transformedNodes = []
        for i in range(len(source_nodes)):
            source_node = source_nodes[i]
            target_node = target_nodes[i]
            if source_node.name is None and target_node.name is None:
                transformedNode = create_diff_text(target=str(target_node), source=str(source_node), plusminus=plusminus)
            else:
                transformedNode = str(source_node)
            transformedNodes.append(transformedNode)
        return " ".join(transformedNodes)
    else:
        return create_diff_text(target=target, source=source, plusminus=plusminus)


def create_unique_translations(translations: [str], ignored_translation_index=-1) -> [str]:
    normalized_filtered = [t.strip().strip('\'"') for i, t in enumerate(translations) if i != ignored_translation_index]
    unique_translations = list(set(normalized_filtered))
    return unique_translations


async def crawl_json(
        data,
        source_language: str,
        target_language: str,
        current_translations: dict,
        table_lines: list,
        file_content: str,
        file_description: str,
        path="",
):
    if isinstance(data, dict):
        for key, value in data.items():
            new_path = f"{path}.{key}" if path else key
            await crawl_json(
                data=value,
                path=new_path,
                source_language=source_language,
                target_language=target_language,
                current_translations=current_translations,
                table_lines=table_lines,
                file_content=file_content,
                file_description=file_description,
            )
        print(current_translations)
        if len(current_translations.keys()) > 0:
            target_language_openai_property = f"_openai_{target_language}"
            if not target_language_openai_property in data:
                target_translation_openai = await get_translation(
                    source_text=data[source_language],
                    source_language=source_language,
                    target_language=target_language,
                    model=OPENAI_MODEL,
                    file_content=file_content,
                    file_description=file_description,
                )
                data[target_language_openai_property] = target_translation_openai
            else:
                target_translation_openai = data[target_language_openai_property]

            target_language_gemini_property = f"_gemini_{target_language}"
            if not target_language_gemini_property in data:
                target_translation_gemini = await get_translation(
                    source_text=data[source_language],
                    source_language=source_language,
                    target_language=target_language,
                    model=GEMINI_MODEL,
                    file_content=file_content,
                    file_description=file_description,
                )
                data[target_language_gemini_property] = target_translation_gemini
            else:
                target_translation_gemini = data[target_language_gemini_property]

            target_language_claude_property = f"_claude_{target_language}"
            if not target_language_claude_property in data:
                target_translation_claude = await get_translation(
                    source_text=data[source_language],
                    source_language=source_language,
                    target_language=target_language,
                    model=CLAUDE_MODEL,
                    file_content=file_content,
                    file_description=file_description,
                )
                data[target_language_claude_property] = target_translation_claude
            else:
                target_translation_claude = data[target_language_claude_property]

            target_language_claude_3_5_property = f"_claude_3_5_{target_language}"
            if not target_language_claude_3_5_property in data:
                target_translation_claude_3_5 = await get_translation(
                    source_text=data[source_language],
                    source_language=source_language,
                    target_language=target_language,
                    model=CLAUDE_MODEL_3_5,
                    file_content=file_content,
                    file_description=file_description,
                )
                data[target_language_claude_3_5_property] = target_translation_claude_3_5
            else:
                target_translation_claude_3_5 = data[target_language_claude_3_5_property]

            target_language_mistral_property = f"_mistral_{target_language}"
            if not target_language_mistral_property in data:
                target_translation_mistral = await get_translation(
                    source_text=data[source_language],
                    source_language=source_language,
                    target_language=target_language,
                    model=MISTRAL_MODEL,
                    file_content=file_content,
                    file_description=file_description,
                )
                data[target_language_mistral_property] = target_translation_mistral
            else:
                target_translation_mistral = data[target_language_mistral_property]

            # target_language_llama3_property = f"_llama3_{target_language}"
            # if not target_language_llama3_property in data:
            #     target_translation_llama3 = await get_translation(
            #         source_text=data[source_language],
            #         source_language=source_language,
            #         target_language=target_language,
            #         model=LLAMA3_MODEL,
            #         file_content=file_content,
            #         file_description=file_description,
            #     )
            #     data[target_language_llama3_property] = target_translation_llama3
            # else:
            #     target_translation_llama3 = data[target_language_llama3_property]

            translation_list = [target_translation_openai, target_translation_gemini, target_translation_claude, target_translation_mistral, target_translation_claude_3_5]

            compare_result_openai_property = f"_compare_openai_{target_language}"
            if not compare_result_openai_property in data:
                compare_result_openai = await compare_translations(
                    source_text=data[source_language],
                    source_language=source_language,
                    translations=create_unique_translations(translation_list, 0),
                    target_language=target_language,
                    model=OPENAI_MODEL,
                    file_content=file_content,
                    file_description=file_description,
                )
                data[compare_result_openai_property] = compare_result_openai
            else:
                compare_result_openai = data[compare_result_openai_property]

            compare_result_gemini_property = f"_compare_gemini_{target_language}"
            if not compare_result_gemini_property in data:
                compare_result_gemini = await compare_translations(
                    source_text=data[source_language],
                    source_language=source_language,
                    translations=create_unique_translations(translation_list, 1),
                    target_language=target_language,
                    model=GEMINI_MODEL,
                    file_content=file_content,
                    file_description=file_description,
                )
                data[compare_result_gemini_property] = compare_result_gemini
            else:
                compare_result_gemini = data[compare_result_gemini_property]

            compare_result_claude_property = f"_compare_claude_{target_language}"
            if not compare_result_claude_property in data:
                compare_result_claude = await compare_translations(
                    source_text=data[source_language],
                    source_language=source_language,
                    translations=create_unique_translations(translation_list, 2),
                    target_language=target_language,
                    model=CLAUDE_MODEL,
                    file_content=file_content,
                    file_description=file_description,
                )
                data[compare_result_claude_property] = compare_result_claude
            else:
                compare_result_claude = data[compare_result_claude_property]

            compare_result_claude_3_5_property = f"_compare_claude_3_5_{target_language}"
            if not compare_result_claude_3_5_property in data:
                compare_result_claude_3_5 = await compare_translations(
                    source_text=data[source_language],
                    source_language=source_language,
                    translations=create_unique_translations(translation_list, 2),
                    target_language=target_language,
                    model=CLAUDE_MODEL_3_5,
                    file_content=file_content,
                    file_description=file_description,
                )
                data[compare_result_claude_3_5_property] = compare_result_claude_3_5
            else:
                compare_result_claude_3_5 = data[compare_result_claude_3_5_property]

            compare_result_mistral_property = f"_compare_mistral_{target_language}"
            if not compare_result_mistral_property in data:
                compare_result_mistral = await compare_translations(
                    source_text=data[source_language],
                    source_language=source_language,
                    translations=create_unique_translations(translation_list, 3),
                    target_language=target_language,
                    model=MISTRAL_MODEL,
                    file_content=file_content,
                    file_description=file_description,
                )
                data[compare_result_mistral_property] = compare_result_mistral
            else:
                compare_result_mistral = data[compare_result_mistral_property]

            # compare_result_llama3_property = f"_compare_llama3_{target_language}"
            # if not compare_result_llama3_property in data:
            #     compare_result_llama3 = await compare_translations(
            #         source_text=data[source_language],
            #         source_language=source_language,
            #         translations=create_unique_translations(translation_list, 4),
            #         target_language=target_language,
            #         model=LLAMA3_MODEL,
            #         file_content=file_content,
            #         file_description=file_description,
            #     )
            #     data[compare_result_llama3_property] = compare_result_llama3
            # else:
            #     compare_result_llama3 = data[compare_result_llama3_property]

            winners_property = f"_winners_{target_language}"
            if not winners_property in data:
                winners = find_most_common_strings(
                    [compare_result_openai, compare_result_gemini, compare_result_claude, compare_result_mistral,
                     compare_result_claude_3_5])
                data[winners_property] = winners
            else:
                winners = data[winners_property]

            unique_translations_property = f"_unique_translations_{target_language}"
            if not unique_translations_property in data:
                unique_translations = create_unique_translations(translation_list)
                data[unique_translations_property] = unique_translations
            else:
                unique_translations = data[unique_translations_property]

            unique_translations_back_property = f"_unique_translations_back_{target_language}"
            if not unique_translations_back_property in data:
                unique_translations_back = [
                    await get_translation(
                        source_text=translation,
                        source_language=target_language,
                        target_language=source_language,
                        model=OPENAI_MODEL,
                        file_content=file_content,
                        file_description=file_description,
                    )
                    for translation in unique_translations
                ]
                data[unique_translations_back_property] = unique_translations_back
            else:
                unique_translations_back = data[unique_translations_back_property]

            table_lines.append({
                "id": path,
                "source_text": current_translations[source_language],
                "translation_1": unique_translations[0],
                "translation_1_raw": unique_translations[0],
                "translation_1_back": unique_translations_back[0],
                "translation_2": create_diff_html(unique_translations[0], unique_translations[1]) if len(unique_translations) > 1 else "",
                "translation_2_raw": unique_translations[1] if len(unique_translations) > 1 else "",
                "translation_2_back": unique_translations_back[1] if len(unique_translations_back) > 1 else "",
                "translation_3": create_diff_html(unique_translations[0], unique_translations[2]) if len(unique_translations) > 2 else "",
                "translation_3_raw": unique_translations[2] if len(unique_translations) > 2 else "",
                "translation_3_back": unique_translations_back[2] if len(unique_translations_back) > 2 else "",
                "translation_4": create_diff_html(unique_translations[0], unique_translations[3]) if len(unique_translations) > 3 else "",
                "translation_4_raw": unique_translations[3] if len(unique_translations) > 3 else "",
                "translation_4_back": unique_translations_back[3] if len(unique_translations_back) > 3 else "",
                "translation_5": create_diff_html(unique_translations[0], unique_translations[4]) if len(unique_translations) > 4 else "",
                "translation_5_raw": unique_translations[4] if len(unique_translations) > 4 else "",
                "translation_5_back": unique_translations_back[4] if len(unique_translations_back) > 4 else "",
                "compare_openai": compare_result_openai,
                "compare_gemini": compare_result_gemini,
                "compare_claude": compare_result_claude,
                "compare_mistral": compare_result_mistral,
                "compare_claude_3_5": compare_result_claude_3_5,
                "winners": winners,
                "unique_translations": unique_translations,
            })
            current_translations.clear()
    elif isinstance(data, list):
        for i, value in enumerate(data):
            new_path = f"{path}[{i}]"
            await crawl_json(
                data=value,
                path=new_path,
                source_language=source_language,
                target_language=target_language,
                current_translations=current_translations,
                table_lines=table_lines,
                file_content=file_content,
                file_description=file_description,
            )
    else:
        if is_translation(path):
            language = path.split(".")[-1]
            current_translations[language] = data
        print(f"{path}: {data}")


async def process_i18n_file(file_path: str, file_description:str, target_language="") -> {str: [object]}:
    with open(file_path) as f:
        data = json.load(f)
    with open(file_path) as f:
        content = f.read()

    table_lines_dict = {}
    for _target_language in (LANGUAGES.keys() if len(target_language) == 0 else [target_language]):
        table_lines = []
        await crawl_json(
            data,
            source_language=SOURCE_LANGUAGE,
            target_language=_target_language,
            current_translations={},
            table_lines=table_lines,
            file_content=content,
            file_description=file_description,
        )
        with open(file_path, "w", encoding="UTF-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        table_lines_dict[_target_language] = table_lines
    return table_lines_dict

async def main():
    table_lines_dict = await process_i18n_file(file_path=INPUT_FILE, file_description=INPUT_FILE_DESCRIPTION, target_language=TARGET_LANGUAGE)
    for target_language in table_lines_dict:
        create_table(translation_lines=table_lines_dict[target_language], source=SOURCE_LANGUAGE, target=target_language, input_file=os.path.basename(INPUT_FILE))


if __name__ == "__main__":
    asyncio.run(main())
