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
GPT_4O = "azure/gpt-4o"
GPT_4_VISION = "azure/gpt-4-vision"
GEMINI_1_5_PRO = "gcp/gemini-1.5-pro"
CLAUDE_3_SONNET = "aws/claude-3-sonnet"
CLAUDE_3_5_SONNET = "aws/claude-3.5-sonnet"
CLAUDE_3_HAIKU = "aws/claude-3-haiku"
MISTRAL_LARGE = "azure/mistral-large"
LLAMA_3_1_70B = "iteratec/Llama3.1-70B-Instruct"
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


async def ask_model(system: str, user: str, model: str, no_of_tries=0) -> str:
    if no_of_tries > 2:
        return ""

    print(f"ask model: {model}")
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
        return await ask_model(system, user, model, no_of_tries=no_of_tries+1)
    except Exception as exc:
        print(f"Exception: {exc}")
        return ""

    if answer == None:
        print("No answer!!!! Wait for 1s and try again")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        time.sleep(1)
        return await ask_model(f"{system}\nPLEASE ALWAYS ANSWER WITH A STRING!!!!", user, model, no_of_tries=no_of_tries+1)

    print(f"answer: {answer}")
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
        file_content="",
        file_description="",
):
    system_msg = \
        f"There is an application called 'Climate Time Machine' with the following description '{ABOUT_SCIARA}'\n" + \
        (f"This application has a language file, with the following content: [{file_content}]\n" if len(file_content) > 0 else "") + \
        (f"The description of this language file is: {file_description}\n" if len(file_description) > 0 else "") + \
        f"You are a language expert and an expert in climate change. In the following you get a {LANGUAGES[source_language]} text from this language file.\n" + \
        f"Translate this {LANGUAGES[source_language]} text into {LANGUAGES[target_language]}. Ensure that the translated text retains the original meaning, tone, and intent.\n" + \
        f"The answer has to contain ONLY the translation itself. No explaining text is allowed in the answer.\n"

    user_lines = [f"Original {LANGUAGES[source_language]}: \"{source_text}\""]
    user = "\n".join(user_lines)
    print(f"get_translation for '{source_text}'")
    answer = await ask_model(system_msg, user, model=model)
    print(f"get_translation: answer={answer}")
    print("----------------------------------------")
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
        answer = await ask_model(system, user, model=GPT_4O)
    elif llm == "gemini":
        answer = await ask_model(system, user, model=GEMINI_1_5_PRO)
    else:
        raise Exception(f"unknown llm: {llm}")
    print(f"check_translation with {llm}: answer={answer}")
    print("----------------------------------------")
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
        f"There is an application called 'Climate Time Machine' with the following description '{ABOUT_SCIARA}'\n"
        f"The description of this language file is: {file_description}\n"
        f"This application has a language file, with the following content: {file_content}\n"
        f"You are a language expert and an expert in climate change. In the following you get a {LANGUAGES[source_language]} text from this language file and {len(translations)} translations in {LANGUAGES[target_language]}.\n"
        "Please decide which translation is the most accurate and the best for this application. Answer only with the number of the translation. Do NOT explain your choice!!! ONLY ONE NUMBER AS ANSWER!!!"
    )
    if len(translations) > 1:
        original_line = [f"Original {LANGUAGES[source_language]}: \"{source_text}\""]
        translation_lines = [f"translation {index+1}: \"{translation}\"" for index, translation in enumerate(translations)]
        user = "\n".join(original_line + translation_lines)
        print(f"compare_translations for '{source_text}'")
        answer = await ask_model(system=system, user=user, model=model)
        print(f"answer: {answer}")
        if answer:
            winning_index = int(catch_single_digit(answer))
            return normalize_string(translations[winning_index - 1])
        else:
            return ""
    else:
        return translations[0]


async def get_best_back_translation(
        source_text: str,
        source_language: str,
        target_language: str,
        translate_models: [str],
        assess_models: [str],
        file_content: str,
        file_description: str,
) -> str:
    file_content_dict = json.loads(file_content)
    delete_key_recursive(file_content_dict, target_language)
    file_content_without_target_language = json.dumps(file_content_dict, indent=4, ensure_ascii=False)
    translations = [
        await get_translation(
            source_text=source_text,
            source_language=source_language,
            target_language=target_language,
            model=model,
            file_content=file_content_without_target_language,
            file_description=file_description,
        ) for model in translate_models
    ]
    best_translations = [
        await compare_translations(
            source_text=source_text,
            source_language=source_language,
            target_language=target_language,
            translations=translations,
            model=model,
            file_content=file_content,
            file_description=file_description
        ) for model in assess_models
    ]
    best_translation = normalize_string(find_most_common_strings(best_translations)[0])
    return best_translation


def is_translation(path: str) -> bool:
    return path.split(".")[-1].lower() in LANGUAGES and not "imageRef" in path


def create_table(input_file: str, translation_lines: [str], source: str, target: str):
    headers = [
        "source text",
        "--- 1 ---",
        "",
        "--- 2 ---",
        "",
        "Assess (GPT 4 Vision)",
        "Assess (Gemini 1.5 Pro)",
        "Assess (Claude 3.5 Sonnet)",
    ]

    env = Environment(loader=FileSystemLoader('./templates'))
    template = env.get_template('compare_table.html.j2')
    html = template.render(headers=headers, translation_lines=translation_lines)
    with open(f"./tables/{input_file}_table.{source}_{target}.light.html", 'w') as f:
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
    normalized_filtered = [normalize_string(in_string=t) for i, t in enumerate(translations) if i != ignored_translation_index]
    unique_translations = list(set(normalized_filtered))
    return unique_translations


def delete_key_recursive(data, key_to_delete):
    if isinstance(data, dict):
        if key_to_delete in data:
            del data[key_to_delete]
        for key in list(data.keys()):
            delete_key_recursive(data[key], key_to_delete)
    elif isinstance(data, list):
        for item in data:
            delete_key_recursive(item, key_to_delete)


def normalize_string(in_string: str) -> str:
    norm_with_spaces = in_string.strip().strip('\'"') \
            .replace("\n", " ") \
            .replace("<br>", " ") \
            .rstrip(".")
    norm = ' '.join(norm_with_spaces.split())
    return norm


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
        if len(current_translations.keys()) > 0:
            # translation_gpt_4o_property = f"_gpt_4o_{target_language}"
            # if not translation_gpt_4o_property in data:
            #     translation_gpt_4o = await get_translation(
            #         source_text=data[source_language],
            #         source_language=source_language,
            #         target_language=target_language,
            #         model=GPT_4O,
            #         file_content=file_content,
            #         file_description=file_description,
            #     )
            #     data[translation_gpt_4o_property] = translation_gpt_4o
            # else:
            #     translation_gpt_4o = data[translation_gpt_4o_property]

            translation_gpt_4_vision_property = f"_gpt_4_vision_{target_language}"
            if not translation_gpt_4_vision_property in data:
                translation_gpt_4_vision = await get_translation(
                    source_text=data[source_language],
                    source_language=source_language,
                    target_language=target_language,
                    model=GPT_4_VISION,
                    file_content=file_content,
                    file_description=file_description,
                )
                data[translation_gpt_4_vision_property] = translation_gpt_4_vision
            else:
                translation_gpt_4_vision = data[translation_gpt_4_vision_property]

            translation_gemini_1_5_pro_property = f"_gemini_1_5_pro_{target_language}"
            if not translation_gemini_1_5_pro_property in data:
                translation_gemini_1_5_pro = await get_translation(
                    source_text=data[source_language],
                    source_language=source_language,
                    target_language=target_language,
                    model=GEMINI_1_5_PRO,
                    file_content=file_content,
                    file_description=file_description,
                )
                data[translation_gemini_1_5_pro_property] = translation_gemini_1_5_pro
            else:
                translation_gemini_1_5_pro = data[translation_gemini_1_5_pro_property]

            # translation_claude_3_sonnet_property = f"_claude_3_sonnet_{target_language}"
            # if not translation_claude_3_sonnet_property in data:
            #     translation_claude_3_sonnet = await get_translation(
            #         source_text=data[source_language],
            #         source_language=source_language,
            #         target_language=target_language,
            #         model=CLAUDE_3_SONNET,
            #         file_content=file_content,
            #         file_description=file_description,
            #     )
            #     data[translation_claude_3_sonnet_property] = translation_claude_3_sonnet
            # else:
            #     translation_claude_3_sonnet = data[translation_claude_3_sonnet_property]

            # translation_claude_3_5_property = f"_claude_3_5_sonnet_{target_language}"
            # if not translation_claude_3_5_property in data:
            #     translation_claude_3_5_sonnet = await get_translation(
            #         source_text=data[source_language],
            #         source_language=source_language,
            #         target_language=target_language,
            #         model=CLAUDE_3_5_SONNET,
            #         file_content=file_content,
            #         file_description=file_description,
            #     )
            #     data[translation_claude_3_5_property] = translation_claude_3_5_sonnet
            # else:
            #     translation_claude_3_5_sonnet = data[translation_claude_3_5_property]

            # translation_claude_3_haiku_property = f"_claude_3_haiku_{target_language}"
            # if not translation_claude_3_haiku_property in data:
            #     translation_claude_3_haiku = await get_translation(
            #         source_text=data[source_language],
            #         source_language=source_language,
            #         target_language=target_language,
            #         model=CLAUDE_3_HAIKU,
            #         file_content=file_content,
            #         file_description=file_description,
            #     )
            #     data[translation_claude_3_haiku_property] = translation_claude_3_haiku
            # else:
            #     translation_claude_3_haiku = data[translation_claude_3_haiku_property]

            # translation_mistral_large_property = f"_mistral_large_{target_language}"
            # if not translation_mistral_large_property in data:
            #     translation_mistral_large = await get_translation(
            #         source_text=data[source_language],
            #         source_language=source_language,
            #         target_language=target_language,
            #         model=MISTRAL_LARGE,
            #         file_content=file_content,
            #         file_description=file_description,
            #     )
            #     data[translation_mistral_large_property] = translation_mistral_large
            # else:
            #     translation_mistral_large = data[translation_mistral_large_property]

            # translation_llama_3_1_70B_property = f"_llama_3_1_70B_{target_language}"
            # if not translation_llama_3_1_70B_property in data:
            #     translation_llama_3_1_70B = await get_translation(
            #         source_text=data[source_language],
            #         source_language=source_language,
            #         target_language=target_language,
            #         model=LLAMA_3_1_70B,
            #         file_content=file_content,
            #         file_description=file_description,
            #     )
            #     data[translation_llama_3_1_70B_property] = translation_llama_3_1_70B
            # else:
            #     translation_llama_3_1_70B = data[translation_llama_3_1_70B_property]

            translation_list = [
                t for t in [translation_gpt_4_vision, translation_gemini_1_5_pro] if t
            ]

            # compare_result_gpt_4o_property = f"_compare_gpt_4o_{target_language}"
            # if not compare_result_gpt_4o_property in data:
            #     compare_result_gpt_4o = await compare_translations(
            #         source_text=data[source_language],
            #         source_language=source_language,
            #         translations=create_unique_translations(translation_list, 0),
            #         target_language=target_language,
            #         model=GPT_4O,
            #         file_content=file_content,
            #         file_description=file_description,
            #     )
            #     data[compare_result_gpt_4o_property] = compare_result_gpt_4o
            # else:
            #     compare_result_gpt_4o = data[compare_result_gpt_4o_property]

            compare_result_gpt_4_vision_property = f"_compare_gpt_4_vision_{target_language}"
            if not compare_result_gpt_4_vision_property in data:
                compare_result_gpt_4_vision = await compare_translations(
                    source_text=data[source_language],
                    source_language=source_language,
                    translations=create_unique_translations(translation_list, 0),
                    target_language=target_language,
                    model=GPT_4_VISION,
                    file_content=file_content,
                    file_description=file_description,
                )
                data[compare_result_gpt_4_vision_property] = compare_result_gpt_4_vision
            else:
                compare_result_gpt_4_vision = data[compare_result_gpt_4_vision_property]

            compare_result_gemini_1_5_pro_property = f"_compare_gemini_1_5_pro_{target_language}"
            if not compare_result_gemini_1_5_pro_property in data:
                compare_result_gemini_1_5_pro = await compare_translations(
                    source_text=data[source_language],
                    source_language=source_language,
                    translations=create_unique_translations(translation_list),
                    target_language=target_language,
                    model=GEMINI_1_5_PRO,
                    file_content=file_content,
                    file_description=file_description,
                )
                data[compare_result_gemini_1_5_pro_property] = compare_result_gemini_1_5_pro
            else:
                compare_result_gemini_1_5_pro = data[compare_result_gemini_1_5_pro_property]

            # compare_result_claude_3_sonnet_property = f"_compare_claude_3_sonnet_{target_language}"
            # if not compare_result_claude_3_sonnet_property in data:
            #     compare_result_claude_3_sonnet = await compare_translations(
            #         source_text=data[source_language],
            #         source_language=source_language,
            #         translations=create_unique_translations(translation_list),
            #         target_language=target_language,
            #         model=CLAUDE_3_SONNET,
            #         file_content=file_content,
            #         file_description=file_description,
            #     )
            #     data[compare_result_claude_3_sonnet_property] = compare_result_claude_3_sonnet
            # else:
            #     compare_result_claude_3_sonnet = data[compare_result_claude_3_sonnet_property]

            # compare_result_mistral_large_property = f"_compare_mistral_large_{target_language}"
            # if not compare_result_mistral_large_property in data:
            #     compare_result_mistral_large = await compare_translations(
            #         source_text=data[source_language],
            #         source_language=source_language,
            #         translations=create_unique_translations(translation_list),
            #         target_language=target_language,
            #         model=MISTRAL_LARGE,
            #         file_content=file_content,
            #         file_description=file_description,
            #     )
            #     data[compare_result_mistral_large_property] = compare_result_mistral_large
            # else:
            #     compare_result_mistral_large = data[compare_result_mistral_large_property]

            compare_result_claude_3_5_sonnet_property = f"_compare_claude_3_5_sonnet_{target_language}"
            if not compare_result_claude_3_5_sonnet_property in data:
                compare_result_claude_3_5_sonnet = await compare_translations(
                    source_text=data[source_language],
                    source_language=source_language,
                    translations=create_unique_translations(translation_list),
                    target_language=target_language,
                    model=CLAUDE_3_5_SONNET,
                    file_content=file_content,
                    file_description=file_description,
                )
                data[compare_result_claude_3_5_sonnet_property] = compare_result_claude_3_5_sonnet
            else:
                compare_result_claude_3_5_sonnet = data[compare_result_claude_3_5_sonnet_property]

            # compare_result_claude_3_haiku_property = f"_compare_claude_3_haiku_{target_language}"
            # if not compare_result_claude_3_haiku_property in data:
            #     compare_result_claude_3_haiku = await compare_translations(
            #         source_text=data[source_language],
            #         source_language=source_language,
            #         translations=create_unique_translations(translation_list, 3),
            #         target_language=target_language,
            #         model=CLAUDE_3_HAIKU,
            #         file_content=file_content,
            #         file_description=file_description,
            #     )
            #     data[compare_result_claude_3_haiku_property] = compare_result_claude_3_haiku
            # else:
            #     compare_result_claude_3_haiku = data[compare_result_claude_3_haiku_property]

            # compare_result_llama_3_1_70B_property = f"_compare_llama_3_1_70B_{target_language}"
            # if not compare_result_llama_3_1_70B_property in data:
            #     compare_result_llama_3_1_70B = await compare_translations(
            #         source_text=data[source_language],
            #         source_language=source_language,
            #         translations=create_unique_translations(translation_list),
            #         target_language=target_language,
            #         model=LLAMA_3_1_70B,
            #         file_content=file_content,
            #         file_description=file_description,
            #     )
            #     data[compare_result_llama_3_1_70B_property] = compare_result_llama_3_1_70B
            # else:
            #     compare_result_llama_3_1_70B = data[compare_result_llama_3_1_70B_property]

            winners_property = f"_winners_{target_language}"
            if not winners_property in data:
                winners = find_most_common_strings(
                    [compare_result_gpt_4_vision_property, compare_result_gemini_1_5_pro_property, compare_result_claude_3_5_sonnet])
                data[winners_property] = winners
            else:
                winners = data[winners_property]

            unique_translations_property = f"_unique_translations_{target_language}"
            if not unique_translations_property in data:
                unique_translations = create_unique_translations(translation_list)
                data[unique_translations_property] = unique_translations
            else:
                unique_translations = data[unique_translations_property]

            def get_translation_sources(translation: str) -> [str]:
                sources = []
                normalized_translation = normalize_string(translation)
                if normalized_translation == normalize_string(translation_gpt_4_vision):
                    sources.append("GPT-4 Vision")
                if normalized_translation == normalize_string(translation_gemini_1_5_pro):
                    sources.append("Gemini 1.5 Pro")
                # if normalized_translation == normalize_string(translation_mistral_large):
                #     sources.append("Mistral Large")
                # if normalized_translation == normalize_string(translation_claude_3_5_sonnet):
                #     sources.append("Claude 3.5 Sonnet")
                # if normalized_translation == normalize_string(translation_llama_3_1_70B):
                #     sources.append("Llama 3.1 70B")
                return ', '.join(sources)

            translation_sources = [get_translation_sources(unique_translation) for unique_translation in unique_translations]

            unique_translations_back_property = f"_unique_translations_back_{target_language}"
            if not unique_translations_back_property in data:
                unique_translations_back = [
                    await get_best_back_translation(
                        source_text=translation,
                        source_language=target_language,
                        target_language=source_language,
                        translate_models=[GPT_4_VISION, GEMINI_1_5_PRO, CLAUDE_3_5_SONNET, LLAMA_3_1_70B, MISTRAL_LARGE],
                        assess_models=[GPT_4_VISION, GEMINI_1_5_PRO, CLAUDE_3_5_SONNET, LLAMA_3_1_70B, MISTRAL_LARGE],
                        file_content=file_content,
                        file_description=file_description,
                    ) if translation in winners else ""
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
                "translation_1_back": create_diff_html(current_translations[source_language], unique_translations_back[0]) if len(unique_translations_back[0]) > 0 else "",
                "translation_2": create_diff_html(unique_translations[0], unique_translations[1]) if len(unique_translations) > 1 else "",
                "translation_2_raw": unique_translations[1] if len(unique_translations) > 1 else "",
                "translation_2_back": create_diff_html(current_translations[source_language], unique_translations_back[1]) if len(unique_translations_back) > 1 and len(unique_translations_back[1]) > 0 else "",
                "translation_3": create_diff_html(unique_translations[0], unique_translations[2]) if len(unique_translations) > 2 else "",
                "translation_3_raw": unique_translations[2] if len(unique_translations) > 2 else "",
                "translation_3_back": create_diff_html(current_translations[source_language], unique_translations_back[2]) if len(unique_translations_back) > 2 and len(unique_translations_back[2]) > 0 else "",
                "translation_4": create_diff_html(unique_translations[0], unique_translations[3]) if len(unique_translations) > 3 else "",
                "translation_4_raw": unique_translations[3] if len(unique_translations) > 3 else "",
                "translation_4_back": create_diff_html(current_translations[source_language], unique_translations_back[3]) if len(unique_translations_back) > 3 and len(unique_translations_back[3]) > 0 else "",
                "translation_5": create_diff_html(unique_translations[0], unique_translations[4]) if len(unique_translations) > 4 else "",
                "translation_5_raw": unique_translations[4] if len(unique_translations) > 4 else "",
                "translation_5_back": create_diff_html(current_translations[source_language], unique_translations_back[4]) if len(unique_translations_back) > 4 and len(unique_translations_back[4]) > 0 else "",
                # "compare_gpt_4o": compare_result_gpt_4o,
                "compare_gpt_4_vision": compare_result_gpt_4_vision,
                "compare_gemini_1_5_pro": compare_result_gemini_1_5_pro,
                # "compare_mistral_large": compare_result_mistral_large,
                "compare_claude_3_5_sonnet": compare_result_claude_3_5_sonnet,
                # "compare_claude_3_haiku": compare_result_claude_3_haiku,
                # "compare_llama_3_1_70B": compare_result_llama_3_1_70B,
                "winners": winners,
                "unique_translations": unique_translations,
                "translation_sources": translation_sources,
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
