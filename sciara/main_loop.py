import difflib
import asyncio
import json
from openai import AsyncOpenAI
from functools import reduce
import openai
import re
import time
import os
from jinja2 import Environment, FileSystemLoader
from bs4 import BeautifulSoup

INPUT_FILE = os.getenv("INPUT_FILE")
TARGET_LANGUAGE = os.getenv("TARGET_LANGUAGE", "")
SOURCE_LANGUAGE = os.getenv("SOURCE_LANGUAGE", "en")

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

def is_translation(path: str) -> bool:
    return path.split(".")[-1].lower() in LANGUAGES and not "imageRef" in path


async def ask_chatgpt(system: str, user: str, model: str) -> str:
    try:
        client = AsyncOpenAI()
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
    except Exception as exc:
        print(f"Exception: {exc}")
        return ""
    return answer


async def get_translation(source_text: str, source_language: str, target_language: str):
    system = (
        f"You are an expert in all languages and climate change. In the following you get an original {LANGUAGES[source_language]} text."
        f"Translate the original {LANGUAGES[source_language]} text into {LANGUAGES[target_language]}. Ensure that the translated text retains the original meaning, tone, and intent."
        f"The answer has to contain ONLY the translation itself. No explaining text. Otherwise the answer is NOT CORRECT"
    )
    user = f"Original {LANGUAGES[source_language]}: \"{source_text}\""
    print(f"get_translation for '{source_text}'")
    answer = await ask_chatgpt(system, user, model="gpt-4o")
    print(f"get_translation: answer={answer}")
    answer = answer.replace('"', '').replace("'", "") if (answer.startswith('"') or answer.startswith("'")) else answer
    return answer

async def check_translation(source_text: str, source_language: str, target_text: str, target_language: str, llm="gpt"):
    system = (
        f"You are an expert in all languages and climate change. In the following you get an original {LANGUAGES[source_language]} text and a translation in {LANGUAGES[target_language]}."
        "Please decide whether both texts have the same meaning, tone and intent. If so, just answer with 'YES', if not, explain the difference and find a better translation."
    )
    user_lines = [f"Original {LANGUAGES[source_language]}: \"{source_text}\"", f"{LANGUAGES[target_language]} translation: \"{target_text}\""]
    user = "\n".join(user_lines)
    print(f"check_translation for '{source_text}' -> '{target_text}'")
    answer = await ask_chatgpt(system, user, model="gpt-4o")
    print(f"check_translation with {llm}: answer={answer}")
    return answer


async def create_translation_property(data, source_language: str, target_language: str, prefix: str) -> str:
    json_property_name = f"{prefix}{target_language}"
    if json_property_name not in data:
        target_translation = await get_translation(
            source_text=data[source_language],
            source_language=source_language,
            target_language=target_language,
        )
        data[json_property_name] = target_translation
    else:
        target_translation = data[json_property_name]
    return target_translation


async def check_translation_property(data, source_language: str, target_language: str, target_text: str, prefix: str) -> str:
    json_property_name = f"{prefix}{target_language}"
    if json_property_name not in data:
        check_result = await check_translation(
            source_language=source_language,
            target_language=target_language,
            source_text=data[source_language],
            target_text=target_text
        )
        data[json_property_name] = check_result
    else:
        check_result = data[json_property_name]
    return check_result


async def crawl_json(data, source_language_1: str, source_language_2: str, target_language: str, current_translations: dict, table_lines: list, path="", official="", second_compare_language=""):
    if isinstance(data, dict):
        for key, value in data.items():
            new_path = f"{path}.{key}" if path else key
            await crawl_json(
                data=value,
                path=new_path,
                source_language_1=source_language_1,
                source_language_2=source_language_2,
                target_language=target_language,
                current_translations=current_translations,
                table_lines=table_lines,
                official=official,
                second_compare_language=second_compare_language,
            )
        print(current_translations)
        if len(current_translations.keys()) > 0:
            translation_1 = await create_translation_property(data=data, source_language=source_language_1, target_language=target_language, prefix=f"_{source_language_1}_")
            translation_2 = await create_translation_property(data=data, source_language=source_language_2, target_language=target_language, prefix=f"_{source_language_2}_")
            check_result_1 = await check_translation_property(data=data, source_language=source_language_1, target_language=target_language, target_text=translation_2, prefix=f"_check_{source_language_1}_")
            check_result_2 = await check_translation_property(data=data, source_language=source_language_2, target_language=target_language, target_text=translation_1, prefix=f"_check_{source_language_2}_")

            table_lines.append({
                "id": path,
                "source_text_1": data[source_language_1],
                "source_text_2": data[source_language_2],
                "translation_1": create_diff_html(source=translation_1, target=translation_2),
                "translation_2": create_diff_html(source=translation_2, target=translation_1),
                "check_result_1": check_result_1,
                "check_result_2": check_result_2,
            })
            current_translations.clear()
    elif isinstance(data, list):
        for i, value in enumerate(data):
            new_path = f"{path}[{i}]"
            await crawl_json(
                data=value,
                path=new_path,
                source_language_1=source_language_1,
                source_language_2=source_language_2,
                target_language=target_language,
                current_translations=current_translations,
                table_lines=table_lines,
                official=official,
                second_compare_language=second_compare_language,
            )
    else:
        if is_translation(path):
            language = path.split(".")[-1]
            current_translations[language] = data
        print(f"{path}: {data}")


async def process_i18n_file(file_path: str, source_language_1: str, source_language_2: str, target_language="") -> list:
    with open(file_path) as f:
        data = json.load(f)

    table_lines = []
    await crawl_json(data, source_language_1=source_language_1, source_language_2=source_language_2, target_language=target_language, current_translations={}, table_lines=table_lines)
    with open(file_path, "w", encoding="UTF-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    return table_lines


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


def create_table(input_file: str, translation_lines: [str], source_language_1: str, source_language_2: str, target_language: str):
    headers = [
        f"source text {source_language_1}",
        f"source text {source_language_2}",
        f"translation {source_language_1}",
        f"translation {source_language_2}",
        f"check (translation {source_language_1} vs. {source_language_2})",
        f"check (translation {source_language_2} vs. {source_language_1})",
    ]

    env = Environment(loader=FileSystemLoader('../templates'))
    template = env.get_template('simple_table_min_2langcompare.html.j2')
    html = template.render(headers=headers, translation_lines=translation_lines)
    with open(f"../tables/{input_file}_table.{source_language_1}_{source_language_2}_{target_language}.html", 'w') as f:
        f.write(html)

async def main():
    table_lines = await process_i18n_file(file_path=INPUT_FILE, source_language_1=SOURCE_LANGUAGE, source_language_2="en", target_language=TARGET_LANGUAGE)
    print(table_lines)
    create_table(translation_lines=table_lines, source_language_1=SOURCE_LANGUAGE, source_language_2="en", target_language=TARGET_LANGUAGE, input_file=os.path.basename(INPUT_FILE))


if __name__ == "__main__":
    asyncio.run(main())
