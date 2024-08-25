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
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "azure/mistral-large")
LLAMA3_MODEL = os.getenv("LLAMA3_MODEL", "iteratec/Llama3.1-70B-Instruct")
INPUT_FILE = os.getenv("INPUT_FILE")
RETRY_CHECK = (os.getenv("RETRY_CHECK", "0") == "1")
TARGET_LANGUAGE = os.getenv("TARGET_LANGUAGE", "")
SOURCE_LANGUAGE = os.getenv("SOURCE_LANGUAGE", "en")
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
        return ask_model(system, user, model)
    except Exception as exc:
        print(f"Exception: {exc}")
        return ""
    return answer


async def get_translation(source_text: str, source_language: str, target_language: str, model: str):
    system = (
        f"You are an expert in all languages and climate change. In the following you get an original {LANGUAGES[source_language]} text."
        f"Translate the original {LANGUAGES[source_language]} text into {LANGUAGES[target_language]}. Ensure that the translated text retains the original meaning, tone, and intent."
        f"The answer has to contain ONLY the translation itself. No explaining text. Otherwise the answer is NOT CORRECT"
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
        f"You are an expert in all languages and climate change. In the following you get an original {LANGUAGES[source_language]} text and a translation in {LANGUAGES[target_language]}."
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


async def compare_translations(source_text: str, source_language: str, translations: [str], target_language: str, model: str):
    system = (
        f"You are an expert in all languages and climate change. In the following you get an original {LANGUAGES[source_language]} text and {len(translations)} translations in {LANGUAGES[target_language]}."
        "Please decide which translation is the most accurate. Just answer just with the number of the translation. Please explain your decision in just one sentence."
    )
    original_line = [f"Original {LANGUAGES[source_language]}: \"{source_text}\""]
    translation_lines = [f"translation {index+1}: \"{translation}\"" for index, translation in enumerate(translations)]
    user = "\n".join(original_line + translation_lines)
    print(f"compare_translations for '{source_text}'")
    answer = await ask_model(system=system, user=user, model=model)
    return answer


def is_translation(path: str) -> bool:
    return path.split(".")[-1].lower() in LANGUAGES and not "imageRef" in path


def create_table(input_file: str, translation_lines: [str], source: str, target: str):
    headers = [
        "source text",
        "OpenAI (1)",
        "Gemini (2)",
        "Claude (3)",
        "Mistral (4)",
        "Llama3 (5)",
        "Compare via OpenAI",
        "Compare via Gemini",
        "Compare via Claude",
        "Compare via Mistral",
        "Compare via Llama3",
    ]

    env = Environment(loader=FileSystemLoader('./templates'))
    template = env.get_template('simple_table_min.html.j2')
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


async def crawl_json(data, source_language: str, target_language: str, current_translations: dict, table_lines: list, path="", official=""):
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
                official=official,
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
                )
                data[target_language_claude_property] = target_translation_claude
            else:
                target_translation_claude = data[target_language_claude_property]

            target_language_mistral_property = f"_mistral_{target_language}"
            if not target_language_mistral_property in data:
                target_translation_mistral = await get_translation(
                    source_text=data[source_language],
                    source_language=source_language,
                    target_language=target_language,
                    model=MISTRAL_MODEL,
                )
                data[target_language_mistral_property] = target_translation_mistral
            else:
                target_translation_mistral = data[target_language_mistral_property]

            target_language_llama3_property = f"_llama3_{target_language}"
            if not target_language_llama3_property in data:
                target_translation_llama3 = await get_translation(
                    source_text=data[source_language],
                    source_language=source_language,
                    target_language=target_language,
                    model=LLAMA3_MODEL,
                )
                data[target_language_llama3_property] = target_translation_llama3
            else:
                target_translation_llama3 = data[target_language_llama3_property]

            compare_result_openai_property = f"_compare_openai_{target_language}"
            if not compare_result_openai_property in data:
                compare_result_openai = await compare_translations(
                    source_text=data[source_language],
                    source_language=source_language,
                    translations=[target_translation_openai, target_translation_gemini, target_translation_claude, target_translation_mistral, target_translation_llama3],
                    target_language=target_language,
                    model=OPENAI_MODEL,
                )
                data[compare_result_openai_property] = compare_result_openai
            else:
                compare_result_openai = data[compare_result_openai_property]

            compare_result_gemini_property = f"_compare_gemini_{target_language}"
            if not compare_result_gemini_property in data:
                compare_result_gemini = await compare_translations(
                    source_text=data[source_language],
                    source_language=source_language,
                    translations=[target_translation_openai, target_translation_gemini, target_translation_claude, target_translation_mistral, target_translation_llama3],
                    target_language=target_language,
                    model=GEMINI_MODEL,
                )
                data[compare_result_gemini_property] = compare_result_gemini
            else:
                compare_result_gemini = data[compare_result_gemini_property]

            compare_result_claude_property = f"_compare_claude_{target_language}"
            if not compare_result_claude_property in data:
                compare_result_claude = await compare_translations(
                    source_text=data[source_language],
                    source_language=source_language,
                    translations=[target_translation_openai, target_translation_gemini, target_translation_claude, target_translation_mistral, target_translation_llama3],
                    target_language=target_language,
                    model=CLAUDE_MODEL,
                )
                data[compare_result_claude_property] = compare_result_claude
            else:
                compare_result_claude = data[compare_result_claude_property]

            compare_result_mistral_property = f"_compare_mistral_{target_language}"
            if not compare_result_mistral_property in data:
                compare_result_mistral = await compare_translations(
                    source_text=data[source_language],
                    source_language=source_language,
                    translations=[target_translation_openai, target_translation_gemini, target_translation_claude, target_translation_mistral, target_translation_llama3],
                    target_language=target_language,
                    model=MISTRAL_MODEL,
                )
                data[compare_result_mistral_property] = compare_result_mistral
            else:
                compare_result_mistral = data[compare_result_mistral_property]

            compare_result_llama3_property = f"_compare_llama3_{target_language}"
            if not compare_result_llama3_property in data:
                compare_result_llama3 = await compare_translations(
                    source_text=data[source_language],
                    source_language=source_language,
                    translations=[target_translation_openai, target_translation_gemini, target_translation_claude, target_translation_mistral, target_translation_llama3],
                    target_language=target_language,
                    model=LLAMA3_MODEL,
                )
                data[compare_result_llama3_property] = compare_result_llama3
            else:
                compare_result_llama3 = data[compare_result_llama3_property]

            table_lines.append({
                "id": path,
                "source_text": current_translations[source_language],
                "translation_openai": create_diff_html(target=target_translation_gemini, source=target_translation_openai),
                "translation_gemini": create_diff_html(target=target_translation_openai, source=target_translation_gemini),
                "translation_claude": create_diff_html(target=target_translation_openai, source=target_translation_claude),
                "translation_mistral": create_diff_html(target=target_translation_openai, source=target_translation_mistral),
                "translation_llama3": create_diff_html(target=target_translation_openai, source=target_translation_llama3),
                "compare_openai": compare_result_openai,
                "compare_gemini": compare_result_gemini,
                "compare_claude": compare_result_claude,
                "compare_mistral": compare_result_mistral,
                "compare_llama3": compare_result_llama3,
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
                official=official,
            )
    else:
        if is_translation(path):
            language = path.split(".")[-1]
            current_translations[language] = data
        print(f"{path}: {data}")


async def process_i18n_file(file_path: str, target_language="") -> {str: [object]}:
    with open(file_path) as f:
        data = json.load(f)

    table_lines_dict = {}
    for _target_language in (LANGUAGES.keys() if len(target_language) == 0 else [target_language]):
        table_lines = []
        await crawl_json(data, source_language=SOURCE_LANGUAGE, target_language=_target_language, current_translations={}, table_lines=table_lines)
        with open(file_path, "w", encoding="UTF-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        table_lines_dict[_target_language] = table_lines
    return table_lines_dict

async def main():
    table_lines_dict = await process_i18n_file(INPUT_FILE, TARGET_LANGUAGE)
    for target_language in table_lines_dict:
        create_table(translation_lines=table_lines_dict[target_language], source=SOURCE_LANGUAGE, target=target_language, input_file=os.path.basename(INPUT_FILE))


if __name__ == "__main__":
    asyncio.run(main())
