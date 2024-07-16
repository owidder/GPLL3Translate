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
# from mlx_lm import load, generate
# model, tokenizer = load("mlx-community/Meta-Llama-3-8B-Instruct-4bit")
import google.generativeai as genai
from bs4 import BeautifulSoup

#model, tokenizer = load("mlx-community/Llama-3-8B-Instruct-1048k-8bit")
#model, tokenizer = load("mlx-community/Meta-Llama-3-8B-8bit")

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
INPUT_FILE = os.getenv("INPUT_FILE")
RETRY_CHECK = (os.getenv("RETRY_CHECK", "0") == "1")
TARGET_LANGUAGE = os.getenv("TARGET_LANGUAGE", "")
SOURCE_LANGUAGE = os.getenv("SOURCE_LANGUAGE", "en")
WITH_CHECKS_FOR_CERTIFIED_LANGUAGES = bool(os.getenv("WITH_CHECKS_FOR_CERTIFIED_LANGUAGES", False))
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
    except openai.RateLimitError as e:
        print(e.json_body["error"]["message"])
        secnum_re = re.match(".*Please try again in ([0-9.]*)s\..*", e.json_body["error"]["message"])
        if secnum_re:
            secnum = float(secnum_re.groups()[0])
        else:
            secnum = 1
        print(f"wait for {secnum}")
        time.sleep(secnum)
        return ask_chatgpt(system, user, model)
    except Exception as exc:
        print(f"Exception: {exc}")
        return ""
    return answer


# def ask_llama3(system: str, user: str) -> str:
#     messages = [ {"role": "system", "content": system}, {"role": "user", "content": user}, ]
#     input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
#     prompt = tokenizer.decode(input_ids)
#     response = generate(model, tokenizer, prompt=prompt)
#     return response


def ask_gemini(system: str, user: str, model: str):
    model = genai.GenerativeModel(model_name=model, system_instruction=system)
    response = model.generate_content(user)
    return response.text


async def get_translation(source_text: str, source_language: str, target_language: str, translations: [(str, str)], llm="gpt"):
    system = (
        f"You are an expert in all languages and climate change. In the following you get an original {LANGUAGES[source_language]} text {'and several translations into other languages' if len(translations) > 0 else ''}."
        f"Translate the original {LANGUAGES[source_language]} text into {LANGUAGES[target_language]}. Ensure that the translated text retains the original meaning, tone, and intent."
        f"The answer has to contain ONLY the translation itself. No explaining text. Otherwise the answer is NOT CORRECT"
    )
    user_lines = [f"Original {LANGUAGES[source_language]}: \"{source_text}\""]
    if len(translations) > 0:
        translations_filtered = [language_pair for language_pair in translations if language_pair[0] not in [source_language, target_language]]
        user_lines.extend([f"{LANGUAGES[language]} Translation: \"{translation}\"" for language, translation in translations_filtered])
    user = "\n".join(user_lines)
    print(f"get_translation for '{source_text}'")
    if llm == "gpt":
        answer = await ask_chatgpt(system, user, model="gpt-4")
    # elif llm == "llama3":
    #     answer = ask_llama3(system, user)
    elif llm == "gemini":
        answer = ask_gemini(system, user, model='gemini-1.5-flash')
    else:
        raise Exception(f"unknown llm: {llm}")
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
        answer = await ask_chatgpt(system, user, model="gpt-4o")
    elif llm == "gemini":
        answer = ask_gemini(system, user, model='gemini-1.5-flash')
    else:
        raise Exception(f"unknown llm: {llm}")
    print(f"check_translation with {llm}: answer={answer}")
    return answer


def is_translation(path: str) -> bool:
    return path.split(".")[-1].lower() in LANGUAGES and not "imageRef" in path


def create_table(input_file: str, translation_lines: [str], source: str, target: str):
    _headers = [
        "source text",
        "GPT4",
        "check (GPT4 via GPT4)",
        "check (GPT4 via Gemini)",
        "Gemini",
        "check (Gemini via GPT4)",
        "check (Gemini via Gemini)",
    ]
    headers = list(
        reduce(
            lambda acc, item: [*acc, f"check: {item}"],
            filter(lambda l: l != source and l != target, CERTIFIED_LANGUAGES),
            _headers
        )
    ) if WITH_CHECKS_FOR_CERTIFIED_LANGUAGES else _headers

    env = Environment(loader=FileSystemLoader('./templates'))
    template = env.get_template('simple_table_min.html.j2')
    html = template.render(headers=headers, translation_lines=translation_lines)
    with open(f"./tables/{input_file}_table.{source}_{target}.html", 'w') as f:
        f.write(html)


def create_diff_text(target: str, source: str, plusminus) -> str:
    diff_list = list(difflib.ndiff(a=target.split(), b=source.split()))
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
                table_lines=table_lines
            )
        print(current_translations)
        translations = [(language, current_translations.get(language)) for language in CERTIFIED_LANGUAGES]
        filtered_translations = list(filter(lambda l: l[1] is not None and l[0] != source_language and l[0] != target_language, translations))
        if len(current_translations.keys()) > 0:
            # if not target_language in data:
            #     target_translation = await get_translation(
            #         source_text=current_translations[source_language],
            #         source_language=source_language,
            #         target_language=target_language,
            #         translations=filtered_translations
            #     )
            #     data[target_language] = target_translation
            # else:
            #     target_translation = data[target_language]

            # target_language_w_property = f"_w_{target_language}"
            # if target_language in CERTIFIED_LANGUAGES:
            #     if REDO_CHATGPT or (len(filtered_translations) > 0 and not target_language_w_property in data):
            #         target_translation_w = await get_translation(
            #             source_text=data[source_language],
            #             source_language=source_language,
            #             target_language=target_language,
            #             translations=filtered_translations
            #         )
            #         data[target_language_w_property] = target_translation_w
            #     else:
            #         target_translation_w = data[target_language_w_property]
            # else:
            #     target_translation_w = ""

            target_language_wo_property = f"_wo_{target_language}"
            if REDO_CHATGPT or (len(filtered_translations) > 0 and not target_language_wo_property in data):
                target_translation_wo = await get_translation(
                    source_text=data[source_language],
                    source_language=source_language,
                    target_language=target_language,
                    translations=[]
                )
                data[target_language_wo_property] = target_translation_wo
            else:
                target_translation_wo = data[target_language_wo_property]

            # target_translation_ll3 = ""
            # if len(data[source_language]) < 200:
            #     target_language_ll3_property = f"_ll3_{target_language}"
            #     if len(filtered_translations) > 0 and not target_language_ll3_property in data:
            #         target_translation_ll3 = await get_translation(
            #             source_text=data[source_language],
            #             source_language=source_language,
            #             target_language=target_language,
            #             translations=filtered_translations,
            #             llm='llama3'
            #         )
            #         data[target_language_ll3_property] = target_translation_ll3
            #     else:
            #         target_translation_ll3 = data[target_language_ll3_property]

            # target_translation_ll3_wo = ""
            # target_language_ll3_wo_property = f"_ll3_wo_{target_language}"
            # if len(filtered_translations) > 0 and not target_language_ll3_wo_property in data:
            #     target_translation_ll3_wo = await get_translation(
            #         source_text=data[source_language],
            #         source_language=source_language,
            #         target_language=target_language,
            #         translations=[],
            #         llm='llama3'
            #     )
            #     data[target_language_ll3_wo_property] = target_translation_ll3_wo
            # else:
            #     target_translation_ll3_wo = data[target_language_ll3_wo_property]

            target_language_gemini_wo_property = f"_gemini_wo_{target_language}"
            if len(filtered_translations) > 0 and not target_language_gemini_wo_property in data:
                target_translation_gemini_wo = await get_translation(
                    source_text=data[source_language],
                    source_language=source_language,
                    target_language=target_language,
                    translations=[],
                    llm='gemini'
                )
                data[target_language_gemini_wo_property] = target_translation_gemini_wo
            else:
                target_translation_gemini_wo = data[target_language_gemini_wo_property]


            # back_property = f"_back_{target_language}"
            # if not back_property in data:
            #     back_translation = get_translation(
            #         source_text=target_translation,
            #         source_language=target_language,
            #         target_language=source_language,
            #         translations=[]
            #     )
            #     data[back_property] = back_translation
            # else:
            #     back_translation = data[back_property]

            # check_property = f"_check_{target_language}"
            # if REDO_CHATGPT or ((not check_property in data) or (RETRY_CHECK and data[check_property].lower() != "yes")):
            #     check_result = await check_translation(
            #         source_language=source_language,
            #         target_language=target_language,
            #         source_text=current_translations[source_language],
            #         target_text=target_translation_w if len(target_translation_w) > 0 else target_translation,
            #     )
            #     data[check_property] = check_result
            # else:
            #     check_result = data[f"_check_{target_language}"]

            check_wo_property = f"_check_wo_{target_language}"
            if REDO_CHATGPT or (len(filtered_translations) > 0 and ((not check_wo_property in data) or (RETRY_CHECK and data[check_wo_property].lower() != "yes"))):
                check_wo_result = await check_translation(
                    source_language=source_language,
                    target_language=target_language,
                    source_text=current_translations[source_language],
                    target_text=target_translation_wo
                )
                data[check_wo_property] = check_wo_result
            else:
                check_wo_result = data[check_wo_property]

            check_wo_via_gemini_property = f"_check_wo_via_gemini_{target_language}"
            if REDO_CHATGPT or (len(filtered_translations) > 0 and ((not check_wo_via_gemini_property in data) or (RETRY_CHECK and data[check_wo_via_gemini_property].lower() != "yes"))):
                check_wo_via_gemini_result = await check_translation(
                    source_language=source_language,
                    target_language=target_language,
                    source_text=current_translations[source_language],
                    target_text=target_translation_wo,
                    llm='gemini'
                )
                data[check_wo_via_gemini_property] = check_wo_via_gemini_result
            else:
                check_wo_via_gemini_result = data[check_wo_via_gemini_property]

            check_gemini_wo_result = ""
            if len(target_translation_gemini_wo) > 0:
                check_gemini_wo_property = f"_check_gemini_wo_{target_language}"
                if len(filtered_translations) > 0 and ((not check_gemini_wo_property in data) or (RETRY_CHECK and data[check_gemini_wo_property].lower() != "yes")):
                    check_gemini_wo_result = await check_translation(
                        source_language=source_language,
                        target_language=target_language,
                        source_text=current_translations[source_language],
                        target_text=target_translation_gemini_wo
                    )
                    data[check_gemini_wo_property] = check_gemini_wo_result
                else:
                    check_gemini_wo_result = data[check_gemini_wo_property]

            check_gemini_wo_via_gemini_result = ""
            if len(target_translation_gemini_wo) > 0:
                check_gemini_wo_via_gemini_property = f"_check_gemini_wo_via_gemini_{target_language}"
                if len(filtered_translations) > 0 and ((not check_gemini_wo_via_gemini_property in data) or (RETRY_CHECK and data[check_gemini_wo_via_gemini_property].lower() != "yes")):
                    check_gemini_wo_via_gemini_result = await check_translation(
                        source_language=source_language,
                        target_language=target_language,
                        source_text=current_translations[source_language],
                        target_text=target_translation_gemini_wo,
                        llm='gemini'
                    )
                    data[check_gemini_wo_via_gemini_property] = check_gemini_wo_via_gemini_result
                else:
                    check_gemini_wo_via_gemini_result = data[check_gemini_wo_via_gemini_property]

            # check_ll3_result = ""
            # if len(target_translation_ll3) > 0:
            #     check_ll3_property = f"_check_ll3_{target_language}"
            #     if (not check_ll3_property in data) or (RETRY_CHECK and data[check_ll3_property].lower() != "yes"):
            #         check_ll3_result = await check_translation(
            #             source_language=source_language,
            #             target_language=target_language,
            #             source_text=current_translations[source_language],
            #             target_text=target_translation_ll3
            #         )
            #         data[check_ll3_property] = check_ll3_result
            #     else:
            #         check_ll3_result = data[check_ll3_property]
            #
            # check_ll3_wo_result = ""
            # if len(target_translation_ll3_wo) > 0:
            #     check_ll3_wo_property = f"_check_ll3_wo_{target_language}"
            #     if len(filtered_translations) > 0 and ((not check_ll3_wo_property in data) or (RETRY_CHECK and data[check_ll3_wo_property].lower() != "yes")):
            #         check_ll3_wo_result = await check_translation(
            #             source_language=source_language,
            #             target_language=target_language,
            #             source_text=current_translations[source_language],
            #             target_text=target_translation_ll3_wo
            #         )
            #         data[check_ll3_wo_property] = check_ll3_wo_result
            #     else:
            #         check_ll3_wo_result = data[check_ll3_wo_property]

            # certified_translation_checks = {}
            # if WITH_CHECKS_FOR_CERTIFIED_LANGUAGES:
            #     for check_language in filter(lambda l: l != source_language and l != target_language, CERTIFIED_LANGUAGES):
            #         if check_language in data:
            #             check_language_property = f"_check_{check_language}"
            #             if (not check_language_property in data) or (RETRY_CHECK and data[check_language_property].lower() != "yes"):
            #                 check_language_result = await check_translation(
            #                     source_language=source_language,
            #                     target_language=check_language,
            #                     source_text=data[source_language],
            #                     target_text=data[check_language]
            #                 )
            #                 data[check_language_property] = check_language_result
            #             certified_translation_checks[check_language] = f"<span style = 'color: gray; font-weight: bold;'>{data[check_language]}</span><br><br><br>{data[check_language_property]}"

            table_lines.append({
                "id": path,
                # "translation": create_diff_html(textA=target_translation_wo, textB=target_translation),
                # "translation_w": create_diff_html(textA=target_translation_wo, textB=target_translation_w) if len(target_translation_w) > 0 else "",
                # "translation_wo": create_diff_html(textA=target_translation_w if len(target_translation_w) > 0 else target_translation, textB=target_translation_wo),
                "translation_wo": create_diff_html(target=target_translation_gemini_wo, source=target_translation_wo),
                # "translation_ll3": create_diff_html(textA=target_translation_ll3_wo, textB=target_translation_ll3),
                # "translation_ll3_wo": create_diff_html(textA=target_translation_ll3, textB=target_translation_ll3_wo),
                "translation_gemini_wo": create_diff_html(target=target_translation_wo, source=target_translation_gemini_wo),
                "source_text": current_translations[source_language],
                "official": official,
                # "check": check_result,
                "check_wo": check_wo_result,
                "check_wo_via_gemini": check_wo_via_gemini_result,
                "check_gemini_wo": check_gemini_wo_result,
                "check_gemini_wo_via_gemini": check_gemini_wo_via_gemini_result,
                # "check_ll3": check_ll3_result,
                # "check_ll3_wo": check_ll3_wo_result,
                # "certified_translation_checks": certified_translation_checks,
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
                table_lines=table_lines
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
