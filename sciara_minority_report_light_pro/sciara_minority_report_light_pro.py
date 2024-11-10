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

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm

from xhtml2pdf import pisa

os.environ['DYLD_FALLBACK_LIBRARY_PATH'] = "/opt/homebrew/lib:" + os.environ.get('DYLD_FALLBACK_LIBRARY_PATH', '')

from weasyprint import HTML

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
INPUT_FILE = os.getenv("INPUT_FILE", None)
INPUT_FILE_SET = os.getenv("INPUT_FILE_SET", None)
INPUT_FILE_ROOT_FOLDER = os.getenv("INPUT_FILE_ROOT_FOLDER", os.getcwd())
INPUT_FILE_DESCRIPTION = os.getenv("INPUT_FILE_DESCRIPTION", None)
RETRY_CHECK = (os.getenv("RETRY_CHECK", "0") == "1")
TARGET_LANGUAGE = os.getenv("TARGET_LANGUAGE", "")
SOURCE_LANGUAGE = os.getenv("SOURCE_LANGUAGE", "de")
TRANSLATION_MODELS = os.getenv("TRANSLATION_MODELS", ','.join([GPT_4O, GEMINI_1_5_PRO]))
TRANSLATION_ASSESS_MODELS = os.getenv("TRANSLATION_ASSESS_MODELS", ','.join([GPT_4O, GEMINI_1_5_PRO, CLAUDE_3_5_SONNET]))
BACK_TRANSLATION_MODELS = os.getenv("BACK_TRANSLATION_MODELS", ','.join([GPT_4O, GEMINI_1_5_PRO, CLAUDE_3_5_SONNET]))
BACK_TRANSLATION_ASSESS_MODELS = os.getenv("BACK_TRANSLATION_ASSESS_MODELS", ','.join([GPT_4O, GEMINI_1_5_PRO, CLAUDE_3_5_SONNET]))


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
    unique_translations = create_unique_translations(translations)
    best_translations = [
        await compare_translations(
            source_text=source_text,
            source_language=source_language,
            target_language=target_language,
            translations=unique_translations,
            model=model,
            file_content=file_content,
            file_description=file_description
        ) for model in assess_models
    ] if len(unique_translations) > 1 and len(assess_models) > 1 else unique_translations
    best_translation = normalize_string(find_most_common_strings(best_translations)[0])
    return best_translation


def is_translation(path: str) -> bool:
    return path.split(".")[-1].lower() in LANGUAGES and not "imageRef" in path


def create_pdf_with_table(input_file: str, translation_lines: [str], source: str, target: str):
    pdf_filename = f"./tables/{input_file}_table.{source}_{target}.light.pro.pdf"
    doc = SimpleDocTemplate(pdf_filename, pagesize=A4)

    # Berechne die verfügbare Breite (A4-Breite minus Ränder)
    available_width = A4[0] - 2 * cm

    # Definiere Stile
    styles = getSampleStyleSheet()
    html_style = ParagraphStyle('HTMLStyle', parent=styles['Normal'])
    html_style.wordWrap = 'CJK'  # Ermöglicht Umbruch für lange Wörter
    html_style.allowWidows = 0
    html_style.allowOrphans = 0
    styleN = styles['Normal']

    translation_headers = []
    model_names = TRANSLATION_MODELS.split(",")
    for i in range(len(model_names)):
        translation_headers.append(Paragraph(text=f"--- {i} ---", style=html_style))
        translation_headers.append(Paragraph(text="", style=styleN))

    assess_headers = [Paragraph(text=model_name, style=html_style) for model_name in TRANSLATION_ASSESS_MODELS.split(",")]

    formatted_data = [["Source text", *translation_headers, *assess_headers]]
    for translation_line in translation_lines:
        formatted_row = [Paragraph(text=translation_line["source_text"], style=html_style)]
        for i in range(len(translation_line["translations"])):
            formatted_row.append(Paragraph(text=translation_line["translations"][i], style=html_style))
            formatted_row.append(Paragraph(text=translation_line["translations_back"][i], style=html_style))
        formatted_data.append(formatted_row)

    col_widths = [available_width / len(formatted_data[0])] * len(formatted_data[0])
    table = Table(formatted_data, colWidths=col_widths)

    style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ])

    table.setStyle(style)

    elements = [table]
    doc.build(elements)


def create_pdf_table(input_file: str, translation_lines: [str], source: str, target: str):
    pdf_filename = f"./tables/{input_file}_table.{source}_{target}.light.pro.pdf"
    pdf = SimpleDocTemplate(
        pdf_filename,
        pagesize=letter
    )

    styles = getSampleStyleSheet()
    styleN = styles['Normal']

    translation_headers = []
    model_names = TRANSLATION_MODELS.split(",")
    for i in range(len(model_names)):
        translation_headers.append(Paragraph(text=f"--- {i} ---", style=styleN))
        translation_headers.append(Paragraph(text="", style=styleN))

    assess_headers = [Paragraph(text=model_name, style=styleN) for model_name in TRANSLATION_ASSESS_MODELS.split(",")]

    data_lines = [[*translation_headers, *assess_headers]]
    for translation_line in translation_lines:
        data_line = [Paragraph(text=translation_line["source_text"], style=styleN)]
        for i in range(len(translation_line["translations"])):
            data_line.append(Paragraph(text=translation_line["translations"][i], style=styleN))
            data_line.append(Paragraph(text=translation_line["translations_back"][i], style=styleN))
        data_lines.append(data_line)

    table = Table(data_lines)

    # Add style to the table
    style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ])
    table.setStyle(style)

    # Build the PDF
    table_width = pdf.pagesize[0] - pdf.leftMargin - pdf.rightMargin
    num_columns = len(model_names)*3 + 1
    col_width = table_width / num_columns
    table._argW = [col_width] * num_columns
    elements = [table]
    pdf.build(elements)


def create_table(input_file: str, translation_lines: [str], source: str, target: str) -> str:
    translation_headers = []
    model_names = TRANSLATION_MODELS.split(",")
    for i in range(len(model_names)):
        translation_headers.append(f"--- {i} ---")
        translation_headers.append("")

    assess_headers = [model_name for model_name in TRANSLATION_ASSESS_MODELS.split(",")]

    headers = ["source text", *translation_headers, *assess_headers]

    env = Environment(loader=FileSystemLoader('../templates'))
    template = env.get_template('compare_light_table.html.j2')
    html = template.render(table_name=input_file, headers=headers, translation_lines=translation_lines, max_translations=len(TRANSLATION_MODELS.split(",")))
    html_file = f"../tables/{input_file}_table.{source}_{target}.light.pro.html"
    with open(html_file, 'w') as f:
        f.write(html)
    #HTML(html _file).write_pdf(f"{html_file}.pdf")
    # pdf_file = f"../tables/{input_file}_table.{source}_{target}.light.pro.html.pdf"
    # with open(pdf_file, "w+b") as pdf_file:
    #     pisa.CreatePDF(html, dest=pdf_file)
    return(html_file)


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


async def translate(
        data: dict,
        property_name: str,
        model_name: str,
        source_language: str,
        target_language: str,
        file_content: str,
        file_description: str,
) -> str:
    if not property_name in data:
        translation = await get_translation(
            source_text=data[source_language],
            source_language=source_language,
            target_language=target_language,
            model=model_name,
            file_content=file_content,
            file_description=file_description,
        )
        data[property_name] = translation
    else:
        translation = data[property_name]
    return translation


async def compare(
        data: dict,
        property_name: str,
        model_name: str,
        source_language: str,
        target_language: str,
        file_content: str,
        file_description: str,
        translations: [str]
) -> str:
    if not property_name in data:
        compare_result = await compare_translations(
            source_text=data[source_language],
            source_language=source_language,
            translations=create_unique_translations(translations),
            target_language=target_language,
            model=model_name,
            file_content=file_content,
            file_description=file_description,
        )
        data[property_name] = compare_result
    else:
        compare_result = data[property_name]
    return compare_result


def check_if_array_contains_no_empty_strings(array: [str]) -> bool:
    return reduce(lambda acc, element: acc and len(element) > 0, array, True)


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

        model_names = TRANSLATION_MODELS.split(",")
        assess_model_names = TRANSLATION_ASSESS_MODELS.split(",")

        if len(current_translations.keys()) > 0:
            translation_list = [
                await translate(
                    data=data,
                    property_name=f"_{model_name}_{target_language}",
                    model_name=model_name,
                    source_language=source_language,
                    target_language=target_language,
                    file_content=file_content,
                    file_description=file_description
                ) for model_name in model_names
            ]

            compare_list = [
                await compare(
                    data=data,
                    property_name=f"_compare_{model_name}_{target_language}",
                    model_name=model_name,
                    source_language=source_language,
                    target_language=target_language,
                    file_content=file_content,
                    file_description=file_description,
                    translations=translation_list,
                ) for model_name in assess_model_names
            ]

            winners = find_most_common_strings(compare_list)
            unique_translations = create_unique_translations(translation_list)

            def get_translation_sources(translation: str) -> [str]:
                sources = []
                model_names = TRANSLATION_MODELS.split(",")
                for i, model_name in enumerate(model_names):
                    if normalize_string(translation_list[i]) == normalize_string(translation):
                        sources.append(model_name)
                return ', '.join(sources)

            translation_sources = [get_translation_sources(unique_translation) for unique_translation in unique_translations]

            unique_translations_back_property = f"_unique_translations_back_{target_language}"
            if not (unique_translations_back_property in data and check_if_array_contains_no_empty_strings(data[unique_translations_back_property])):
                unique_translations_back = [
                    await get_best_back_translation(
                        source_text=translation,
                        source_language=target_language,
                        target_language=source_language,
                        translate_models=[GPT_4O],
                        assess_models=[],
                        file_content=file_content,
                        file_description=file_description,
                    ) for translation in unique_translations
                ]
                data[unique_translations_back_property] = unique_translations_back
            else:
                unique_translations_back = data[unique_translations_back_property]

            if len(unique_translations) > 1 and len(unique_translations) < 3:
                translations = [create_diff_html(unique_translations[1], unique_translations[0]), create_diff_html(unique_translations[0], unique_translations[1])]
            else:
                translations = unique_translations

            table_lines.append({
                "id": path,
                "source_text": current_translations[source_language],
                "translations": translations,
                "translations_raw": unique_translations,
                "translations_back": [create_diff_html(normalize_string(current_translations[source_language]), normalize_string(unique_translation_back)) if len(unique_translation_back) > 0 else "" for unique_translation_back in unique_translations_back],
                "compare_list": compare_list,
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
        if not _target_language in CERTIFIED_LANGUAGES:
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


async def process_one_file(input_file: str, file_description: str) -> ([str], [str]):
    table_lines_dict = await process_i18n_file(file_path=input_file, file_description=file_description, target_language=TARGET_LANGUAGE)
    html_tables = []
    pdf_tables = []
    for target_language in table_lines_dict:
        html_table = create_table(translation_lines=table_lines_dict[target_language], source=SOURCE_LANGUAGE, target=target_language, input_file=os.path.basename(input_file))
        html_tables.append(html_table)
        # pdf_table = create_pdf_with_table(translation_lines=table_lines_dict[target_language], source=SOURCE_LANGUAGE, target=target_language, input_file=os.path.basename(input_file))
        # pdf_tables.append(pdf_table)
    return html_tables, pdf_tables


async def process_input_file_set(file_set_path: str) -> ([str], [str]):
    all_html_tables = []
    all_pdf_tables =[]
    with open(file_set_path, 'r') as f:
        for line in f:
            if len(line.strip()) > 0:
                print(f"==========> {line.strip()}")
                relative_file_path, file_description = line.strip().split("///")
                html_tables, pdf_tables = await process_one_file(input_file=os.path.join(INPUT_FILE_ROOT_FOLDER, relative_file_path), file_description=file_description)
                all_html_tables.extend(html_tables)
                all_pdf_tables.extend(pdf_tables)
    return all_html_tables, all_pdf_tables


def create_all_pdf_file(html_files: [str]):
    if len(html_files) > 0:
        html_objects = [HTML(filename=f) for f in html_files]
        combined_doc = html_objects[0].render()
        for html in html_objects[1:]:
            combined_doc.pages.extend(html.render().pages)
        with open(f"./tables/table.{SOURCE_LANGUAGE}_{TARGET_LANGUAGE if len(TARGET_LANGUAGE) > 0 else 'all'}.light.pro.html", 'wb') as f:
            combined_doc.write_pdf(f)


async def main():
    if INPUT_FILE_SET:
        await process_input_file_set(file_set_path=INPUT_FILE_SET)
    elif INPUT_FILE and INPUT_FILE_DESCRIPTION:
        await process_one_file(input_file=INPUT_FILE, file_description=INPUT_FILE_DESCRIPTION)
    else:
        raise RuntimeError("no environment variable set")


if __name__ == "__main__":
    asyncio.run(main())
