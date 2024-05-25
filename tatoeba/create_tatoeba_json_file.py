import csv
import json
import os
import zipfile


def filter_tatoeba_file(source_filename: str, min_text_length: int, max_text_length: int, max_sentences=100):

    sentence_count = 0

    out_name = f"source_texts/{'.'.join(source_filename.split('.')[0:-1])}.{min_text_length}_{max_text_length}_{max_sentences}.json"

    text_dicts = []
    if os.path.exists(out_name):
        with open(out_name, 'r', encoding='utf-8') as file:
            if len(file.readlines()) > 0:
                text_dicts = json.load(file)

    existing_sentence_nos = map(lambda td: td['sentence_no'], text_dicts)

    with zipfile.ZipFile(source_filename, 'r') as zip:
        with zip.open('.'.join(os.path.basename(source_filename).split('.')[0:-1])) as source, open(out_name, 'w', encoding='utf-8') as target:
            reader = csv.DictReader(source.read().decode(), delimiter='\t', quotechar='|',
                                    fieldnames=["sentence_no", "language", "text", "author", "ts1", "ts2"])

            for index, row in enumerate(reader):
                t = row.get("text", "")
                l = len(t) if t else 0
                if l > 0 and not row['sentence_no'] in existing_sentence_nos and l >= min_text_length and l < max_text_length:
                    text_dict = {
                        "sentence_no": row["sentence_no"],
                        "source_language": row["language"],
                        "source_text": row["text"],
                    }
                    text_dicts.append(text_dict)
                    sentence_count += 1
                    print(f"{l}: {index} / {row['sentence_no']} / {sentence_count}")
                    if sentence_count >= max_sentences:
                        break

            target.write(json.dumps(text_dicts))


if __name__ == '__main__':
    min_text_length = os.getenv("MIN_TEXT_LENGTH", 0)
    max_text_length = os.getenv("MAX_TEXT_LENGTH", 30)
    max_sentences = os.getenv("MAX_SENTENCES", 50)
    filter_tatoeba_file(source_filename="./eng_sentences_detailed.tsv.zip", min_text_length=min_text_length, max_text_length=max_text_length, max_sentences=max_sentences)
