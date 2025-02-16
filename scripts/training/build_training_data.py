import json
import os
from tqdm import tqdm
from datasets import load_dataset
import re
from nltk import sent_tokenize
from transformers import AutoTokenizer
import random
import pickle


def format_sections(text):
    # 匹配 <section> 和 <sub-section> 的标签
    section_pattern = r"Section::::([^.:]+)"
    subsection_pattern = r"\.:([^.:]+)\."

    # 用正则找到所有的section和sub-section
    sections = re.split(section_pattern, text)
    sections = [ii for ii in sections if ii != ""]
    formatted_text = ""

    if len(sections) == 0:
        return None

    if len(sections[0].split(" ")) > 10:
        # 说明第一段是abstract
        content = sections[0]
        formatted_text += f"<abstract>{extract_first_last_sentences(content)}</abstract>"
        sections = sections[1:]
    elif len(sections) % 2 != 0:
        print("abstract < 10")
        # 说明第一段是abstract
        content = sections[0]
        formatted_text += f"<abstract>{extract_first_last_sentences(content)}</abstract>"
        sections = sections[1:]

    i = 0
    current_section = None
    while i < len(sections):
        # Section 标题
        section_title = sections[i].strip()
        section_content = sections[i + 1].strip()

        if i == 0:
            current_section = section_title
            formatted_text += f"<section: {section_title}>"
        if i != 0 and section_title != current_section:
            formatted_text += f"</section: {current_section}>"
            current_section = section_title
            formatted_text += f"<section: {section_title}>"

        # 匹配sub-section
        subsections = re.split(subsection_pattern, section_content)
        subsections = [ss for ss in subsections if ss != ""]
        if len(subsections) > 1:
            # 存在subsection
            sub_title = subsections[0].strip()
            sub_content = subsections[1].strip()
            formatted_text += f"<sub-section: {sub_title}>"
            formatted_text += extract_first_last_sentences(sub_content)
            formatted_text += f"</sub-section: {sub_title}>"
        else:
            # 不存在subsection
            formatted_text += extract_first_last_sentences(section_content)

        i += 2
    if current_section is not None:
        formatted_text += f"</section: {current_section}>"
    return formatted_text


# 提取第一句和最后一句
def extract_first_last_sentences(content):
    def get_token(text, num=12):
        tokens = tokenizer(text, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokens) <= 2 * num:
            return text

        # 获取前五个单词和后五个单词
        first = tokenizer.decode(tokens[:num], skip_special_tokens=True)
        last = tokenizer.decode(tokens[-num:], skip_special_tokens=True)
        new_text = f"{first}...{last}"

        return new_text

    def get_sent(text):
        sentences = sent_tokenize(text)
        if len(sentences) > 2:
            return f"{sentences[0].strip()} <...> {sentences[-1].strip()}"
        elif len(sentences) == 2:
            return f"{sentences[0].strip()} {sentences[1].strip()}"
        else:
            return sentences[0].strip() if sentences else ""

    content = content.strip("\n").strip()
    if content[0] == ".":
        content = content[1:]
    all_para = content.split("<new_tag>")
    all_para = [i.strip() for i in all_para if i.strip() != ""]
    final_str = ""
    temp = ""
    idx = 0
    for para in all_para:
        if len(para.split(" ")) < 20:
            temp = temp + " " + para if temp != "" else para
            continue
        if temp != "":
            para = temp + " " + para
            temp = ""
        final_str += get_token(para, num=8)
        final_str += "<br>"
        idx += 1
    final_str = final_str.strip("<br>")

    return final_str


def build_single_data(retrieval_data):
    all_title = []
    for query, docs in list(retrieval_data.items())[:10000]:
        for doc in docs[:8]:
            all_title.append(doc["title"])
    # 记录所有使用的items
    use_items = [title2doc[title] for title in all_title if title in kilt_title]

    clean_items = []  # {"title":"", "text": "", }
    for item in tqdm(use_items):
        text_list = item["text"]
        # 去除所有带BULLET的句子
        text_list = [text for text in text_list if "BULLET" not in text]
        new_text_list = []
        for idx, text in enumerate(text_list):
            if "Section" in text:
                if "Section::::See also" in text:
                    break
                if idx + 1 != len(text_list) and "Section" not in text_list[idx + 1]:
                    new_text_list.append(text)
            else:
                new_text_list.append(text)

        clean_items.append({"title": item["title"], "text": new_text_list})
    print("total clean items num:", len(clean_items))

    # 构造训练数据
    training_data = []
    for item in tqdm(clean_items):
        title = item["title"]
        text_list = item["text"]
        if len(text_list) < 3:
            print("error")
            continue
        if text_list[0].strip("\n").strip() == title:
            text_list = text_list[1:]

        new_text_list = []
        for text in text_list:
            if "Section" not in text:
                new_text = text.replace("\n", "<new_tag>").strip()
            else:
                new_text = text.strip("\n").strip()
            new_text_list.append(new_text)

        new_text_list = [i.strip("\n").strip() for i in new_text_list if i.count(".:") < 2]  # 去掉三层的小标题
        target_text = " ".join(new_text_list)

        content = " ".join([text.strip("\n").strip() for text in text_list if "Section" not in text])
        # content = f"{title}\n{content}"
        # 构造最终的target字符串
        try:
            target = format_sections(target_text)
        except Exception as e:
            print(f"error: {e}")
            continue
        training_data.append(
            {
                "instruction": "Divide the following long text into well-structured, appropriately sized chapters.",
                "input": content,
                "output": target,
            }
        )
    return training_data


tokenizer = AutoTokenizer.from_pretrained("Qwen/qwen2.5-3B-Instruct")
training_data = build_single_data()
with open("training_data.json", "w") as f:
    json.dump(training_data, f, indent=4)
