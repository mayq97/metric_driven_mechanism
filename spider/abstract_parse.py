import requests
import os
from lxml import etree
import json
import requests
from tqdm import tqdm
from semanticscholar import SemanticScholar

html_dir = "./spider/html/"
paper_meta_dir = "./spider/parse_result"


s2_api_key = 's2_api_key'
sch = SemanticScholar(api_key=s2_api_key)


def get_paper_meta(paper_abs_text):
    paper_pdf_info = paper_abs_text.getprevious()
    pdf_url = paper_pdf_info.xpath(
        ".//a[@class='badge badge-primary align-middle mr-1']/@href")
    if len(pdf_url) > 0:
        pdf_url = pdf_url[0]
    else:
        pdf_url = ""
    title = paper_pdf_info.xpath(".//a[@class='align-middle']//text()")
    if len(title) > 0:
        title = "".join(title)
    else:
        title = ""
    code = paper_pdf_info.xpath(
        ".//a[@class='badge badge-secondary align-middle mr-1 pwc-reduced-padding']/@href")
    if len(code) > 0:
        code = code[0]
    else:
        code = ""
    return {
        "pdf_url": pdf_url,
        "title": title,
        "pwc": code
    }


def warp_text(lines):
    new_lines = [line.strip() for line in lines]
    text = " ".join(new_lines)
    new_lines = [line.strip() for line in text.split("\n")]
    return " ".join(new_lines)


def get_acl_paper_abs_by_ss(acl_id):
    # acl官网上的论文摘要信息不完善，通过semanticscholar进行补全

    paper = sch.paper('ACL:{}'.format(acl_id))
    if len(paper) <= 0:
        return "", ""
    return paper['title'], paper["abstract"]


def get_2010s_papers(year):
    data = []
    page = etree.HTML(open(os.path.join(
        html_dir, "acl_{}.html".format(year)), "r", encoding="utf-8").read())

    Contents = page.xpath("//ul[@class='list-pl-responsive']/li/a/@href")
    main_papers_id = Contents[0]
    main_papers = page.xpath("//div[@id='{}']".format(main_papers_id[1:]))[0]

    papers = main_papers.xpath(".//p[@class='d-sm-flex align-items-stretch']")
    for paper_meta in tqdm(papers):
        pdf_url = paper_meta.xpath(
            ".//a[@class='badge badge-primary align-middle mr-1']/@href")[0]
        paper_id = pdf_url.split("/")[-1].replace(".pdf", "")
        meta = {}

        title, abstract = get_acl_paper_abs_by_ss(paper_id)

        meta["id"] = paper_id
        meta["text"] = abstract
        meta["pdf_url"] = pdf_url
        meta["title"] = title
        meta["pwc"] = ""
        meta["year"] = year
        data.append(meta)
    return data


def get_2020s_papers(year=2022):

    data = []
    page = etree.HTML(open(os.path.join(
        html_dir, "acl_{}.html".format(year)), "r", encoding="utf-8").read())

    Contents = page.xpath("//ul[@class='list-pl-responsive']/li/a/@href")
    main_papers_id = Contents[0]
    main_papers = page.xpath("//div[@id='{}']".format(main_papers_id[1:]))[0]
    temp_long_papers_prefix = main_papers_id[1:].upper()

    papers = main_papers.xpath(
        ".//div[@class='card bg-light mb-2 mb-lg-3 collapse abstract-collapse']")
    for paper_abs_text in papers:
        paper_id = paper_abs_text.get("id", None)
        meta = get_paper_meta(paper_abs_text)
        if not paper_id is None and ("--acl-main" in paper_id or "--acl-long" in paper_id or temp_long_papers_prefix in paper_id):

            paper_abstract = paper_abs_text.xpath(".//text()")
            if len(paper_abstract) > 0:
                meta["id"] = paper_id
                meta["text"] = warp_text(paper_abstract).strip()
                data.append(meta)
    print(len(data))
    return data


if __name__ == "__main__":
    data = []
    papers = get_2010s_papers(2010)
    data.extend(papers)
    papers = get_2010s_papers(2011)
    data.extend(papers)
    json.dump(
        data,
        open(paper_meta_dir + "/{}_data.json".format("2010-2011"),
             "w", encoding="utf-8"),
        indent=2,
        ensure_ascii=False
    )
