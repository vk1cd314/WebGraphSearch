
import requests
from bs4 import BeautifulSoup as bs
from collections import deque
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.tokenize import sent_tokenize
from transformers import BartForConditionalGeneration, BartTokenizer

nltk.download('punkt')

SEARCH_DEPTH = 3
PER_LEVEL_LIMIT = 2

def write_to_file(outputfile, text):
    with open(outputfile, 'a', encoding='utf-8') as file:
        file.write(text)
        file.write('\n')

def read_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def find_relevant_sections(text, prompt, threshold=0.5):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sentences = sent_tokenize(text)
    prompt_embedding = model.encode(prompt, convert_to_tensor=True)
    relevant_sections = []

    for sentence in sentences:
        sentence_embedding = model.encode(sentence, convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(sentence_embedding, prompt_embedding)

        if cosine_scores.item() >= threshold:
            relevant_sections.append(sentence)

    print(len(relevant_sections))
    return relevant_sections

def summarize_text(text):
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

    inputs = tokenizer([text], max_length=1024, return_tensors='pt', truncation=True)
    summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=200, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary

def filter(links):
    filtered_links = []
    for lnk in links:
        if "instagram.com" not in lnk and "youtube.com" not in lnk and "google.com" not in lnk and "facebook.com" not in lnk and "twitter.com" not in lnk:
            domain = lnk.split('//')[-1].split('/')[0]
            domain_parts = domain.split('.')
            if domain_parts[1] == 'wikipedia' and domain_parts[0] != 'en':
                continue
            filtered_links.append(lnk)
    return filtered_links

def fetch_google_links(query):
    links = []
    url = f"https://www.google.com/search?q={query}&hl=en"

    headers = {"User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
    added = []
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        soup = bs(response.text, "lxml")
        a_tags = soup.find_all("a")
        for link in a_tags:
            href = link.get("href")
            links.append(href)
        filtered_links = []
        for lnk in links:
            if lnk and lnk.startswith("http"):
                domain = lnk.split('//')[-1].split('/')[0]
                filtered_links.append(lnk)
                added.append(domain)
        links = filtered_links
    return links

def dfs_visit(url, depth, prompt):
    if depth == SEARCH_DEPTH:
        return []
    try:
        headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
        response = requests.get(url, headers=headers)
        soup = bs(response.content, 'html.parser')
        write_to_file("dfs.txt", soup.get_text())

        print("what")
        links = [lnk['href'] for lnk in soup.find_all('a', href=True) if lnk['href'].startswith("http")]
        links = filter(links)

        extracted_links = []
        for link in links[:PER_LEVEL_LIMIT]:
            extracted_links.extend(dfs_visit(link, depth + 1, prompt))
        return links[:PER_LEVEL_LIMIT] + extracted_links
    except Exception as e:
        print(f"Error processing {url}: {e}")
        return []

def bfs_visit(starting_links, depth):
    visited = set()
    queue = deque([(link, 0) for link in starting_links[:PER_LEVEL_LIMIT]])
    result_links = []

    while queue:
        url, level = queue.popleft()
        if level == depth or url in visited:
            continue

        visited.add(url)
        try:
            headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
            response = requests.get(url, headers=headers)
            soup = bs(response.content, 'html.parser')
            write_to_file("bfs.txt", soup.get_text())
            links = [lnk['href'] for lnk in soup.find_all('a', href=True) if lnk['href'].startswith("http")]
            links = filter(links)

            result_links.extend(links[:PER_LEVEL_LIMIT])
            for link in links[:PER_LEVEL_LIMIT]:
                queue.append((link, level + 1))
        except Exception as e:
            print(f"Error processing {url}: {e}")

    return result_links

def util_dfs(links, prompt):
    res = []
    print(links[:PER_LEVEL_LIMIT])
    for lnk in links[:PER_LEVEL_LIMIT]:
        tmp_res = dfs_visit(lnk, 0, prompt)
        res.extend(tmp_res)
    return res

def main(query, prompt):
    google_links = filter(fetch_google_links(query=query))
    # google_links = ['https://en.wikipedia.org/wiki/Kak%C3%A1', 'https://en.wikipedia.org/wiki/Kak%C3%A1', 'https://en.wikipedia.org/wiki/Caroline_Celico', 'https://en.wikipedia.org/wiki/Marcelo_Saragosa', 'https://en.wikipedia.org/wiki/Gama,_Federal_District', 'https://en.wikipedia.org/wiki/Eduardo_Delani', 'https://en.wikipedia.org/wiki/Kak%C3%A1', 'https://www.transfermarkt.com/kaka/profil/spieler/3366', 'https://www.transfermarkt.com/kaka/profil/spieler/3366', 'https://www.soccerconference.org/personnel/ricardo-izecson-kaka/', 'https://www.soccerconference.org/personnel/ricardo-izecson-kaka/', 'https://www.gettyimages.com/photos/ricardo-izecson-dos-santos-leite', 'https://www.gettyimages.com/photos/ricardo-izecson-dos-santos-leite', 'https://sports.ndtv.com/football/players/90532-kak%C3%A1-playerprofile', 'https://sports.ndtv.com/football/players/90532-kak%C3%A1-playerprofile', 'https://commons.wikimedia.org/wiki/File:Ricardo_Izecson_dos_Santos_Leite_(Kak%C3%A1)_01.jpg', 'https://commons.wikimedia.org/wiki/File:Ricardo_Izecson_dos_Santos_Leite_(Kak%C3%A1)_01.jpg', 'https://www.shutterstock.com/search/ricardo-izecson-dos-santos-leite', 'https://www.shutterstock.com/search/ricardo-izecson-dos-santos-leite', 'https://www.researchgate.net/figure/Figure-2-Ricardo-Izecson-dos-Santos-Leite-was-introduced-in-2009-by-Real-Madrid-as_fig2_331481610', 'https://www.researchgate.net/figure/Figure-2-Ricardo-Izecson-dos-Santos-Leite-was-introduced-in-2009-by-Real-Madrid-as_fig2_331481610']
    print("Google links:", google_links)

    print('DFS:')
    dfs_links = util_dfs(google_links, prompt)
    print(len(dfs_links), dfs_links)

    print('BFS:')
    bfs_links = bfs_visit(google_links, SEARCH_DEPTH)
    print(len(bfs_links), bfs_links)
    
    text = read_text_from_file("dfs.txt")
    relevant_sections = find_relevant_sections(text, prompt)

    summaries = [summarize_text(section) for section in relevant_sections]
    final_summary = '\n'.join(summaries)

    print(final_summary)
    write_to_file("summary_dfs.txt", final_summary)
    
    text = read_text_from_file("bfs.txt")
    relevant_sections = find_relevant_sections(text, prompt)

    print(final_summary)
    summaries = [summarize_text(section) for section in relevant_sections]
    final_summary = '\n'.join(summaries)
    
    write_to_file("summary_bfs.txt", final_summary)

main("Stephen Hawking", "Stephen Hawking and Black Holes")
