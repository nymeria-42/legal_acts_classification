# %%
import re
import os
import pandas as pd

from  pathlib import Path
import helpers

def process(txts: list[str], mask = False, stopwords=False) -> list[str]:
    new_txts = []
    texts_processed = []

    file_path = "data/act_names.csv"
    act_types = pd.read_csv(file_path)
    act_types = act_types["act_names"].tolist()

    for txt in txts:

        for act_type in act_types:
            if mask:
                txt = re.sub(f"\W{re.escape(act_type)}\W", " [TIPO DE ATO] ", txt, flags=re.IGNORECASE|re.DOTALL)
            else:
                txt = re.sub(f"\W{re.escape(act_type)}\W", " ", txt, flags=re.IGNORECASE|re.DOTALL)

        if not mask:
            txt = re.sub(r"resolu[cç][aã]o\s+homologat[oó]ria\s+", " ", txt, flags=re.IGNORECASE|re.DOTALL)
            txt = re.sub(r"resolu[cç][aã]o\s+normativa\s+", " ", txt, flags=re.IGNORECASE|re.DOTALL)
            txt = re.sub(r"resolu[cç][aã]o\s+autorizativa\s+", " ", txt, flags=re.IGNORECASE|re.DOTALL)
            txt = re.sub(r"resolu[cç][aã]o", " ", txt, flags=re.IGNORECASE|re.DOTALL)
            txt = re.sub(r"diretoria colegiada", " ", txt, flags=re.IGNORECASE|re.DOTALL)
            txt = re.sub(r"rdc", " ", txt, flags=re.IGNORECASE|re.DOTALL)
        else:
            txt = re.sub(r"resolu[cç][aã]o\s+homologat[oó]ria\s+", " [TIPO DE ATO] ", txt, flags=re.IGNORECASE|re.DOTALL)
            txt = re.sub(r"resolu[cç][aã]o\s+normativa\s+", " [TIPO DE ATO] ", txt, flags=re.IGNORECASE|re.DOTALL)
            txt = re.sub(r"resolu[cç][aã]o\s+autorizativa\s+", " [TIPO DE ATO] ", txt, flags=re.IGNORECASE|re.DOTALL)
            txt = re.sub(r"resolu[cç][aã]o", " [TIPO DE ATO] ", txt, flags=re.IGNORECASE|re.DOTALL)
            txt = re.sub(r"diretoria colegiada", " [TIPO DE ATO] ", txt, flags=re.IGNORECASE|re.DOTALL)
            txt = re.sub(r"rdc", "", txt, flags=re.IGNORECASE|re.DOTALL)

        if not mask:
            txt = re.sub(r"[N|n][º|°]\s", " ", txt)

        agencias = ["ANEEL", "ANVISA", "ANS", "ANTT", "ANA", "ANAC", "CVM", "ANTAQ", "ANP", "ANATEL", "ANM", "ANCINE", "BACEN"] 
        for agencia in agencias:
            if mask:
                txt = re.sub(f"\W{agencia}\W", " [AGENCIA] ", txt, flags=re.IGNORECASE|re.DOTALL)
            else:
                txt = re.sub(f"\W{agencia}\W", " ", txt, flags=re.IGNORECASE|re.DOTALL)  

        if mask:
            txt = re.sub(r"[\d\.\-]+", " [NUM] ", txt, flags=re.IGNORECASE|re.DOTALL)
        else:
            txt = re.sub(r"[\d\.\-]+", " ", txt, flags=re.IGNORECASE|re.DOTALL)

        
        meses_ano = ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto", "setembro", "outubro", "novembro", "dezembro"]
        for mes in meses_ano:
            if mask:
                txt = re.sub(mes, " [MES] ", txt, flags=re.IGNORECASE|re.DOTALL)
            else:
                txt = re.sub(mes, " ", txt, flags=re.IGNORECASE|re.DOTALL)

        if mask:
            txt = re.sub(r"mar[çc]o", " [MES] ", txt, flags=re.IGNORECASE|re.DOTALL)
        else:
            txt = re.sub(r"mar[çc]o", " ", txt, flags=re.IGNORECASE|re.DOTALL)
        
      
        txt = re.sub(r"http\S+", "", txt)
        txt = re.sub(r"www\S+", "", txt)
        txt = re.sub(r"/\S+", "", txt)

        texts_processed.append(txt)

    texts = texts_processed

    # aggregate articles
    for txt in texts:

        initial_paragraphs = re.split(r"(?<!\n)\n{2,}(?!\n)", txt)

        new_txt = []

        for parapraph in initial_paragraphs:
            type_paragraph = helpers.find_legal_structure(parapraph)
            if type_paragraph:
                lines = re.split(r"(?<!\n)\n(?!\n)", parapraph)

                for line in lines:
                    type_line = helpers.find_legal_structure(line)

                    if re.search(r"\.{4,}", line) is not None:
                        continue

                    inciso = re.compile(r"^[^\"]?(I|V|X|L|C)+\s")
                    paragrafo = re.compile(r"^[^\"]?§")
                    parag_unico = re.compile(r"^[^\"]?(par[áa]grafo)\s([úu]nico)|P\.U\.", re.IGNORECASE)
                    alinea = re.compile(r"^[^\"]?[a-z](\s)?\)")
                    
                    if not mask and stopwords:
                        line = helpers.remove_stopwords(line.lower())

                    for dispositivo in [inciso, paragrafo, parag_unico, alinea]:
                        line = re.sub(dispositivo, "", line)
                        
                    line = re.sub(r"\s+", " ", line)

                    if type_line == "artigo":
                        new_txt.append(line)

                    elif new_txt:
                        new_txt[-1]+= line

        if len(new_txt):
            new_txt[-1] = new_txt[-1].split("\n\n")[0]
        
        new_txt = "\n".join(new_txt)
        new_txts.append(new_txt)

    return new_txts


if __name__ == "__main__":
    Path.mkdir(Path("data/preprocessed"), exist_ok=True, parents=True)

    masks = [False, True]
    for mask in masks:
        df_concretes = pd.read_csv("data/to_process/texts_concretes.csv")
        df_concretes["labels"] = "concreta"

        df_abstracts = pd.read_csv("data/to_process/texts_abstracts_without_ANVISA_ANEEL.csv")
        df_abstracts["labels"] = "abstrata"

        df = pd.concat([df_concretes, df_abstracts])
        
        df_anvisa_aneel = pd.read_csv("data/to_process/texts_ANVISA_ANEEL.csv")

        if mask:
            stopwords=False
        else:
            stopwords=True
            
        txts = df["text"]

        new_txts_dispositivos = process(txts, mask=mask, stopwords=stopwords)
        df["text"] = new_txts_dispositivos
        df.dropna(subset=["text"], inplace=True)
        
        
        if mask:
            df.to_csv("data/preprocessed/texts_without_ANVISA_ANEEL_mask.csv", index=False)
        else:
            df.to_csv("data/preprocessed/texts_without_ANVISA_ANEEL.csv", index=False)

        txts = df_anvisa_aneel["text"]
        new_txts_dispositivos = process(txts, mask=mask, stopwords=stopwords)
        df_anvisa_aneel["text"] = new_txts_dispositivos
        df_anvisa_aneel.dropna(subset=["text"], inplace=True)
        
        if mask:
            df_anvisa_aneel.to_csv("data/preprocessed/texts_ANVISA_ANEEL_mask.csv", index=False)
        else:
            df_anvisa_aneel.to_csv("data/preprocessed/texts_ANVISA_ANEEL.csv", index=False)
