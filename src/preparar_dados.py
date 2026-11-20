"""Limpeza e harmonização dos dados de acidentes da PRF (DATATRAN 2017-2025).

Uso:  python src/preparar_dados.py <pasta_com_datatranAAAA.csv>
Saída: dados/processados/acidentes_limpo.parquet + tabelas agregadas (CSV) para o Looker Studio.
"""
import sys, pathlib, unicodedata
import numpy as np
import pandas as pd

RAIZ = pathlib.Path(__file__).resolve().parents[1]
SAIDA = RAIZ / "dados" / "processados"

# ---------------------------------------------------------------- causas
# A PRF trocou a taxonomia de causas em 2021. Agrupamos as ~90 categorias
# em 12 macrogrupos estáveis no tempo.
GRUPOS_CAUSA = {
    "Álcool/drogas (condutor)": [
        "Ingestão de Álcool", "Ingestão de álcool pelo condutor",
        "Ingestão de Substâncias Psicoativas", "Ingestão de substâncias psicoativas pelo condutor"],
    "Velocidade incompatível": ["Velocidade Incompatível", "Participar de racha"],
    "Desatenção/reação tardia": [
        "Falta de Atenção à Condução", "Reação tardia ou ineficiente do condutor",
        "Ausência de reação do condutor", "Condutor usando celular",
        "Condutor deixou de manter distância do veículo da frente",
        "Não guardar distância de segurança", "Frear bruscamente"],
    "Ultrapassagem/contramão/manobra": [
        "Ultrapassagem Indevida", "Transitar na contramão", "Manobra de mudança de faixa",
        "Trafegar com motocicleta (ou similar) entre as faixas", "Retorno proibido",
        "Conversão proibida"],
    "Desrespeito a normas/preferência": [
        "Desobediência às normas de trânsito pelo condutor", "Desrespeitar a preferência no cruzamento",
        "Condutor desrespeitou a iluminação vermelha do semáforo",
        "Acessar a via sem observar a presença dos outros veículos", "Acesso irregular",
        "Estacionar ou parar em local proibido", "Transitar no acostamento", "Transitar no Acostamento",
        "Transitar na calçada", "Deixar de acionar o farol da motocicleta (ou similar)"],
    "Sono/mal súbito do condutor": [
        "Condutor Dormindo", "Mal Súbito", "Mal súbito do condutor",
        "Transtornos Mentais (exceto suicidio)", "Suicídio (presumido)"],
    "Pedestre": [
        "Falta de Atenção do Pedestre", "Entrada inopinada do pedestre",
        "Desobediência às normas de trânsito pelo pedestre", "Pedestre andava na pista",
        "Pedestre cruzava a pista fora da faixa", "Pedestre - Ingestão de álcool/ substâncias psicoativas",
        "Ingestão de álcool e/ou substâncias psicoativas pelo pedestre",
        "Ingestão de álcool ou de substâncias psicoativas pelo pedestre",
        "Área urbana sem a presença de local apropriado para a travessia de pedestres"],
    "Veículo (falha mecânica/carga)": [
        "Defeito Mecânico no Veículo", "Demais falhas mecânicas ou elétricas",
        "Avarias e/ou desgaste excessivo no pneu", "Problema com o freio", "Problema na suspensão",
        "Carga excessiva e/ou mal acondicionada", "Faróis desregulados", "Modificação proibida",
        "Deficiência ou não Acionamento do Sistema de Iluminação/Sinalização do Veículo"],
    "Via/infraestrutura": [
        "Defeito na Via", "Pista esburacada", "Acostamento em desnível", "Afundamento ou ondulação no pavimento",
        "Pista em desnível", "Falta de acostamento", "Faixas de trânsito com largura insuficiente",
        "Curva acentuada", "Declive acentuado", "Ausência de sinalização",
        "Sinalização da via insuficiente ou inadequada", "Sinalização encoberta", "Sinalização mal posicionada",
        "Deficiência do Sistema de Iluminação/Sinalização", "Iluminação deficiente", "Semáforo com defeito",
        "Redutor de velocidade em desacordo", "Obras na pista", "Desvio temporário", "Demais falhas na via",
        "Sistema de drenagem ineficiente", "Falta de elemento de contenção que evite a saída do leito carroçável",
        "Restrição de visibilidade em curvas horizontais", "Restrição de visibilidade em curvas verticais",
        "Restrição de Visibilidade", "Objeto estático sobre o leito carroçável", "Obstrução na via",
        "Acumulo de areia ou detritos sobre o pavimento", "Acumulo de óleo sobre o pavimento",
        "Acumulo de água sobre o pavimento", "Pista Escorregadia"],
    "Clima/fenômenos naturais": ["Chuva", "Neblina", "Fumaça", "Fenômenos da Natureza",
                                  "Demais Fenômenos da natureza"],
    "Animais na pista": ["Animais na Pista"],
    "Outros (agressão/assalto)": ["Agressão Externa", "Obstrução Via tentativa Assalto"],
}
MAPA_CAUSA = {c: g for g, lst in GRUPOS_CAUSA.items() for c in lst}

# ---------------------------------------------------------------- tipos
MAPA_TIPO = {
    "Colisão lateral mesmo sentido": "Colisão lateral",
    "Colisão lateral sentido oposto": "Colisão lateral",
    "Colisão com objeto estático": "Colisão com objeto",
    "Colisão com objeto em movimento": "Colisão com objeto",
    "Colisão com objeto fixo": "Colisão com objeto",
    "Colisão com objeto móvel": "Colisão com objeto",
    "Danos eventuais": "Outros",
    "Eventos atípicos": "Outros",
    "Derramamento de carga": "Outros",
    "Sinistro pessoal de trânsito": "Outros",
}

DIAS = ["segunda", "terça", "quarta", "quinta", "sexta", "sábado", "domingo"]


def num(s):
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False).str.strip(), errors="coerce")


def carregar(pasta):
    partes = []
    for arq in sorted(pathlib.Path(pasta).glob("datatran20*.csv")):
        d = pd.read_csv(arq, sep=";", encoding="latin1", dtype=str)
        d["ano_arquivo"] = int(arq.stem[-4:])
        partes.append(d)
    return pd.concat(partes, ignore_index=True)


def limpar(df):
    out = pd.DataFrame()
    out["id"] = num(df["id"]).astype("Int64")
    out["data"] = pd.to_datetime(df["data_inversa"], errors="coerce")
    out["ano"] = out["data"].dt.year
    out["mes"] = out["data"].dt.month
    out["hora"] = pd.to_datetime(df["horario"], format="%H:%M:%S", errors="coerce").dt.hour
    out["dia_semana"] = out["data"].dt.dayofweek.map(dict(enumerate(DIAS)))
    out["fim_de_semana"] = out["data"].dt.dayofweek >= 5
    out["uf"] = df["uf"].str.strip()
    out["br"] = num(df["br"]).astype("Int64")
    out["km"] = num(df["km"])
    out["municipio"] = df["municipio"].str.strip().str.upper()
    out["causa_original"] = df["causa_acidente"].str.strip()
    out["causa_grupo"] = out["causa_original"].map(MAPA_CAUSA).fillna("Não informado")
    tipo = df["tipo_acidente"].str.strip()
    out["tipo_acidente"] = tipo.map(MAPA_TIPO).fillna(tipo).fillna("Não informado")
    out["classificacao"] = df["classificacao_acidente"].str.strip()
    out["fase_dia"] = df["fase_dia"].str.strip()
    out["noite"] = out["fase_dia"].eq("Plena Noite")
    out["clima"] = df["condicao_metereologica"].str.strip().replace({"Ignorado": np.nan})
    out["tipo_pista"] = df["tipo_pista"].str.strip()
    out["tracado_via"] = df["tracado_via"].str.split(";").str[0].str.strip()
    out["urbano"] = df["uso_solo"].str.strip().eq("Sim")
    for c in ["pessoas", "mortos", "feridos_leves", "feridos_graves", "ilesos", "ignorados", "feridos", "veiculos"]:
        out[c] = num(df[c]).fillna(0).astype(int)
    out["latitude"] = num(df["latitude"])
    out["longitude"] = num(df["longitude"])
    # coordenadas fora do território brasileiro viram nulas
    fora = ~out["latitude"].between(-34, 6) | ~out["longitude"].between(-74, -34)
    out.loc[fora, ["latitude", "longitude"]] = np.nan
    out["fatal"] = (out["mortos"] > 0).astype(int)
    out = out.drop_duplicates("id")
    return out


def letal(g):
    return g.agg(acidentes=("id", "size"), mortos=("mortos", "sum"),
                 feridos_graves=("feridos_graves", "sum"), acidentes_fatais=("fatal", "sum")
                 ).assign(mortos_por_100=lambda x: (100 * x.mortos / x.acidentes).round(2)).reset_index()


def agregados(df):
    SAIDA.mkdir(parents=True, exist_ok=True)
    tabs = {
        "agg_ano": letal(df.groupby("ano")),
        "agg_uf_ano": letal(df.groupby(["uf", "ano"])),
        "agg_br": letal(df.groupby(["br", "uf"])).query("acidentes >= 300")
                  .assign(trecho=lambda x: "BR-" + x.br.astype(str).str.zfill(3) + "/" + x.uf),
        "agg_tipo": letal(df.groupby("tipo_acidente")),
        "agg_causa": letal(df.groupby("causa_grupo")),
        "agg_hora_fase": letal(df.groupby(["hora", "fase_dia"])),
        "agg_dia_hora": letal(df.groupby(["dia_semana", "hora"])),
        "agg_condicoes": letal(df.groupby(["tipo_pista", "urbano", "clima"], dropna=False)),
    }
    # tabela "plana" para o Looker: uma linha por combinação de dimensões
    dims = ["ano", "uf", "tipo_acidente", "causa_grupo", "fase_dia", "tipo_pista", "urbano"]
    plano = df.groupby(dims, dropna=False).agg(
        acidentes=("id", "size"), acidentes_fatais=("fatal", "sum"), mortos=("mortos", "sum"),
        feridos_graves=("feridos_graves", "sum"), feridos_leves=("feridos_leves", "sum"),
        veiculos=("veiculos", "sum")).reset_index()
    plano["urbano"] = plano["urbano"].map({True: "Urbano", False: "Rural"})
    tabs["looker_acidentes_agregado"] = plano
    for nome, t in tabs.items():
        t.to_csv(SAIDA / f"{nome}.csv", index=False, encoding="utf-8")
    return tabs


if __name__ == "__main__":
    bruto = carregar(sys.argv[1])
    print("linhas brutas:", len(bruto))
    df = limpar(bruto)
    print("linhas limpas:", len(df), "| sem grupo de causa:", (df.causa_grupo == "Não informado").sum())
    df.to_parquet(SAIDA / "acidentes_limpo.parquet", index=False)
    tabs = agregados(df)
    print(tabs["agg_ano"].to_string(index=False))
    print("looker linhas:", len(tabs["looker_acidentes_agregado"]))
