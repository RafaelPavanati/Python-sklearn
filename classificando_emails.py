#!-*- coding: utf8 -*-

import pandas as pd
from collections import Counter
import nltk
import fit_algoritimo
import extrai_top_words
import algoritomos

classificacoes = pd.read_csv('emails.csv', encoding = 'utf-8')
textosPuros = classificacoes['email']
frases = textosPuros.str.lower()
textosQuebrados = [ nltk.tokenize.word_tokenize(frase) for frase in frases]

dicionario = extrai_top_words.extrai_words(classificacoes,textosPuros,textosQuebrados)

vetoresDeTexto = [extrai_top_words.texto_in_vetor(dicionario,texto) for texto in textosQuebrados]

marcas = classificacoes['classificacao']
X = vetoresDeTexto
Y = marcas
porcentagem_de_treino = 0.8
tamanho_de_treino = int(porcentagem_de_treino * len(Y))
tamanho_de_validacao = len(Y) - tamanho_de_treino

treino_dados = X[0:tamanho_de_treino]
treino_marcacoes = Y[0:tamanho_de_treino]
validacao_dados = X[tamanho_de_treino:]
validacao_marcacoes = Y[tamanho_de_treino:]

resultados = algoritomos.execulta_algoritimos(treino_dados,treino_marcacoes)

maximo = max(resultados)
vencedor = resultados[maximo]

print("Vencerdor: ")
print(vencedor)

vencedor.fit(treino_dados, treino_marcacoes)

fit_algoritimo.teste_real(vencedor, validacao_dados, validacao_marcacoes)

acerto_base = max(Counter(validacao_marcacoes).values())
taxa_de_acerto_base = 100.0 * acerto_base / len(validacao_marcacoes)
print("Taxa de acerto base: %f" % taxa_de_acerto_base)

total_de_elementos = len(validacao_dados)
print("Total de teste: %d" % total_de_elementos)