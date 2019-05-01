import nltk

def extrai_words(classificacoes,textosPuros,textosQuebrados):
    # criamos aqui um dicionario das palavras de nosso 
    # arquivo , removendo as stopwords da lingua portuguesa
    stopwords = nltk.corpus.stopwords.words('portuguese')
    stemmer = nltk.stem.RSLPStemmer()
    dicionario = set()
    for lista in textosQuebrados:
        validas = [stemmer.stem(palavra) for palavra in lista if palavra not in stopwords and len(palavra) > 2]
        dicionario.update(validas)
    return dicionario

def texto_in_vetor(dicionario,textosPuros):
    stemmer = nltk.stem.RSLPStemmer()
    totalDePalavras = len(dicionario)
    tuplas = zip(dicionario, range(totalDePalavras))
    tradutor = {palavra: indice for palavra, indice in tuplas}
    vetor = [0] * len(tradutor)
    for palavra in textosPuros:
        if len(palavra) > 0 and stemmer.stem(palavra) in tradutor:
            posicao = tradutor[stemmer.stem(palavra)]
            vetor[posicao] += 1
    return vetor
