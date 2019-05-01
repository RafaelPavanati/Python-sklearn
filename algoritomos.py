from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import fit_algoritimo
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB

def execulta_algoritimos(treino_dados,treino_marcacoes):

    resultados = {}

    modeloOneVsRest = OneVsRestClassifier(LinearSVC(random_state=0))
    resultadoOneVsRest = fit_algoritimo.fit_and_predict("OneVsRest", modeloOneVsRest, treino_dados, treino_marcacoes)
    resultados[resultadoOneVsRest] = modeloOneVsRest
  
    modeloOneVsOne = OneVsOneClassifier(LinearSVC(random_state=0))
    resultadoOneVsOne = fit_algoritimo.fit_and_predict("OneVsOne", modeloOneVsOne, treino_dados, treino_marcacoes)
    resultados[resultadoOneVsOne] = modeloOneVsOne

    modeloMultinomial = MultinomialNB()
    resultadoMultinomial = fit_algoritimo.fit_and_predict("MultinomialNB", modeloMultinomial, treino_dados, treino_marcacoes)
    resultados[resultadoMultinomial] = modeloMultinomial 

    modeloAdaBoost = AdaBoostClassifier(random_state=0)
    resultadoAdaBoost = fit_algoritimo.fit_and_predict("AdaBoostClassifier", modeloAdaBoost, treino_dados, treino_marcacoes)
    resultados[resultadoAdaBoost] = modeloAdaBoost
    return resultados