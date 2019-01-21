# Previsão da Qualidade de Vinhos portugueses

Neste repositório encontra-se a implementação de um modelo para estimar a qualidade de vinho verde do Portugal. As soluções se encontram em um programa em Python (arquivo .py), para ser executado em forma de script, como também se encontram em notebooks implementados no Jupyter.

## Análise Exploratória dos Dados
Dentro do diretório "eda" encontra-se a descrição e código fonte da análise exploratória realizada nos dados em questão, a qual foi implementada no arquivo "EDA.ipynb". Tal notebook pode ser visualizado, já processado, no arquivo "EDA.html". Para fazer tal análise, implementou-se, em "edautils.py", um conjunto de funções úteis para esta análise.

A limpeza dos dados, detecção de outliers e feature engineering foram realizados dentro da análise exploratória, onde o notebook responsável pela mesma ("eda/EDA.ipynb") gera um arquivo ("winequality_processed.csv") já processado e pronto para ser utilizado na definição do modelo.

## Definição do Modelo
O código fonte da implementação do modelo estimador da qualidade do vinho se encontra em "modelGenerator.py". Para gerar tal modelo, utilizou-se o notebook "WineQualityModeling.ipynb", o qual também está salvo em uma versão ".hmtl" dentro do mesmo diretório. Dentro de tal notebook se descrevem todos passos percorridos e conclusões atingidas para a definição deste modelo. 

Tal notebook carrega a amostra de dados já processada, no qual utliza-se validação cruzada (com k=10) para avaliar a precisão atingida nos dados de treino fornecidos. Inicialmente avaliaram-se 5 tipos de classificadores, implementados na biblioteca sklearn. Posteriormente, nos 2 algoritmos que apresentaram melhor score, ajustou-se seus parâmetros utilizando a implementação de GridSearch da mesma biblioteca. Foram combinados diversos valores de parâmetros em 5 iterações (k=5), nas quais, em cada uma, uma parte dos dados foi separada para ser o conjunto de teste.

Em tal notebook também foi mencionado a função custo utilizada pelo algoritmo responsável por generalizar as características do conjunto de dados de treino. 

Para avaliar o modelo, além dos scores da validação cruzada executado no modelo final, traçou-se uma curva de aprendizagem para avaliar a generalização do modelo, observado-se que o mesmo não está sobre ajustado (overfitted). Por fim, comparou-se os scores de precisão atingidos pelo modelo final com os scores atingidos utilizando o conjunto dados originalmente disponibilizado pelo problema.
