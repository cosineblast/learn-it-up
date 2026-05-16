
<!-- remova esse link quando for gerar o pdf -->
[Versão PDF](/proposta.pdf)

Meu nome é Renan Ribeiro Marcelino. Estou no quinto ano do Bacharelado em Ciência de Computação na Universidade de São Paulo, e este documento registra um projeto de pesquisa em Machine Learning do qual estou desenvolvendo.

## DanceDanceRevolution e Pump it Up
DanceDanceRevolution e Pump it Up são duas franquias de jogos de ritmo populares, onde os jogadores apertam botões com seus pés em um tapete especial conforme uma música toca, com setas na tela indicando quais botões devem ser apertados. A principal diferença entre esses jogos é no layout do tapete, onde DanceDanceRevolution utiliza 4 teclas, e PumpItUp utiliza 5.

![**Imagem 1:** Duas pessoas jogando Pump it Up](/pump_coop.jpg)

![**Imagem 2:** Duas pessoas jogando DanceDanceRevolution](/ddr_coop.jpg)

##  Aprendizado de Coreografia
Em 2017, o paper [DanceDanceConvolution](https://arxiv.org/abs/1703.06891) introduziu o conceito de "aprender a coreografar", que consiste no uso de machine learning para produzir uma coreografia de botões de Dance Dance Revolution, vulgo _chart_,  a partir de um arquivo de música arbitrário. Este trabalho foi expandido por diferentes autores em 2019 com o paper [DanceDanceGradation](https://inria.hal.science/hal-02128628/file/472623_1_En_15_Chapter.pdf) para dar suporte a músicas de dificuldade baixa, e em 2025 com o paper [DanceDanceConvLSTM](https://arxiv.org/abs/2507.01644) para experimentar com células Long Short Term Memory Convolucionais.

Como é possível notar, esses papers todos trabalham em coreografar músicas do jogo DanceDanceRevolution, mas não consideram o jogo Pump it Up. Apesar de similares, a diferença de layout dos jogos traz diversas diferenças em tradições de _charts_, como a presença de padrões técnicos como, _spins_ e _twists_ e _brackets_ no jogo de 5 teclas.

Estudar a viabilidade das técnicas apresentadas no estado da arte de aprendizado de coreografias de DanceDanceRevolution para o PumpItUp é um trabalho que ainda não foi publicado.

Além disso, Pump it Up apresenta um potencial adicional de pesquisa, comparado a DanceDanceRevolution: O projeto [piu-analysis](https://github.com/maxwshen/piu-analysis) de Max Shen que consiste na classificação automática de todas as dificuldades de todas as músicas oficiais do jogo Pump it Up, classificando, para cada _chart_, quais são os principais padrões de setas que ocorrem nas músicas e qual a densidade de setas de cada seção da música. Esse dataset exclusivo de Pump it Up abre um potencial considerável para permitir uma geração mais precisa de charts de Pump.
Este projeto já é utilizado no site [piucenter.com](https://www.piucenter.com/) que permite que jogadores procurem níveis com características específicas que desejam treinar.

![**Imagem 3:** A distribuição dos níveis das músicas e densidade de notas (notas por segundo) das músicas de Pump it Up. O jogo pump it up apresenta mais granularidade de dificuldade do que o jogo de 4 teclas, o que introduz trabalho adicional na adaptação dos modelos existentes de DDR, uma vez que a dificuldade das músicas é utilizada como entrada para os modelos generativos.](/chart_dist.jpg)

Outro ponto importante, é sobre as técnicas e tecnologias aplicadas nos papers que estudam DanceDanceRevolution até agora. Em geral, todos os papers mencionados utilizam LSTM como fundação de implementação. A arquitetura de Transformer anda demonstrando resultados excepcionais em diversas outras tarefas de geração de sequências. Um [paper](https://arxiv.org/abs/2311.13687) de 2023 já começa a trabalhar no uso de transformers para estas tarefas.

Além disso, em todos esses papers, o algoritmo de previsão da ordem das setas não leva em consideração as informações musicais de notas da música (apenas usam como entrada informações de timing). Nos papers utilizando LSTM, a introdução dessas informações aumentou o overfitting dos modelos, mas isso não foi trabalhado no paper de 2023 que inclui o uso de transformers.

## Proposta
Com isso, apresento um projeto de pesquisa que consiste em:

- Estudar e documentar a aplicabilidade de técnicas existentes de geração de _charts_ de DanceDanceRevolution para Pump it Up. 
- Estudar a aplicabilidade de informações musicais (além de timing) em arquiteturas de transformers na geração de charts de Pump it Up.
- Tentar produzir o melhor gerador de charts de pump it up (até o momento) com auxílio do dataset providenciado do projeto piu-analysis.
- Providenciar acesso fácil (por meio de um website) ao modelo de geração de charts, e coletar feedback dos jogadores sobre os charts gerados, assim como o paper DanceDanceConvolution original faz, e utilizar isto para determinar a performance dos resultados obtidos.
