# Machado-GPT 📚

## Trabalho de Conclusão do Bacharelado de Matemática Aplicada e Computacional do Instituto de Matemática e Estatística da Universidade de São Paulo (IME-USP)

### Contextualização 🌍

Este repositório armazena os scripts utilizados para treinar um modelo do tipo GPT (Generative Pre-trained Transformer) utilizando como base de dados de treino a obra completa do escritor Machado de Assis.

📥 O dataset foi obtido em https://www.kaggle.com/datasets/luxedo/machado-de-assis. 

A base de dados possui apenas 11MB e o modelo treinado tem 28,5 milhões de parâmetros. Tais números são considerados muito pequenos para os padrões atuais de treinamento dos modelos de linguagem. Deste modo, é valido salientar que o objetivo principal deste trabalho foi o de treinar um protótipo de um modelo GPT para fins didáticos. Os detalhes estão descritos no TCC intitulado "Anatomia do GPT: Aspectos matemáticos e computacionais de um grande modelo de linguagem", disponível neste repositório. 

Para o treinamento, que durou 8 horas, foi utilizada uma única GPU NVIDIA A100 (40GB). A busca de parâmetros, contudo, consumiu +100 horas de computação. 

Como futuros desdobramentos deste trabalho, poderíamos realizar procedimentos de data augmentation, ou seja, criação sintética de mais exemplos a partir da obra de Machado de Assis, com o objetivo de aumentar a base de dados. Além disso, se disponíveis os devidos recursos computacionais, poderíamos experimentar com modelos maiores, com mais parâmetros. 

### Resultados e Reflexões 🔍

A pergunta de pesquisa que fica é: dado um conjunto de dados pequeno, porém muito rico (escrito por aquele que talvez seja considerado o maior escritor brasileiro de todos os tempos), e dada a disponibilidade de recursos computacionais, qual o ganho de performance que conseguiríamos obter? 🤔

Abaixo, compartilhamos alguns excertos gerados pelo modelo pré-treinado. Possivelmente, aos leitores atentos de Machado de Assis, tais excertos gerados soem como "sonhos Machadianos". 💭

```
"ter sido mais que um sorriso de almas sem madrujo. Visões proviciosos! Que pode ter morro e paterno? Quereis que o Sr. Costa e a assinato definireis tudo? Cuidamos de versos paraúnos. "Viremos ter os sentimentos de quinze e seis linhas juntos extintas"; a amparata é votata, até sempre. Entretanto, um futuro, o ato, o desejo de ficar, é padecer sem um discurso às pessoas das câmaras. Acredito também ver que eu vejo tantas coisas que o alienado não for promovido em casa, se é assim que diabo, o relógio baixo-"
```

```
"por gente, pegou do marido e fê consegregar o filho para ele, e fechar-se-ia sem a falar; tinha confuso com igualdade que nos pensavam. Atalhou esta fraude ação com o seu Ablat? — Pode ser, sim, creio até em mim; eu recupei dez. E comi foi satisfeito; fecho os olhos, peguei-lhe nos lábios. Sondei que tomar à família aberta, pegando nas mãos abandonadas, onde não inclinariam algum tempo. Cinco ou meses era verdade, profundamente dinheiro, como ele mesmo pensava nos enrudoscos, mestre quando na minha opinião"
```

```
"o céu da morte, A linguagem em que passa usa o raciocanto Simão da harmonia. Ilusão desse domingo? Teve, Roma, como um pesante. A folha lá mesma e cala Feitas, ainda que ainda as moças levans à cau. Os uma feição que lhe levam tais alguns nomentos ao passado em cinco mil-réis, com outros outros e mais concertos, não menos grandes que aquele gozo secreto é sacrificado entre mim e o título sim. A observação nacional, a dona da casa, acha-se a primeira elegância do céu à nossa faca, e eu saí ao ponto de si do"
```

### Estrutura do Repositório 📂

Disponibilizamos três scripts em Pythons:

- machado-gpt.py: realiza o treinamento local do modelo, tomando como base a obra completa concatenada e disponibilizada na pasta /processado

- machado-gpt-wandb.py: realiza o treinamento local do modelo com log de métricas na plataforma Weights & Biases.

- machado-gpt-wandb-sweep.py: realiza o treinamento local do modelo com varredura de parâmetros

Note que os scripts podem ser copiados e colados em um notebook do Google Colab para que eles se aproveitem dos recursos de GPU da plataforma. 

### Instalação das dependências 🛠️

- O arquivo requirements.txt possui todos os pacotes necessários para rodar os scripts. 

### Inspiração 💡

O código em PyTorch foi inspirado na implementação de Andrej Karpathy, encontrada no vídeo do YouTube: 
"Let's build GPT: from scratch, in code, spelled out"
