import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import time

print("\n")
print("###################### Machado-GPT ######################")
print("\n")

# Usamos GPU se estiver disponível, caso contrário usamos CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dispositivo de treinamento definido: {device}")

# Hiperparâmetros do melhor modelo obtido durante o treinamento
batch_size = 512  # Tamanho do lote de dados
context_len = 128  # Tamanho da janela de contexto
n_embd = 512  # Dimensão dos embeddings
n_head = 32  # Número de cabeças de atenção
n_layer = 9  # Número de blocos Transformer (Atenção + FFNN)
learning_rate = 1e-3  # Taxa de aprendizagem da Descida do Gradiente
dropout = 0.2  # Percentual de ativações ignoradas
eval_interval = 1000  # Intervalo de avaliações da função de perda
eval_iters = 50  # Número de iterações usadas para o cálculo da perda
max_iters = 10000  # Número total de iterações do treinamento

# Leitura dos dados concatenados
print("\nLendo os dados de treinamento...")
text = open("processado/textos_concatenados.txt", "r", encoding="utf-8").read()

# Criação do vocabulário composto de caracteres únicos
chars = sorted(set(text))
vocab_size = len(chars)
print("\nNúmero de tokens do vocabulário:", vocab_size)


# Indexação dos tokens
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# Funções para codificar e decodificar índices e tokens
def encode(text):
    return [char_to_idx[ch] for ch in text]

def decode(indices):
    return "".join(idx_to_char[i] for i in indices)

# Codificação em índices dos dados de treinamento
data = torch.tensor(encode(text), dtype=torch.long)
print("\nNúmero total de tokens no conjunto de dados:", len(data))

print("\nDividindo os dados em conjuntos de treino e validação...")

# Divisão dos dados em treino e validação
train_percent = 0.9
n = int(train_percent * len(data))
train_data = data[:n]
val_data = data[n:]

# Função para obter lotes de dados
def get_batch(split):
    # Seleciona se os dados virão do conjunto de treino ou validação
    data = train_data if split == "train" else val_data
    # Gera batch_size sequências aleatórias de índices iniciais para cada lote
    ix = torch.randint(len(data) - context_len, (batch_size,))
    # Gera sequências de treinamento baseadas no tamanho da janela de contexto
    x = torch.stack([data[i : i + context_len] for i in ix])
    # Gera rótulos para as sequências de treinamento
    y = torch.stack([data[i + 1 : i + context_len + 1] for i in ix])
    # Envia os tensores para a GPU ou CPU
    x, y = x.to(device), y.to(device)
    return x, y

# Decorador para evitar o cálculo dos gradientes durante o cálculo da perda
@torch.no_grad()
# Função que calcula a perda
def estimate_loss():
    out = {}
    # Coloca o modelo em modo de avaliação
    model.eval()
    # Itera sobre os conjuntos de dados
    for split in ["train", "val"]:
        # Inicializa o vetor de perdas com zeros
        losses = torch.zeros(eval_iters)
        # Itera sobre o número de avaliações pré-determinadas
        for k in range(eval_iters):
            # Obtém exemplos aleatórios de sequências dos conjuntos de dados
            X, Y = get_batch(split)
            # Obtém os logits e a perda para os exemplos selecionados
            logits, loss = model(X, Y)
            # Atualiza o vetor de perdas
            losses[k] = loss.item()
        # Calcula a média da perda para cada conjunto de dados
        out[split] = losses.mean()
    # Recoloca o modelo em modo de treinamento
    model.train()
    return out

# Define a função que calcula a codificação posicional
def positional_encoding(context_len, n_embd, device):
    # Gera um tensor de posições
    position = torch.arange(context_len, 
                            dtype=torch.float32, 
                            device=device).unsqueeze(1)
    # Calcula o termo divisor para as funções seno e cosseno
    div_term = torch.exp(
        torch.arange(0, n_embd, 2, device=device).float() * 
        (-torch.log(torch.tensor(10000.0, device=device)) / n_embd))
    # Inicializa o tensor de codificação posicional com zeros
    pe = torch.zeros(context_len, n_embd, device=device)
    # Aplica a função seno aos índices pares da matriz posicional
    pe[:, 0::2] = torch.sin(position * div_term)
    # Aplica a função cosseno aos índices ímpares da matriz posicional
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

# Define a classe para uma cabeça de atenção
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        # Inicializa as matrizes de chaves, consultas e valores
        # Não inicializamos os viéses para simplificar o modelo
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Registra um buffer para a máscara de atenção triangular inferior
        # Trata-se de matriz com 1's abaixo da diagonal principal (inclusive) 
        # e zeros acima. O uso de 'register_buffer' permite que o tensor 'tril' 
        # seja movido junto com o modelo para a GPU, se disponível, e 
        # não seja considerado um parâmetro do modelo.
        self.register_buffer("tril", torch.tril(
            torch.ones(context_len, context_len)
            ))
        # Inicializa a camada de Dropout
        self.dropout = nn.Dropout(dropout)

    # Define a passagem forward do modelo
    def forward(self, x):
        # Obtém as dimensões do conjunto de dados
        batch_size, context_len, n_embd = x.shape
        # Projeção linear da entrada através da matriz de pesos das chaves
        k = self.key(x)
        # Projeção linear da entrada através da matriz de pesos das consultas
        q = self.query(x)
        # Produto escalar entre q e k com dimensionamento sqrt(n_embd)
        wei = q @ k.transpose(-2, -1) * n_embd ** (-0.5)
        # Aplicação da máscara na matriz resultado da operação anterior
        # Note que as entradas iguais a zero são redefinidas como -inf
        wei = wei.masked_fill(
            self.tril[:context_len, :context_len] == 0, float("-inf")
            )
        # Aplicação da função Softmax
        wei = F.softmax(wei, dim=-1)
        # Aplicação do Dropout
        wei = self.dropout(wei)
        # Projeção linear da entrada através da matriz de pesos dos valores
        v = self.value(x)
        # Multiplicação final pela matriz de valores
        out = wei @ v
        return out
    
# Definição da classe de múltiplas cabeças
class MultiHeadAttention(nn.Module):
    def __init__(self, num_head, head_size):
        super().__init__()
        # Inicializa uma lista de módulos 
        # com várias instâncias da cabeça de atenção
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_head)])
        # Inicializa uma camada linear de projeção dos dados
        self.proj = nn.Linear(n_embd, n_embd)
        # Inicializa a camada de Dropout
        self.dropout = nn.Dropout(dropout)
    # Define a passagem forward do modelo
    def forward(self, x):
        # Aplica as cabeças de atenção sobre os dados 
        #e as concatena no final 
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # Realiza a projeção linear do resultado anterior 
        # e aplica o dropout em seguida
        out = self.dropout(self.proj(out))
        return out
    
# Define a classe da rede Feed Forward    
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        # Cria a arquitetura da rede com uma camada linear, 
        # seguida de uma ativação ReLU, seguida por outra 
        # camada linear e, por fim, com uma camada de dropout
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    # Define a passagem forward do modelo
    def forward(self, x):
        # Aplica a sequência de camadas definida para a FFNN
        return self.net(x)

# Define um bloco Transformer
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        # Determina o tamanho de cada cabeça de atenção
        head_size = n_embd // n_head
        # Inicializa as múltiplas cabeças de atenção
        self.sa = MultiHeadAttention(n_head, head_size)
        # Inicializa a camada FFNN
        self.ffwd = FeedForward(n_embd)
        # Inicializa as camadas de normalização
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    # Define a passagem forward do modelo
    def forward(self, x):
        # Computa a normalização dos dados, seguida pela atenção com
        # múltiplas cabeças, além da soma da conexão residual
        x = x + self.sa(self.ln1(x))
        # Computa a normalização dos dados, seguida pela FFNN, 
        # além da soma da conexão residual
        x = x + self.ffwd(self.ln2(x))
        return x
    
# Define a classe do modelo GPT
class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Inicializa a matriz de pesos dos embeddings dos tokens
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        # Gera e armazena embeddings posicionais
        self.positional_embeddings = positional_encoding(
            context_len, n_embd, device
            )
        # Cria uma sequência de tamanho n_layer de blocos Transformer
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )
        # Inicializa a camada de normalização
        self.ln_f = nn.LayerNorm(n_embd)
        # Inicializa a camada linear
        self.lm_head = nn.Linear(n_embd, vocab_size)

    # Define a passagem forward do modelo
    def forward(self, idx, targets=None):
        # Obtém as dimensões do conjunto de dados
        batch_size, context_len = idx.shape
        # Embeddings de tokens
        tok_emb = self.token_embedding(idx)
        # Codificações posicionais dos tokens
        pos_emb = self.positional_embeddings[:context_len, :]
        # Soma os embeddings às codificações posicionais
        x = tok_emb + pos_emb
        # Aplica os Blocos Transformer nos dados
        x = self.blocks(x)
        # Aplica a camada de normalização
        x = self.ln_f(x)
        # Aplica a camada linear final para obtenção dos logits
        logits = self.lm_head(x)

        # Calcula a perda se os rótulos forem fornecidos
        if targets is None:
            loss = None
        else:
            # Obtém as dimensões dos logits
            batch_size, context_len, n_embd = logits.shape
            # Redimensiona (flattens) os 'logits' para uma matriz 2D, 
            # onde a primeira dimensão é batch_size * context_len e 
            # a segunda é n_embd .Isso é necessário para cálculo da 
            # função de perda de entropia cruzada
            logits = logits.view(batch_size * context_len, n_embd)
            # Redimensiona os rótulos alvo para um vetor 1D com tamanho 
            # batch_size * context_len para corresponder 
            # aos 'logits' redimensionados
            targets = targets.view(batch_size * context_len)
            # Calcula a perda usando a entropia cruzada, que compara 
            # os 'logits' da rede com os rótulos alvo. 
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    # Gera novas sequências de texto
    def generate(self, idx, max_new_tokens):
        # Itera até o tamanho máximo de tokens a ser gerado
        for _ in range(max_new_tokens):
            # Limita a entrada ao tamanho da janela de contexto
            idx_cond = idx[:, -context_len:]
            # Calcula os logits para o contexto de entrada
            logits, loss = self(idx_cond)
            # Extrai os logits
            logits = logits[:, -1, :]
            # Calcula as probabilidades com a Softmax
            probs = F.softmax(logits, dim=-1)
            # Sorteia o próximo token a partir de uma distribuição
            # multinomial. Aqui, obtemos apenas uma amostra (token)
            idx_net = torch.multinomial(probs, num_samples=1)
            # Concatena o novo token ao contexto de entrada
            idx = torch.cat((idx, idx_net), dim=1)
        return idx

def train():
    start = time.time()

    # Barra de progresso do treinamento
    progress_bar = tqdm(total=max_iters, desc="Progresso do treinamento")
    
    # Loop de treinamento para max_iters iterações
    for iter in range(max_iters):
        # Verifica se é hora de estimar a perda e exibir informações
        if iter % eval_interval == 0 or iter == max_iters-1:
            # Calcula e exibe as perdas
            losses = estimate_loss()

            print(f"\nIteração {iter}:",
                    f"Perda de Treino {losses['train']:.4f}", 
                    f"Perda de Validação {losses['val']:.4f}")
    
            # Gera novo texto a partir do modelo
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
            output_gerado = decode(
                model.generate(context, max_new_tokens=512)[0].tolist()
                )
            print(f"\nTexto gerado pelo modelo na iteração: {iter}")
            print(output_gerado)
        
            # Salva checkpoint do modelo
            caminho_salvamento_modelo = f"modelo_{iter}.pth"
            torch.save(model.state_dict(), caminho_salvamento_modelo)

        # Obtém lotes de dados de treinamento
        xb, yb = get_batch("train")
        # Realiza o passo à frente do modelo, calculando os logits e loss
        logits, loss = model(xb, yb)
        # Zera os gradientes dos parâmetros do modelo
        optimizer.zero_grad(set_to_none=True)
        # Calcula os gradientes da perda em relação aos parâmetros 
        # Backpropagation
        loss.backward()
        # Atualiza os parâmetros do modelo usando o otimizador
        optimizer.step()

        # Atualiza barra de progresso
        progress_bar.update(1)
    
    # Encerra barra de progresso
    progress_bar.close()
    
    end = time.time()
    total_time = end - start
    print(f"\nTempo total de treinamento em segundos: {float(total_time):.2f}")
    print("\n")

# Inicializa o modelo
model = GPTLanguageModel()
# Move o modelo para o dispositivo apropriado (GPU/CPU)
m = model.to(device)
# Configura o otimizador com taxa de aprendizagem específica
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Calcula o número de parâmetros
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nNúmero total de parâmetros: {total_params}")

print("\nInício do processo de treinamento...\n")
train()
print("\nTreinamento concluído!")

print("\n###################### Fim do programa ######################")