# Road Signs Detection System 🚦

Este projeto implementa um pipeline completo de detecção de objetos para sinalização de trânsito (Edge AI), utilizando **YOLOv11** e otimização com **TensorRT**, conforme os requisitos do Desafio Técnico de Engenharia de Software e Visão Computacional.

O foco da solução é a maximização do throughput através de **processamento em lote (Batch Processing)** e redução de latência com precisão **FP16**.

## 📂 Estrutura do Projeto

```text
.
├── main.py                 # Script principal (carrega engine, processa batch, salva vídeo)
├── README.md               # Documentação do projeto
├── requirements.txt        # Dependências do projeto
├── src
│   ├── detector.py         # Wrapper para carregar modelo e realizar inferência
│   ├── video_loader.py     # Gerenciamento de vídeo e criação de batches de frames
│   ├── visualizer.py       # Utilitários para desenhar bounding boxes
│   └── __init__.py
└── train_pipeline.ipynb    # Notebook de treino, validação e exportação para TensorRT
````

## 🛠️ Pré-requisitos

  * **Python 3.10+**
  * **GPU NVIDIA** (Necessário para suporte ao TensorRT e CUDA)
  * Drivers CUDA compatíveis instalados

## 🚀 Instalação

1.  **Clone o repositório:**

    ```bash
    git clone https://github.com/pauloricms12/road_signs
    cd road_signs
    ```

2.  **Crie e ative o ambiente virtual:**

    ```bash
    # Linux/Mac
    python3 -m venv .venv
    source .venv/bin/activate

    # Windows
    # .venv\Scripts\activate
    ```

3.  **Instale as dependências:**

    ```bash
    pip install -r requirements.txt
    ```

-----

## ⚙️ Como Executar

O fluxo de execução é dividido em duas etapas: **Preparação (Treino/Exportação)** e **Inferência**.

### 1\. Preparação do Modelo e Ambiente

Antes da inferência, é necessário rodar o pipeline de treinamento para baixar os dados, treinar o modelo base e exportá-lo para o formato otimizado `.engine`.

1.  Defina sua chave de API do Roboflow (necessária para baixar o dataset):

    ```bash
    # Linux/Mac
    export ROBOFLOW_API_KEY="SUA_CHAVE_AQUI"

    # Windows (Powershell)
    # $env:ROBOFLOW_API_KEY="SUA_CHAVE_AQUI"
    ```

2.  Execute o notebook `train_pipeline.ipynb`.

    Este notebook automatiza as seguintes tarefas:

      * Instalação do dataset **Roboflow 100: Road Signs**.
      * Download do videoclipe de amostra para inferência.
      * Treinamento do modelo **YOLOv11**.
      * Conversão do modelo para **TensorRT** com:
          * **Precisão:** FP16 (Half=True).
          * **Batch Size:** 16 (para processamento paralelo de frames).

### 2\. Execução da Inferência

Com o modelo otimizado gerado, execute o script principal:

```bash
python main.py
```

**Comportamento do script:**

  * Carrega o modelo TensorRT gerado no passo anterior.
  * Utiliza o `video_loader.py` para carregar o vídeo e agrupar os quadros em lotes de **16 frames**.
  * Realiza a inferência em lote (evitando processamento frame-a-frame).
  * Salva o vídeo resultante com as detecções visuais na pasta do projeto.

-----

## 📊 Otimizações Implementadas

  * **Arquitetura:** YOLOv11
  * **Inferência:** TensorRT (Formato `.engine`)
  * **Precision:** FP16 (Half-precision)
  * **Batch Processing:** 16 frames simultâneos

<!-- end list -->