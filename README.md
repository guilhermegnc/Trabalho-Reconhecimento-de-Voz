# Trabalho de Reconhecimento de Voz

Este repositório contém um projeto desenvolvido em Python para reconhecimento de voz, utilizando bibliotecas específicas para captura e processamento de áudio.

## Estrutura do Repositório

- **`data/recordings/`**: Diretório contendo gravações de áudio utilizadas no projeto.
- **`src/`**: Código-fonte do projeto.
- **`README.txt`**: Informações adicionais sobre o projeto.
- **`requirements.txt`**: Lista de dependências necessárias para executar o projeto.

## Tecnologias Utilizadas

- **Python 3.7.0**: Linguagem de programação utilizada no desenvolvimento.
- **Bibliotecas Python**: As bibliotecas necessárias estão listadas no arquivo `requirements.txt`.

## Instalação

1. **Clone o repositório**:
   ```bash
   git clone https://github.com/guilhermegnc/Trabalho-Reconhecimento-de-Voz.git
2. **Navegue até o diretório do projeto**:
    ```bash
    cd Trabalho-Reconhecimento-de-Voz
3. **Crie um ambiente virtual (opcional, mas recomendado)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows: venv\Scripts\activate
4. **Instale as dependências**:
    ```bash
    pip install -r requirements.txt
  > Nota: Além das bibliotecas listadas no requirements.txt, é necessário instalar a biblioteca PyAudio. Devido a questões de compatibilidade, recomenda-se instalá-la utilizando o pipwin:

    ```bash
    pip install pipwin
    pipwin install pyaudio
    
## Execução

1. **Navegue até o diretório src/**:
    ```bash
    cd src
2. **Execute o script principal**:
    ```bash
    python main.py
  > Nota: Substitua main.py pelo nome do arquivo principal do projeto, caso seja diferente.

## Gravações de Áudio

- As gravações de áudio são armazenadas no diretório data/recordings/.
- Foram adicionados alguns arquivos de áudio nas pastas data/recordings/moreSpeakersTest e data/recordings/moreSpeakersTraining.
- As gravações realizadas através do programa AudioRecorder.py também são salvas neste diretório.
  
## Contato

Para mais informações, entre em contato comigo através do e-mail: guilhermegnog24@gmail.com
