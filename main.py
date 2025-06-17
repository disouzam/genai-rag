"""
Este módulo processa um PDF, gera embeddings para seus textos, cria um índice FAISS para busca vetorial,
e responde a perguntas do usuário utilizando a API da OpenAI.
"""

import os
import re
import logging
import pickle
from typing import List

import numpy as np
import faiss
import PyPDF2
from dotenv import load_dotenv
from openai import OpenAI


def main():
    """
    Função principal que configura o logging, carrega variáveis de ambiente, processa o PDF,
    gera embeddings, cria o índice FAISS e responde a perguntas do usuário.
    """
    # Configuração do logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    # Carregar variáveis de ambiente
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.error("Chave da API da OpenAI não encontrada. Defina OPENAI_API_KEY no seu arquivo .env.")
        return

    # Criar o cliente OpenAI
    client = OpenAI(api_key=api_key, max_retries=10)

    embeddings, chunks, index = load_embeddings()
    if embeddings is None:
        logging.info("Embeddings não encontrados. Processando PDF e criando embeddings...")
        pdf_path = 'pdfs/manual_de_normalizacao_abnt.pdf'
        text = extract_text_from_pdf(pdf_path)
        chunks = split_text_into_chunks(text)
        embeddings = get_embeddings(chunks, client)
        index = create_faiss_index(embeddings)
        save_embeddings(embeddings, chunks, index)
        logging.info("Embeddings e índice salvos.")
    else:
        logging.info("Embeddings carregados dos arquivos.")

    print("Digite sua pergunta (ou 'sair' para terminar):")
    while True:
        query = input(">> ")
        if query.lower() == 'sair':
            break
        answer = answer_query(query, index, chunks, client)
        print("\nResposta:\n", answer)


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extrai o texto de um arquivo PDF.

    Args:
        pdf_path (str): Caminho para o arquivo PDF.

    Returns:
        str: Texto extraído do PDF.
    """
    logging.info(f"Extraindo texto do PDF: {pdf_path}")
    text = ''
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
    except Exception as e:
        logging.error(f"Erro ao ler o arquivo PDF: {e}")
    return text


def split_text_into_chunks(text: str, max_chunk_size: int = 5000) -> List[str]:
    """
    Divide o texto em chunks menores.

    Args:
        text (str): Texto a ser dividido.
        max_chunk_size (int): Tamanho máximo de cada chunk.

    Returns:
        List[str]: Lista de chunks de texto.
    """
    logging.info("Dividindo o texto em chunks.")
    sentences = re.split(r'(?<=[.?!])\s+', text)
    chunks = []
    current_chunk = ''
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += ' ' + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    logging.info(f"Total de chunks criados: {len(chunks)}")
    return chunks


def get_embedding(text: str, client, model: str = "text-embedding-3-small") -> List[float]:
    """
    Gera um embedding para um texto usando a API da OpenAI.

    Args:
        text (str): Texto para o qual o embedding será gerado.
        client: Cliente da API da OpenAI.
        model (str): Modelo de embedding a ser utilizado.

    Returns:
        List[float]: Embedding gerado para o texto.
    """
    text = text.replace("\n", " ")
    try:
        response = client.embeddings.create(input=[text], model=model)
        embedding = response.data[0].embedding
        return embedding
    except Exception as e:
        logging.error(f"Erro ao obter embedding para o texto: {e}")
        return []


def get_embeddings(texts: List[str], client, model: str = "text-embedding-3-small") -> List[List[float]]:
    """
    Gera embeddings para uma lista de textos.

    Args:
        texts (List[str]): Lista de textos.
        client: Cliente da API da OpenAI.
        model (str): Modelo de embedding a ser utilizado.

    Returns:
        List[List[float]]: Lista de embeddings gerados.
    """
    embeddings = []
    logging.info("Gerando embeddings para os chunks.")
    for i, text in enumerate(texts):
        embedding = get_embedding(text, client, model)
        embeddings.append(embedding)
        if (i + 1) % 10 == 0 or (i + 1) == len(texts):
            logging.info(f"Processados {i + 1}/{len(texts)} chunks.")
    return embeddings


def create_faiss_index(embeddings: List[List[float]]) -> faiss.IndexFlatL2:
    """
    Cria um índice FAISS a partir de uma lista de embeddings.

    Args:
        embeddings (List[List[float]]): Lista de embeddings.

    Returns:
        faiss.IndexFlatL2: Índice FAISS criado.
    """
    logging.info("Criando índice FAISS.")
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    return index


def save_embeddings(embeddings: List[List[float]], chunks: List[str], index: faiss.IndexFlatL2,
                    embeddings_file: str = 'embeddings.pkl',
                    chunks_file: str = 'chunks.pkl',
                    index_file: str = 'faiss.index'):
    """
    Salva embeddings, chunks e índice FAISS no disco.

    Args:
        embeddings (List[List[float]]): Lista de embeddings.
        chunks (List[str]): Lista de chunks de texto.
        index (faiss.IndexFlatL2): Índice FAISS.
        embeddings_file (str): Caminho para o arquivo de embeddings.
        chunks_file (str): Caminho para o arquivo de chunks.
        index_file (str): Caminho para o arquivo de índice FAISS.
    """
    logging.info("Salvando embeddings, chunks e índice no disco.")
    with open(embeddings_file, 'wb') as f:
        pickle.dump(embeddings, f)
    with open(chunks_file, 'wb') as f:
        pickle.dump(chunks, f)
    faiss.write_index(index, index_file)


def load_embeddings(embeddings_file: str = 'embeddings.pkl',
                    chunks_file: str = 'chunks.pkl',
                    index_file: str = 'faiss.index'):
    """
    Carrega embeddings, chunks e índice FAISS do disco.

    Args:
        embeddings_file (str): Caminho para o arquivo de embeddings.
        chunks_file (str): Caminho para o arquivo de chunks.
        index_file (str): Caminho para o arquivo de índice FAISS.

    Returns:
        Tuple: Embeddings, chunks e índice FAISS carregados.
    """
    if os.path.exists(embeddings_file) and os.path.exists(chunks_file) and os.path.exists(index_file):
        logging.info("Carregando embeddings, chunks e índice do disco.")
        with open(embeddings_file, 'rb') as f:
            embeddings = pickle.load(f)
        with open(chunks_file, 'rb') as f:
            chunks = pickle.load(f)
        index = faiss.read_index(index_file)
        return embeddings, chunks, index
    else:
        logging.warning("Arquivos de embeddings não encontrados.")
        return None, None, None


def search_index(index: faiss.IndexFlatL2, query_embedding: List[float], k: int = 5):
    """
    Pesquisa no índice FAISS por embeddings similares.

    Args:
        index (faiss.IndexFlatL2): Índice FAISS.
        query_embedding (List[float]): Embedding da consulta.
        k (int): Número de resultados a serem retornados.

    Returns:
        Tuple: Índices e distâncias dos embeddings similares.
    """
    logging.info("Pesquisando no índice FAISS por embeddings similares.")
    query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)
    distances, indices = index.search(query_embedding, k)
    return indices[0], distances[0]


def answer_query(query: str, index: faiss.IndexFlatL2, chunks: List[str], client, k: int = 5) -> str:
    """
    Responde a uma pergunta do usuário utilizando embeddings e a API da OpenAI.

    Args:
        query (str): Pergunta do usuário.
        index (faiss.IndexFlatL2): Índice FAISS.
        chunks (List[str]): Lista de chunks de texto.
        client: Cliente da API da OpenAI.
        k (int): Número de chunks relevantes a serem utilizados.

    Returns:
        str: Resposta gerada para a pergunta.
    """
    logging.info("Respondendo à pergunta do usuário.")
    query_embedding = get_embedding(query, client)
    indices, distances = search_index(index, query_embedding, k)
    relevant_chunks = [chunks[i] for i in indices]
    context = '\n\n'.join(relevant_chunks)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Você é um assistente que ajuda com perguntas sobre o manual de normalização ABNT."},
                {"role": "system", "content": "Contexto:\n{context}\n\n"},
                {"role": "user", "content": f"Pergunta: {query}"}
            ],
            temperature=1
        )
        answer = response.choices[0].message.content
        return answer
    except Exception as e:
        logging.error(f"Erro ao gerar a resposta: {e}")
        return "Desculpe, ocorreu um erro ao gerar a resposta."


if __name__ == "__main__":
    main()
