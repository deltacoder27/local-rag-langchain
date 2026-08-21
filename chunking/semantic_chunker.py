from langchain_community.utils.math import cosine_similarity

def create_semantic_chunks(chunk_embeddings, chunks):
    chunk_list = []
    similarities = cosine_similarity(chunk_embeddings, chunk_embeddings)
    current_text = chunks[0]

    for i in range(1, len(chunks)):
        next_text = chunks[i]

        similarity = similarities[i-1][i]

        if similarity > 0.4:
            potential_text = current_text + " " + next_text

            if len(potential_text) <= 100:
                current_text = potential_text
            else:
                chunk_list.append(current_text)
                current_text = next_text

        else:
            chunk_list.append(current_text)
            current_text = next_text
    chunk_list.append(current_text)
            
    return chunk_list