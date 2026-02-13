def simple_rerank(query, chunks):
    scored = []

    for chunk in chunks:
        score = sum(word.lower() in chunk.lower() for word in query.split())
        scored.append((score, chunk))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [chunk for _, chunk in scored]
