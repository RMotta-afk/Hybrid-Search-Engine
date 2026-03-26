import hashlib


def chunk_document(text: str, chunk_size: int = 500, overlap: int = 100) -> list[dict]:
    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            chunk_text = text[start:]
        else:
            window = text[start:end]
            # Find last sentence boundary (period) past 50% of chunk_size
            midpoint = chunk_size // 2
            last_period = -1
            for i in range(len(window) - 1, midpoint - 1, -1):
                if window[i] == '.' and (i + 1 >= len(window) or window[i + 1] == ' '):
                    last_period = i
                    break
            if last_period != -1:
                chunk_text = window[:last_period + 1]
            else:
                chunk_text = window

        chunk_text = chunk_text.strip()
        if chunk_text:
            end_pos = start + len(chunk_text)
            chunks.append({
                "id": hashlib.md5(chunk_text.encode()).hexdigest(),
                "text": chunk_text,
                "chunk_index": chunk_index,
                "start": start,
                "end": end_pos,
            })
            chunk_index += 1
            advance = max(1, len(chunk_text) - overlap)
            start += advance
        else:
            start += 1

    return chunks
