from datasets import load_dataset
from stanza_utils import annotation_plus

for split in ("train", "test"):
    for domain in ("restaurant", "laptop"):
        dataset = load_dataset(f"yqzheng/semeval2014_{domain}s", split=split)

        content = []

        sentences = annotation_plus([line["text"] for line in dataset])
        senti_map = {-1: "NEG", 0: "NEU", 1: "POS"}
        for data, sentence in zip(dataset, sentences):
            tokens = sentence.tokens
            text = ' '.join([token.text for token in tokens])
            tags = []
            for token in tokens:
                if token.start_char < data["start"] or token.end_char > data["end"]:
                    tags.append("O")
                else:
                    tags.append(f"T-{senti_map[data['label']]}")
            assert len(tags) == len(tokens)
            labels = ' '.join(tags)
            content.append(f"{text}***{labels}\n")
        with open(f"{domain}.{split}.txt", "w") as f:
            f.writelines(content)
