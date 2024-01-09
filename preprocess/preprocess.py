import logging
from glob import glob
import os.path as osp
import os
import nltk
from nltk.tokenize import word_tokenize
nltk.download("punkt")
sentis = {
    "1": "T-POS",
    "0": "T-NEU",
    "-1": "T-NEG"
}

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info("Split the sentence having multiple aspect terms")
SEP = "^^^^^^^^^^^"
for file in glob("data/raw/*.raw"):
    dirname = osp.dirname(file)
    file_name = osp.splitext(osp.basename(file))[0]
    fw = open(osp.join(dirname, file_name + ".txt"), "w")
    with open(file, "r") as f:
        while True:
            line, aspect, polarity = f.readline().strip(), f.readline().strip(), f.readline().strip()
            if not line: break
            line = line.replace("$T$", SEP)
            line = ' '.join(word_tokenize(line))
            aspect = ' '.join(word_tokenize(aspect))
            text = line.replace(SEP, aspect)
            words = line.split(' ')
            anns = ['O'] * len(words)
            for i, word in enumerate(words):
                if word != SEP: continue
                anns[i] = ' '.join([sentis[polarity]] * len(aspect.split(' ')))
            assert len(words) == len(anns)
            fw.write(f"{text}***{' '.join(anns)}\n")
    fw.close()
os.makedirs("data/dataset", exist_ok=True)
for file in glob("data/raw/*.raw"):
    dirname = osp.join(osp.split(osp.dirname(file))[0], "dataset")
    file_name = osp.splitext(osp.basename(file))[0]
    fw = open(osp.join(dirname, file_name + ".txt"), "w")
    with open(file, "r") as f:
        while True:
            line, aspect, polarity = f.readline().strip(), f.readline().strip(), f.readline().strip()
            if not line: break
            line = line.replace("$T$", SEP)
            line = ' '.join(word_tokenize(line))
            aspect = ' '.join(word_tokenize(aspect))
            text = line.replace(SEP, aspect)
            words = line.split(' ')
            anns = ['O'] * len(words)
            for i in range(len(words)):
                if words[i] == SEP:
                    anns[i] = ' '.join(['O' for _ in aspect.split(' ')])
            for i, word in enumerate(words):
                if word != SEP: continue
                temp = anns[i]
                anns[i] = ' '.join([sentis[polarity] for _ in aspect.split(' ')])
                fw.write(f"{text}***{' '.join(anns)}\n")
                anns[i] = temp
    fw.close()
# for file in glob("data/raw/laptop.*") + glob("data/raw/restaurant.*"):
#     dirname = osp.join(osp.split(osp.dirname(file))[0], "dataset")
#     fw = open(osp.join(dirname, osp.basename(file)), "w")
#     with open(file, "r") as f:
#         for line in f:
#             text, ann = line.rsplit("***", maxsplit=1)
#             anns = ann.strip().split(' ')
#             count = 0
#             for i in range(0, len(anns)):
#                 if anns[i] != 'O': count += 1
#                 if (i == len(anns)-1 and anns[i] != 'O') or (i < len(anns)-1 and anns[i] != 'O' and anns[i+1] == 'O'):
#                     s = ' '.join(['O']*(i-count+1)+[anns[i]]*count+['O']*(len(anns)-i-1))
#                     assert len(s.split(' ')) == len(anns)
#                     fw.write(f"{text}***{s}\n")
#                     count = 0
#     fw.close()