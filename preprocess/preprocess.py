from glob import glob
import os.path as osp

sentis = {
    "1": "T-POS",
    "0": "T-NEU",
    "-1": "T-NEG"
}
for file in glob("FDGR/data/twitter.*"):
    dirname = osp.join(osp.dirname(file), "dataset")
    fw = open(osp.join(dirname, osp.basename(file)), "w")
    with open(file, "r") as f:
        while True:
            line, aspect, polarity = f.readline().strip(), f.readline().strip(), f.readline().strip()
            if not line: break
            text = line.replace("$T$", aspect)
            words = line.split(' ')
            anns = ['O'] * len(words)
            for i, word in enumerate(words):
                if word != "$T$": continue
                anns[i] = ' '.join([sentis[polarity] for _ in aspect.split(' ')])
                fw.write(f"{text}***{' '.join(anns)}\n")
                anns[i] = 'O'
    fw.close()
for file in glob("FDGR/data/laptop.*") + glob("FDGR/data/restaurant.*"):
    dirname = osp.join(osp.dirname(file), "dataset")
    fw = open(osp.join(dirname, osp.basename(file)), "w")
    with open(file, "r") as f:
        for line in f:
            text, ann = line.rsplit("***", maxsplit=1)
            anns = ann.strip().split(' ')
            count = 0
            for i in range(0, len(anns)):
                if anns[i] != 'O': count += 1
                if (i == len(anns)-1 and anns[i] != 'O') or (i < len(anns)-1 and anns[i] != 'O' and anns[i+1] == 'O'):
                    s = ' '.join(['O']*(i-count+1)+[anns[i]]*count+['O']*(len(anns)-i-1))
                    assert len(s.split(' ')) == len(anns)
                    fw.write(f"{text}***{s}\n")
                    count = 0
    fw.close()