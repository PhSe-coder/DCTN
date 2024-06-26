TAGS = ['O', 'B-POS', 'B-NEG', 'B-NEU', 'I-POS', 'I-NEG', 'I-NEU']
SUPPORTED_MODELS = ('bert', 'mmt')
# POS_TAGS = [
#     "[CLS]", "[SEP]", "[MASK]", "[UNK]", "[PAD]", "ADD", "AFX", "CC", "CD", "DT", "EX", "FW",
#     "HYPH", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NFP", "NN", "NNS", "NP", "NNP", "NNPS", "NPS",
#     "PDT", "POS", "PP", "PRP", "PRP$", "PP$", "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "VB",
#     "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB", "GW", ".", ",", "``", "''", ":",
#     "-RRB-", "-LRB-", "$"
# ]
# POS_DICT = {
#     "[PAD]": 0,
#     "[CLS]": 1,
#     "[SEP]": 2,
#     "[MASK]": 3,
#     "[UNK]": 4,
#     "O": 5,
#     "CC": 6,
#     "CD": 7,
#     "DT": 8,
#     "IN": 9,
#     "JJ": 10,
#     "MD": 11,
#     "NN": 12,
#     "NNS": 13,
#     "NNP": 14,
#     "PRP": 15,
#     "PRP$": 16,
#     "RB": 17,
#     "TO": 18,
#     "VB": 19,
#     "VBD": 20,
#     "VBG": 21,
#     "VBN": 22,
#     "VBP": 23,
#     "VBZ": 24,
#     ".": 25,
#     ",": 26
# }
POS_DICT = {
    "[PAD]": 0,
    "NN": 1,
    "NNS": 2,
    "NNP": 3,
    "JJ": 4,
    "O": 5,
}
DEPREL_DICT = {
    "[PAD]": 0,
    "nsubj": 1,
    "compound": 2,
    "obj": 3,
    "obl": 4,
    "conj": 5,
    "nmod": 6,
    "amod": 7,
    "root": 8,
    "punct": 9,
    "O": 10
}
DEPREL_TAGS = [
    "[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]", 'nsubj', 'nsubj:pass', 'nsubj:outer', 'obj',
    'iboj', 'csubj', 'csubj:pass', 'csubj:outer', 'ccomp', 'xcomp', 'obl', 'obl:npmod', 'obl:tmod',
    'advcl', 'advcl:relcl', 'advmod', 'vocative', 'aux', 'mark', 'discourse', 'aux:pass', 'expl',
    'cop', 'nummod', 'acl', 'amod', 'acl:relcl', 'appos', 'det', 'det:predet', 'nmod', 'nmod:npmod',
    'nmod:tmod', 'nmod:poss', 'compound', 'flat', 'compound:prt', 'flat:foreign', 'fixed',
    'goeswith', 'conj', 'cc', 'cc:preconj', 'case', 'list', 'parataxis', 'orphan', 'dislocated',
    'reparandum', 'root', 'punct', 'dep'
]
