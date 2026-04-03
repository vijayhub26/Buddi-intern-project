import re

words = ['tochange', 'Pricesare', 'detailswithin', 'offerends', 'Customreceipt', 'monthuntilyou', 'Invoice']
glue_words = ['to', 'for', 'with', 'within', 'from', 'are', 'and', 'the', 'this', 'until', 'you', 'month', 'ends', 'receipt', 'Custom']

prefix_pattern = r'^(' + '|'.join(glue_words) + r')([a-z]{4,})$'
suffix_pattern = r'^([a-z]{4,})(' + '|'.join(glue_words) + r')$'

for w in words:
    w_low = w.lower()
    if re.match(prefix_pattern, w_low, re.I):
        res = re.sub(prefix_pattern, r"\1 \2", w, flags=re.I)
        print(w, "-> PREFIX MATCH:", res)
    elif re.match(suffix_pattern, w_low, re.I):
        res = re.sub(suffix_pattern, r"\1 \2", w, flags=re.I)
        print(w, "-> SUFFIX MATCH:", res)
    else:
        print(w, "-> NO MATCH")
