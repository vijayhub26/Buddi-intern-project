import re
text = 'Customreceipt detailswithin offerends monthuntilyou tochange Pricesaresubjecttochange Cancel.Seehere Invoice'
glue = r'(to|for|with|within|from|are|and|the|this|until|you|month|ends|receipt|custom|prices|subject|details|offer|see|here)'
text = re.sub(rf'\b{glue}([a-zA-Z]{{3,}})\b', r'\1 \2', text, flags=re.IGNORECASE)
text = re.sub(rf'\b([a-zA-Z]{{3,}}){glue}\b', r'\1 \2', text, flags=re.IGNORECASE)
# Do another pass to catch deep nested things like 'Pricesaresubjecttochange' -> 'Prices are subject to change'
text = re.sub(rf'\b{glue}([a-zA-Z]{{3,}})\b', r'\1 \2', text, flags=re.IGNORECASE)
text = re.sub(rf'\b([a-zA-Z]{{3,}}){glue}\b', r'\1 \2', text, flags=re.IGNORECASE)
print(text)
