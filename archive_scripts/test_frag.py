import re

tests = [
    'vijay dm 26@gmail.com',
    'Contact me at john doe @ gmail .com today',
    'support team@company.com'
]

for text in tests:
    # First: Clean up spaces around @ and common TLDs
    text = re.sub(r'\s*@\s*', '@', text)
    text = re.sub(r'\s*\.\s*(com|net|org|co|in|edu)\b', r'.\1', text, flags=re.IGNORECASE)

    # 4. Clean fragmented usernames. Match purely alphanumeric chunks separated by spaces before @
    def email_clean(match):
        user = match.group(1)
        domain = match.group(2)
        # Avoid catching standard English phrases
        if any(w in user.lower() for w in [' at ', ' me ', ' contact ', ' reach ', ' email ']):
            return match.group(0)
        return user.replace(' ', '') + '@' + domain

    text = re.sub(r'\b((?:[a-zA-Z0-9._-]+\s+){1,3}[a-zA-Z0-9._-]+)@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b', email_clean, text)
    print("->", text)
