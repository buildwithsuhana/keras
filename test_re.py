import re
pattern = ".*kernel"
key = "dense/kernel"
print(f"fullmatch: {re.fullmatch(pattern, key)}")
