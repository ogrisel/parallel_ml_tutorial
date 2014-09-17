def strip_headers(post):
    """Find the first blank line and drop the headers to keep the body"""
    if '\n\n' in post:
        headers, body = post.split('\n\n', 1)
        return body.lower()
    else:
        # Unexpected post inner-structure, be conservative
        # and keep everything
        return post.lower()


print("#" * 72)
print("Original text:\n\n")
original_text = all_twenty_train.data[0]
print(original_text)

print("#" * 72)
print("Stripped headers text:\n\n")
text_body = strip_headers(original_text)
print(text_body)
