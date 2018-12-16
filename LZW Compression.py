def lzw_enc(text):
    
    d = {chr(i):i for i in range(256)}
    curr_code = 256
    zipped_code = []
    s = ""
    for c in text:
        if s+c in d:
            s = s+c
        else:
            zipped_code.append(d[s])
            d[s+c] = curr_code
            curr_code += 1
            s = c
    zipped_code.append(d[s])
            
    return zipped_code
        
text = "National Tsing Hua University" * 100
z = lzw_enc(text)
len(z)
def lzw_dec(z):
    d = {i:chr(i) for i in range(256)}
    dd = {chr(i): i for i in range(256)}
    text = ""
    s = ""
    curr_code = 256
    for code in z:
        if code in d:
            decoded = d[code]
            text += decoded
            if s + decoded[0] not in dd:
                d[curr_code] = s + decoded[0]
                dd[s + decoded[0]] = curr_code
                curr_code += 1
            s = decoded
        else:
            decoded = s + s[0]
            text += decoded
            d[curr_code] = s + decoded[0]
            dd[s + decoded[0]] = curr_code
            curr_code += 1
            s = decoded
            
    return text
    
lzw_dec(z)
