def getKey(item):
    return item[0]

def gen_huffman_tree(text):
    text_list = [c for c in text]
    alphabet = set(text_list)
    h_tree = [ (text_list.count(c), (c, ) ) for c in alphabet]
    h_tree = sorted(h_tree, key=getKey)
    while len(h_tree) >= 2:
        h_tree = [(h_tree[0][0] + h_tree[1][0], (h_tree[0], h_tree[1]))] + h_tree[2:]
        h_tree = sorted(h_tree, key=getKey)
    return h_tree

def gen_huffman_dict(t, code, h_dict):
    if len(t[1]) == 1:
        h_dict[t[1][0]] = code
    else:
        gen_huffman_dict(t[1][0], code+'0', h_dict)
        gen_huffman_dict(t[1][1], code+'1', h_dict)
    return

def huffman_enc(h_dict, text):
    h_code = ''
    for t in text:
        h_code = h_code + h_dict[t]
    return h_code

def huffman_dec(h_tree, h_code):
    text = ''
    i = 0
    while i < len(h_code):
        t_ptr = h_tree[0]
        while len(t_ptr[1]) == 2:
            t_ptr = t_ptr[1][0] if h_code[i] == '0' else t_ptr[1][1]
            i += 1
        text = text + t_ptr[1][0]
            
    return text

def huffman_dec_list(h_tree, h_code):
    text = []
    i = 0
    while i < len(h_code):
        t_ptr = h_tree[0]
        while len(t_ptr[1]) == 2:
            t_ptr = t_ptr[1][0] if h_code[i] == '0' else t_ptr[1][1]
            i += 1
        text.append(t_ptr[1][0])
            
    return text
