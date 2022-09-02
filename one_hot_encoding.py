# -*- coding: UTF-8 -*-
import numpy as np
import captcha_setting

#实际模型训练当中不要传入小写字母

def encode(text):
    vector = np.zeros(captcha_setting.ALL_CHAR_SET_LEN * captcha_setting.MAX_CAPTCHA, dtype=float)
    def char2pos(c):
        if c =='_':
            k = 62
            return k
        elif c == '#':
            return 36
        elif c == '!':
            return 37
        elif c=='+':
            return 38
        elif c == '-':
            return 39
        elif c == '=':
            return 40
        else:
            pass
        
        k = ord(c)-48#判读是不是数字
        if k > 9:#True表明是字符
            k = ord(c) - 65 + 10
            if k > 35:#判断是不是大写
                k = ord(c) - 97 + 26 + 10
                if k > 61:
                    raise ValueError('error')
        return k
    for i, c in enumerate(text):
        idx = i * captcha_setting.ALL_CHAR_SET_LEN + char2pos(c)
        vector[idx] = 1.0
    return vector

def decode(vec):
    char_pos = vec.nonzero()[0]
    text=[]
    for i, c in enumerate(char_pos):
        char_at_pos = i #c/63
        char_idx = c % captcha_setting.ALL_CHAR_SET_LEN
        
        if char_idx ==36:
            text.append("#")
            continue
        elif char_idx == 37:
            text.append('!')
            continue
        elif char_idx == 38:
            text.append('+')
            continue
        elif char_idx == 39:
            text.append('-')
            continue
        elif char_idx == 40:
            text.append('=')
            continue
        else:
            pass
        
        if char_idx < 10:
            char_code = char_idx + ord('0')
        elif char_idx <36:
            char_code = char_idx - 10 + ord('A')
        elif char_idx < 62:
            char_code = char_idx - 36 + ord('a')
        elif char_idx == 62:
            char_code = ord('_')
        else:
            raise ValueError('error')
        text.append(chr(char_code))
    return "".join(text)

if __name__ == '__main__':
    e = encode("BK7=")
    print(decode(e))