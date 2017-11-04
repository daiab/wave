

a = '2[2[2adc]4g]5ff#'
stack = []
tmp_str = ''
total_str = ''
tmp_num = None
for c in a:
    try:
        c = int(c)
        if tmp_str and tmp_num:
            total_str += tmp_str * tmp_num
        tmp_str = ''
        tmp_num = c
    except:
        if c == '[':
            if total_str:
                stack.append(total_str)
            stack.append(tmp_num)
            tmp_str = total_str = ''
            tmp_num = None
        elif c == ']' or c == '#':
            if tmp_str and tmp_num:
                total_str += tmp_str * tmp_num
            if stack:
                repeat_num = stack.pop(-1)
                total_str *= repeat_num
            tmp_str = ''
            tmp_num = None
        else:
            tmp_str += c


while len(stack) > 0:
    ele = stack.pop(-1)
    if isinstance(ele, str):
        total_str = ele + total_str
    else:
        total_str *= ele

print(total_str)


