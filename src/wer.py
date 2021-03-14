import sys
import os
import numpy


def editDistance(r, h):
    '''
    This function is to calculate the edit distance of reference sentence and the hypothesis sentence.
    Main algorithm used is dynamic programming.
    Attributes:
        r -> the list of words produced by splitting reference sentence.
        h -> the list of words produced by splitting hypothesis sentence.
    '''
    d = numpy.zeros((len(r) + 1) * (len(h) + 1), dtype=numpy.uint8).reshape((len(r) + 1, len(h) + 1))
    for i in range(len(r) + 1):
        for j in range(len(h) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitute = d[i - 1][j - 1] + 1
                insert = d[i][j - 1] + 1
                delete = d[i - 1][j] + 1
                d[i][j] = min(substitute, insert, delete)
    return d


def getStepList(r, h, d):
    '''
    This function is to get the list of steps in the process of dynamic programming.
    Attributes:
        r -> the list of words produced by splitting reference sentence.
        h -> the list of words produced by splitting hypothesis sentence.
        d -> the matrix built when calulating the editting distance of h and r.
    '''
    x = len(r)
    y = len(h)
    wer_list = []
    match_list = []
    while True:
        if x == 0 and y == 0:
            break
        elif x >= 1 and y >= 1 and d[x][y] == d[x - 1][y - 1] and r[x - 1] == h[y - 1]:
            wer_list.append("e")
            x = x - 1
            y = y - 1
        elif y >= 1 and d[x][y] == d[x][y - 1] + 1:
            wer_list.append("i")
            x = x
            y = y - 1
        elif x >= 1 and y >= 1 and d[x][y] == d[x - 1][y - 1] + 1:
            wer_list.append("s")
            match_list.append((x - 1, y - 1))
            x = x - 1
            y = y - 1
        else:
            wer_list.append("d")
            x = x - 1
            y = y
    return wer_list[::-1], match_list


def alignedPrint(wer_list, r, h, result):
    '''
    This funcition is to print the result of comparing reference and hypothesis sentences in an aligned way.

    Attributes:
        list   -> the list of steps.
        r      -> the list of words produced by splitting reference sentence.
        h      -> the list of words produced by splitting hypothesis sentence.
        result -> the rate calculated based on edit distance.
    '''
    ref = []
    hyp = []
    error_type = []
    for i in range(len(wer_list)):
        if wer_list[i] == "i":
            count = 0
            for j in range(i):
                if wer_list[j] == "d":
                    count += 1
            index = i - count
            ref += " " * (len(h[index])),
        elif wer_list[i] == "s":
            count1 = 0
            for j in range(i):
                if wer_list[j] == "i":
                    count1 += 1
            index1 = i - count1
            count2 = 0
            for j in range(i):
                if wer_list[j] == "d":
                    count2 += 1
            index2 = i - count2
            if len(r[index1]) < len(h[index2]):
                ref += r[index1] + " " * (len(h[index2]) - len(r[index1])),
            else:
                ref += r[index1],
        else:
            count = 0
            for j in range(i):
                if wer_list[j] == "i":
                    count += 1
            index = i - count
            ref += r[index],

    # print "HYP:",
    for i in range(len(wer_list)):
        if wer_list[i] == "d":
            count = 0
            for j in range(i):
                if wer_list[j] == "i":
                    count += 1
            index = i - count
            hyp += " " * (len(r[index])),
        elif wer_list[i] == "s":
            count1 = 0
            for j in range(i):
                if wer_list[j] == "i":
                    count1 += 1
            index1 = i - count1
            count2 = 0
            for j in range(i):
                if wer_list[j] == "d":
                    count2 += 1
            index2 = i - count2
            if len(r[index1]) > len(h[index2]):
                hyp += h[index2] + " " * (len(r[index1]) - len(h[index2])),
            else:
                hyp += h[index2],
        else:
            count = 0
            for j in range(i):
                if wer_list[j] == "d":
                    count += 1
            index = i - count
            hyp += h[index],

    # print "EVA:",
    for i in range(len(wer_list)):
        if wer_list[i] == "d":
            count = 0
            for j in range(i):
                if wer_list[j] == "i":
                    count += 1
            index = i - count
            error_type += "D" + " " * (len(r[index]) - 1),
        elif wer_list[i] == "i":
            count = 0
            for j in range(i):
                if wer_list[j] == "d":
                    count += 1
            index = i - count
            error_type += "I" + " " * (len(h[index]) - 1),
        elif wer_list[i] == "s":
            count1 = 0
            for j in range(i):
                if wer_list[j] == "i":
                    count1 += 1
            index1 = i - count1
            count2 = 0
            for j in range(i):
                if wer_list[j] == "d":
                    count2 += 1
            index2 = i - count2
            if len(r[index1]) > len(h[index2]):
                error_type += "S" + " " * (len(r[index1]) - 1),
            else:
                error_type += "S" + " " * (len(h[index2]) - 1),
        else:
            count = 0
            for j in range(i):
                if wer_list[j] == "i":
                    count += 1
            index = i - count
            error_type += " " * (len(r[index])),
    return ' '.join(ref), ' '.join(hyp), ' '.join(error_type)


def wer(r, h):
    """
    This is a function that calculate the word error rate in ASR.
    You can use it like this: wer("what is it".split(), "what is".split())
    """
    # build the matrix
    d = editDistance(r, h)
    # wer_list, match_list = getStepList(r, h, d)
    # ref, hyp, error_type = alignedPrint(wer_list, r, h, 0)
    return d[len(r)][len(h)], len(r) #, ref, hyp, error_type


def main():
    ref_f = open(sys.argv[1],'r')
    hyp_f = open(sys.argv[2],'r')
    index = 0
    total_error = 0
    total_ref = 0

    for line1, line2 in zip(ref_f, hyp_f):
        line1, line2 = line1.strip(), line2.strip()
      
        error_num, ref_len = wer(line1.split(), line2.split())
        # error_num, ref_len = wer(line1, line2)
        total_error += error_num
        total_ref += ref_len
    print('WER is {0}'.format(total_error*1.0/total_ref))


if __name__ == '__main__':
    main()

