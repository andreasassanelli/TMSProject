import glob
import json

reslist = glob.glob("redump_*_20_0.75.json")

scribe = open('results_models.csv','w')

scribe.write("head,quote,sign,spars,model,acc,trtime\n")

for f in reslist:

    with open(f,'r') as fin:
        res = json.load(fin)

    #pieces = f[11:-5].split('_')
    pieces = f[:-5].split('_')
    assert pieces[0]== 'redump'
    pieces.pop(0)

    head = bool(int(pieces[0][0]))
    quote = bool(int(pieces[0][1]))
    sign = bool(int(pieces[0][2]))
    #minDF = int(pieces[1])
    #maxDF = float(pieces[2])

    ndocs, nterms = res[0]['Shape']
    spar = res[0]['Sparsity']
    for m in res[1].keys():
        acc = res[1][m]['Acc']
        ttime = res[1][m]['Train_time']

        scribe.write("%s,%s,%s,%s,%s,%s,%s\n" % (head, quote, sign, spar, m, acc, ttime))
        print("%s,%s,%s,%s,%s,%s,%s\n" % (head, quote, sign, spar, m, acc, ttime))

scribe.close()