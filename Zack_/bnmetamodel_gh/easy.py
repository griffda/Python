import BN_Metamodel_easy

csvfilepath = '/Users/tomgriffiths/OneDrive - Imperial College London/Research/Python/gitlibraries/Python/output2.csv'

targets = ['vf', 'KE']
b=BN_Metamodel_easy.BN_Metamodel_easy(csvdata=csvfilepath, targets=targets)
bn=b.generate()
query = {'vf':0, 'KE':0,}
evidence = {'m':[1.0,0.0,0,0], 'theta':[1.0,0.0,0,0], 'v0':[1.0,0.0,0,0]}
a, posteriors = bn.inferPD_3(query, evidence )
print (a)
print (posteriors)
bn.plotPDs('Posteriors', 'Ranges', 'Probabilities', displayplt=True, posteriorPD=posteriors, evidence=evidence.keys())