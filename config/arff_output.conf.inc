[componentInstances:cComponentManager]
instance[arffsink].type=cArffSink

[arffsink:cArffSink]
reader.dmLevel=output
frameIndex=0
frameTime=0
filename=\cm[arffout(O){output.arff}:name of WEKA Arff output file]
relation=\cm[corpus{SMILEfeaturesLive}:corpus name, arff relation]
instanceName=\cm[instname(N){noname}:name of arff instance]
class[0].name = emotion
class[0].type = \cm[classes{unknown}:all classes for arff file attribute]
target[0].all = \cm[class{unknown}:instance class label]
append=0
