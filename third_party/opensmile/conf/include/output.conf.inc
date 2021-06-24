[componentInstances:cComponentManager]
instance[arffsink].type=cArffSink
instance[csvsink].type=cCsvSink

[arffsink:cArffSink]
reader.dmLevel=output
frameIndex=0
frameTime=0
filename=\cm[output{?}:name of WEKA Arff output file]
relation=\cm[corpus{corpus}:corpus name, arff relation]
instanceName=\cm[instname(N){noname}:name of instance]
class[0].name = emotion
class[0].type = \cm[classes{unknown}:all classes for arff file attribute]
target[0].all = \cm[class{unknown}:instance class label]
append=0

[csvsink:cCsvSink]
reader.dmLevel=output
frameIndex=0
frameTime=0
filename=\cm[csvoutput{?}:output CSV file for summarised parameters, set to a valid filename to enable this output sink, data is appended if file exists]
delimChar=,
instanceName=\cm[instname(N){noname}:name of instance]
printHeader=1
append=0
