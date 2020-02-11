[componentInstances:cComponentManager]
instance[waveIn].type=cWaveSource

;
; Components
;

[waveIn:cWaveSource]
writer.dmLevel=wave
filename=\cm[inputfile(I){test.wav}:name of input file]
monoMixdown = 1
