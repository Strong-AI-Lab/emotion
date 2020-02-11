[componentInstances:cComponentManager]
instance[fr].type=cFramer
instance[pe].type=cVectorPreemphasis
instance[win].type=cWindower
instance[fft].type=cTransformFFT
instance[fftmagphase].type=cFFTmagphase
instance[mspec].type=cMelspec
instance[logmel].type=cVectorOperation

[fr:cFramer]
reader.dmLevel=wave
writer.dmLevel=frames
frameSize = 0.025
frameStep = 0.010
frameCenterSpecial = left

[pe:cVectorPreemphasis]
reader.dmLevel=frames
writer.dmLevel=framespe
processArrayFields = 1
k=0.97

[win:cWindower]
reader.dmLevel=frames
writer.dmLevel=winframes
processArrayFields = 1
winFunc = Ham

[fft:cTransformFFT]
reader.dmLevel=winframes
writer.dmLevel=fftc
processArrayFields = 1
inverse = 0

[fftmagphase:cFFTmagphase]
reader.dmLevel=fftc
writer.dmLevel=fftmag
processArrayFields = 1
magnitude = 1
phase = 0

[mspec:cMelspec]
reader.dmLevel=fftmag
writer.dmLevel=mspec
processArrayFields = 1
htkcompatible = 0
nBands = 40
lofreq = 20
hifreq = 6500
usePower = 1
specScale = mel
bwMethod = lr

[logmel:cVectorOperation]
reader.dmLevel=mspec
writer.dmLevel=logmel
writer.levelconf.growDyn = 1
writer.levelconf.isRb = 0
writer.levelconf.nT = 1000
operation = log
copyInputName = 0
nameAppend=logMFB
