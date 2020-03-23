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
k=0.97

[win:cWindower]
reader.dmLevel=framespe
writer.dmLevel=winframes
winFunc = Ham

[fft:cTransformFFT]
reader.dmLevel=winframes
writer.dmLevel=fftc
inverse = 0

[fftmagphase:cFFTmagphase]
reader.dmLevel=fftc
writer.dmLevel=fftmag
magnitude = 1
phase = 0

[mspec:cMelspec]
reader.dmLevel=fftmag
writer.dmLevel=mspec
htkcompatible = 0
nBands = 40
lofreq = 0
hifreq = 8000
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
