semitoneNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

A4_tune = 440
semitones = [A4_tune * 2**((i-57)/12) for i in range(108)]
# for i in range(9):
#     print(str(i) + ' ' + str(semitones[i*12 + 4]))

for i in range(len(semitones)):
    print(semitoneNames[i%12] + str(i//12) + ': ' + str(semitones[i]))
