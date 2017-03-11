# import music21 as mu

import sys
import midi
import itertools
import operator

def main(argv):
    if len(argv) < 2:
        print("main.py [filename]")
    else:
        # read the midi file
        pattern = midi.read_midifile(argv[1])

        # extract key signature
        key_sig = [event.data for event in pattern[0] if isinstance(event, midi.KeySignatureEvent)]
        key_sig = key_sig[0]

        # make ticks absolute instead of relative
        pattern.make_ticks_abs()

        # extract only timing and note data
        tracks = [[(event.tick, event.data) for event in track if isinstance(event, midi.NoteOnEvent)] for track in pattern]

        # concatenate tracks
        new_list = []
        for i in tracks:
            new_list.extend(i)

        # sort events
        sort = sorted(new_list)

        # merge events by timing
        mrg = merge(sort)

        # carries over active notes to the next event
        carried_over = carry_over(mrg)

        # remove "silence" events
        no_silence = [x for x in carried_over if len(x[1]) > 0]

        # unique
        uniq = [(event[0], unique(event[1])) for event in no_silence]

        # pitch class (ignore octave)
        no_oct = [sorted([x[0] % 12 for x in event[1]]) for event in uniq]

        # unique, ignoring octave
        final = [unique(notes) for notes in no_oct]

        # get scale
        scale = get_scale(key_sig)

        # get named chords for that scale
        named_chords = get_named_chords(scale)

        # for each chord, find the nearest named chord
        final_final = [find_closest_named_chord(chord, named_chords) for chord in final]

        # print out
        for i in final_final:
            print(i)

# given a (sorted) list of (tick, note_data), combine into  a list of (tick, [note_data])'s
def merge(some_list):
    if len(some_list) == 0:
        return([])
    else:
        x = some_list[0]
        dat = [n[1] for n in some_list if n[0] == x[0]]

        entry = [(x[0], dat)]
        the_rest = merge(some_list[len(dat):])
        return(entry + the_rest)

# given that merged thing, like move stuff up and then delete, y'know?
def carry_over(some_list):
    if len(some_list) == 1:
        return([])
    else:
        first = some_list[0]
        second = some_list[1]
        combined_data = first[1] + second[1]

        new_list = combined_data
        for _ in range(0,2):
            for x in combined_data:
                if ([x[0], 0] in new_list) and ([x[0], 127] in new_list):
                    new_list.remove([x[0], 0])
                    new_list.remove([x[0], 127])
                else:
                    continue

        the_rest = carry_over([(second[0], sorted(new_list))] + some_list[2:])
        return([first] + the_rest)

def unique(some_list):
    s = []
    for i in some_list:
        if i not in s:
            s.append(i)
    return s

# from the key signature tuple, get the scale interval
def get_scale(key_sig):
    if (key_sig[1] == 0):
        return [(n - key_sig[0] * 5) % 12 for n in [0,2,4,5,7,9,11]]
    else:
        return [(n - key_sig[0] * 5) % 12 for n in [9,11,0,2,4,6,8]]

# from the scale, return a list of 96 named chords
def get_named_chords(scale):
    chords = [("note", [0]),
    ("major", [0,4,7]),
    ("minor", [0,3,7]),
    ("suspended", [0,5,7]),
    ("augmented", [0,4,8]),
    ("diminished", [0,3,6]),
    ("major_sixth", [0,4,7,9]),
    ("minor_sixth", [0,3,7,9]),
    ("dominant_seventh", [0,4,7,10]),
    ("major_seventh", [0,4,7,11]),
    ("minor_seventh", [0,3,7,10]),
    ("half_diminished_seventh", [0,3,6,10]),
    ("diminished_seventh", [0,3,6,9]),
    ("major_ninth", [0,4,7,11,14]),
    ("dominant_ninth", [0,4,7,10,14]),
    ("dominant_minor_ninth", [0,4,7,10,13]),
    ("minor_ninth", [0,3,7,10,14])]

    named_chords = []
    for note in scale:
        for chord in chords:
            new_string = note_lookup(note) + "_" + chord[0]
            new_chord = [(chord_note + note) % 12 for chord_note in chord[1]]
            named_chords.append((new_string, new_chord))
    return named_chords

# from the note's number, get the note's name (string)
def note_lookup(note):
    if note == 0:
        return "C"
    elif note == 1:
        return "C#"
    elif note == 2:
        return "D"
    elif note == 3:
        return "D#"
    elif note == 4:
        return "E"
    elif note == 5:
        return "F"
    elif note == 6:
        return "F#"
    elif note == 7:
        return "G"
    elif note == 8:
        return "G#"
    elif note == 9:
        return "A"
    elif note == 10:
        return "A#"
    elif note == 11:
        return "B"

def find_closest_named_chord(chord, named_chords):
    chord_and_dist = [(named_chord[0], chord_distance(chord, named_chord[1])) for named_chord in named_chords]
    closest = min(chord_and_dist, key=operator.itemgetter(1))
    return closest[0]

# distance between two chords
def chord_distance(chord1, chord2):
    # chord1 always larger or equal to chord2
    if len(chord1) < len (chord2):
        chord1, chord2 = chord2, chord1

    set1 = set(chord1)
    set2 = set(chord2)

    # find the intersection of the two chords
    inter = set1.intersection(set2)

    # find the difference of the two chords
    diff1 = set1.difference(set2)
    diff2 = set2.difference(set1)

    # return smallest distance
    return chord_distance_iter(list(diff1), list(diff2))

def chord_distance_iter(chord1, chord2):
    paths = []

    # if the two chords are different lengths, only allow deleting
    if (len(chord1) > len(chord2)):
        paths = [delete(k, chord1, chord2) for k in range(0, len(chord1))]

    # otherwise only allow incrementing
    else:
        # this part deals with the subtracting and adding stuff
        # should be pretty fast as long as chords are less than like 6 notes
        perms = itertools.permutations(chord1)
        for x in perms:
            m = []
            for i in range(0,len(x)):
                m += [min((x[i] - chord2[i]) % 12, (chord2[i] - x[i]) % 12)]
            paths += [sum(m)]

    best_path = min(paths)
    return best_path

def delete(j, chord1, chord2):
    new_chord = chord1[:j] +  chord1[(j+1):]
    d = chord_distance_iter(new_chord, chord2)
    return d + 3


if __name__ == "__main__":
    main(sys.argv)
