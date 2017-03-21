import midi
import itertools
import operator
import sys

def preprocess(pat):

    # make ticks absolute instead of relative
    pat.make_ticks_abs()

    # extract only timing and note data
    event_list = []
    for track in pat:
        for event in track:
            if isinstance(event, midi.NoteOnEvent):
                if event.data[1] > 0:
                    event_list.append((event.tick, [event.data[0], 127]))
                else:
                    event_list.append((event.tick, event.data))
            elif isinstance(event, midi.NoteOffEvent):
                event_list.append((event.tick, [event.data[0], 0]))

    # sort events
    sort = sorted(event_list)

    # merge events by timing
    mrg = merge(sort)

    # carry over active notes to the next event
    carried_over = carry_over(mrg)

    # remove "silence" events
    no_silence = [x for x in carried_over if len(x[1]) > 0]

    # pitch class (ignore octave)
    no_oct = [(event[0], sorted([x[0] % 12 for x in event[1]])) for event in no_silence]

    # keep unique only - assumes chords are sets of notes
    uniq = [(event[0], unique(event[1])) for event in no_oct]

    # extract key signature
    key_sig = []
    for channel in pat:
        key_sig.extend([event.data for event in channel if isinstance(event, midi.KeySignatureEvent)])

    # get scale
    scale = []
    if len(key_sig) == 0:
        notes = [event[1][0] for event in event_list if event[1][1] == 127]
        scale = find_scale(notes)
    elif len(key_sig) > 0:
        if len(key_sig) == 1:
            key_sig = key_sig[0]
        else:
            if key_sig.count(key_sig[0]) == len(key_sig):
                key_sig = key_sig[0]
            else:
                # fuck it, just use the first one
                key_sig = key_sig[0]
                # raise ValueError('One of the files has two or more different key signatures and must be analyzed in multiple parts. Please let this not fire.')

        # if the key signature's first byte is greater than 7 then it should be negative negative
        if key_sig[0] > 7:
            key_sig = [key_sig[0] - 256, key_sig[1]]

        scale = get_scale(key_sig)

    # get named chords for that scale
    named_chords = get_named_chords(scale)

    # for each chord, find the nearest named chord
    # list of (time, chord name), chord name gives the "place" in the scale and the chord
    final = [(chord[0], find_closest_named_chord(chord[1], named_chords)) for chord in uniq]

    return final

# find the scale based on distribution of notes
def find_scale(notes):
    # note distribution
    dist = []
    for i in range(11):
        dist.append(notes.count(i))

    # make a list of all scales
    scales = []
    for k in [0,1]:
        for j in range(-7,7):
            if k == 0:
                scales.append([(n - j * 5) % 12 for n in [0,2,4,5,7,9,11]])
            else:
                scales.append([(n - j * 5) % 12 for n in [9,11,0,2,4,6,8]])

    # find out-of-scale rate for each scale
    out_of_scale = [0] * len(scales)
    for l in range(len(scales)):
        for m in range(11):
            if m not in scales[l]:
                out_of_scale[l] += dist[m]

    # pick minimum
    return scales[out_of_scale.index(min(out_of_scale))]

# given a (sorted) list of (tick, note_data), combine into  a list of (tick, [note_data])'s
def merge(event_list):
    out = [(event_list[0][0], [event_list[0][1]])]
    for i in range(1,len(event_list)):
        # if the current event happens at the same time as the last event
        if (event_list[i][0] == event_list[i-1][0]):
            # append the current event to the last timestamp in the out array
            out[-1][1].append(event_list[i][1])
        # otherwise start a new timestamp in the out array
        else:
            out.append((event_list[i][0], [event_list[i][1]]))
    return out

# given that merged list, move all notes up a timestamp until they are "annihilated"
def carry_over(some_list):
    out = [some_list[0]]
    for i in range(1,len(some_list)):
        # annihalate the current timestamp's notes with the last out entry's notes
        combined_data = some_list[i][1] + out[-1][1]
        new_data = combined_data
        for _ in range(0,2):
            for x in combined_data:
                if ([x[0], 0] in new_data) and ([x[0], 127] in new_data):
                    new_data.remove([x[0], 0])
                    new_data.remove([x[0], 127])
                else:
                    continue

        # append the (sorted) resultant data to the out array
        out.append((some_list[i][0], sorted(new_data)))
    return out

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
            # for specific chords
            # new_string = note_lookup(note) + "_" + chord[0]
            new_string = str(scale.index(note)) + "_" + chord[0]
            new_chord = [(chord_note + note) % 12 for chord_note in chord[1]]
            named_chords.append((new_string, new_chord))
    return named_chords

# from the note's number, get the note's name (string)
# not used
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
    # deleting
    delete_paths = []
    delete_cost = []
    if (len(chord1) > len(chord2)):
        delete_paths = itertools.combinations(chord1, len(chord2))
        delete_cost = (len(chord1) - len(chord2)) * 3
    else:
        delete_paths = [chord1]
        delete_cost = 0

    # incrementing
    path_cost = []
    for path in delete_paths:
        # this part deals with the subtracting and adding stuff
        # should be pretty fast as long as chords are less than like 6 notes
        perms = itertools.permutations(path)
        for x in perms:
            m = []
            for i in range(len(x)):
                m += [min((x[i] - chord2[i]) % 12, (chord2[i] - x[i]) % 12)]
            path_cost += [sum(m) + delete_cost]

    best_path = min(path_cost)
    return best_path
