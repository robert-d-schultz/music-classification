# import music21 as mu

import sys
import midi

def main(argv):
    if len(argv) < 2:
        print("main.py [filename]")
    else:
        # read the midi file
        pattern = midi.read_midifile(argv[1])

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

        for i in mrg:
            print(i)
        print(pattern)



# given a (sorted) list of (tick, note_data), combine into  a list of (tick, [bote_data])'s
def merge(some_array):
    if len(some_array) == 0:
        return []
    else:
        x = some_array[0]
        dat = [n[1] for n in some_array if n[0] == x[0]]
        entry = [(x[0], dat)]
        the_rest = merge(some_array[len(dat):])
        return entry + the_rest


if __name__ == "__main__":
    main(sys.argv)
