""" This module implements the transcription of the Hebrew text. """


def script(f, line):

    """ Converts names of Hebrew characters to symbols and implements transcription.
    :param f: Transcription file
    :param line: The line to be transcript
    :return: The updated file after transcription
    """

    hebrew_dict = {
        'Alef': 'א',
        'Bet': 'ב',
        'Gimel': 'ג',
        'Dalet': 'ד',
        'He': 'ה',
        'Waw': 'ו',
        'Zayin': 'ז',
        'Tet': 'ט',
        'Yod': 'י',
        'Kaf': 'כ',
        'Kaf-final': 'ך',
        'Lamed': 'ל',
        'Mem': 'ם',
        'Mem-medial': 'מ',
        'Nun-final': 'ן',
        'Nun-medial': 'נ',
        'Samekh': 'ס',
        'Ayin': 'ע',
        'Pe': 'פ',
        'Pe-final': 'ף',
        'Tsadi-final': 'ץ',
        'Tsadi-medial': 'צ',
        'Qof': 'ק',
        'Resh': 'ר',
        'Shin': 'ש',
        'Taw': 'ת',
        'Het': 'ח'
    }

    # Convert the names of Hebrew characters to symbols from the dictionary
    replace = [x if x not in hebrew_dict else hebrew_dict[x] for x in list(line)]
    # Reverse order from right-to-left.
    replace.reverse()
    for element in replace:
        f.write(element)
    f.write('\n')

    return f
