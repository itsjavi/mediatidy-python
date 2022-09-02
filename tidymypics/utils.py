import warnings


def progress_bar(current, total, bar_length=25):
    if total == 0:
        return
    
    fraction = current / total

    arrow = int(fraction * bar_length - 1) * '-' + '>'
    padding = int(bar_length - len(arrow)) * ' '

    ending = '\n' if current == total else '\r'

    print(
        f' Progress: [{arrow}{padding}] {int(fraction*100)}% ({current} from {total})            ', end=ending)
