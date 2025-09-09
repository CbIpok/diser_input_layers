import json
from pathlib import Path

def main():
    p = Path('prompt.txt')
    data = p.read_bytes()
    try:
        text = data.decode('utf-8')
    except UnicodeDecodeError:
        # Fallbacks if file was saved in a local 8-bit codepage
        for enc in ('cp1251', 'koi8-r', 'latin1'):
            try:
                text = data.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            text = data.decode('utf-8', errors='replace')
    lines = [ln.rstrip('\r') for ln in text.split('\n') if ln.strip()]
    last = lines[-1] if lines else ''
    print(json.dumps({
        'lines': lines,
        'last_line': last,
    }, ensure_ascii=True))

if __name__ == '__main__':
    main()

