
import pathlib

def write(path, content):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding='utf-8', newline='
' if '' not in content else None)
    print(f'  wrote {path}')

print('Starting file writes...')
print('Done bootstrap')
