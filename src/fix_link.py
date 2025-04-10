import glob
import re

html_files = glob.glob('book/_build/html/**/*.html', recursive=True)

for html_file in html_files:
    with open(html_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Colabへのリンクを.mdから.ipynbに変更
    modified_content = re.sub(
        r'(https://colab\.research\.google\.com/github/matt76k/ds-seminar/blob/gh-pages/_sources/.*?)\.md', 
        r'\1.ipynb', 
        content
    )
    
    if content != modified_content:
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        print(f"Fixed Colab links in {html_file}")