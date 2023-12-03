import os
from bs4 import BeautifulSoup


def generate_toc(html_content):
    # Parse the HTML
    soup = BeautifulSoup(html_content, 'html.parser')
    headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5'])

    # Assign IDs and Create ToC Structure
    toc_structure = []
    for i, heading in enumerate(headings):
        heading_id = f"heading_{i}"
        heading['id'] = heading_id
        # Remove pilcrow symbols from heading text
        heading_text = heading.get_text().replace("¶", "").strip()
        toc_structure.append({
            'text': heading_text,
            'id': heading_id,
            'level': int(heading.name[1]),
        })

    # Generate ToC HTML with a toggle button
    toc_html = '''
    <nav id="toc">
        <button id="toc-toggle">
            <svg width="24" height="24" xmlns="http://www.w3.org/2000/svg" fill-rule="evenodd" clip-rule="evenodd">
                <path d="M4 3h16v2H4V3zm0 8h16v2H4v-2zm0 5h16v2H4v-2zM4 8h16v2H4V8z"/>
            </svg>
        </button>
        <ul id="toc-list">
    '''
    for item in toc_structure:
        #font weight bold for h1, normal for h2-5
        font_weight = "bold" if item['level'] == 1 else "normal"
        font_size = 20 - (item['level'] - 2)
        margin_left = item['level'] * 20
        toc_html += f"<li style='font-weight: {font_weight}; font-size: {font_size}px; margin-left: {margin_left}px;'><a href='#{item['id']}'>{item['text']}</a></li>"

    #attributions to the creator of the ToC
    toc_html += "<li style='margin-left: 2px;'><a href='https://mathiasschjoedt-bavngaard.github.io/'>Creator Of ToC</a></li></ul></nav>"

    toc_html += "<li style='font-size: 1px; margin-left: 2px;'><a href='https://mathiasschjoedt-bavngaard.github.io/' target='_blank' >Mathias Schjoedt-Bavngaard</a></li></ul></nav>"

    # Insert ToC into Original HTML and Style with CSS
    soup.body.insert(0, BeautifulSoup(toc_html, 'html.parser'))
    css = '''
    nav#toc {
        position: fixed;
        left: 0;
        top: 0;
        width: 30vw;
        height: 100%;
        overflow-x: hidden;
        background-color: #f8f8f8;
        padding: 10px;
        transition: width 0.3s ease-in-out;

    }
    nav#toc a {
        text-decoration: none;
        color: #333;
    }
    nav#toc a:hover {
        text-decoration: underline;
    }
    #toc-toggle {
        background-color: #f8f8f8;
        border: none;
        padding: 10px;
        cursor: pointer;
        position: absolute;
        top: 10px;
        right: 10px;
    }
    #toc-list {
        max-height: 90%;
        overflow-y: auto;
        .collapsed {
            display: none;
        }
    }
    nav#toc.collapsed {
        width: 40px;
    }
    body.expanded {
        margin-left: 32vw;
    }
    body.transition {
        transition: margin-left 0.3s ease-in-out;
    }
    '''
    soup.head.append(BeautifulSoup(f"<style>{css}</style>", 'html.parser'))

    # Add JavaScript for Smooth Scroll and ToC Toggle Button
    js = '''
    document.querySelectorAll('#toc a').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });
    var bodyfirst = document.body;
    bodyfirst.classList.add('expanded');
    bodyfirst.classList.add('transition');
    document.getElementById('toc-toggle').addEventListener('click', function() {
        var toc = document.getElementById('toc');
        var list = document.getElementById('toc-list');
        var body = document.body;
        if (toc.classList.contains('collapsed')) {
            toc.classList.remove('collapsed');
            list.classList.remove('collapsed');
            body.classList.add('expanded');
        } else {
            toc.classList.add('collapsed');
            toc.classList.add('collapsed');
            body.classList.remove('expanded');
        }
    });
    '''
    soup.body.append(BeautifulSoup(f"<script>{js}</script>", 'html.parser'))

    return str(soup)


# List of HTML files to transform
html_files = ['O1.html', 'O2.html', 'O3.html', 'index.html']

# Path to the directory containing the HTML files
html_dir = "mergednotebooks/"

# Apply the transformation to each HTML file
for html_file in html_files:
    with open(os.path.join(html_dir, html_file), "r", encoding="utf-8") as file:
        original_html = file.read()
    transformed_html = generate_toc(original_html)
    with open(os.path.join(html_dir, html_file), "w", encoding="utf-8") as file:
        file.write(transformed_html)
