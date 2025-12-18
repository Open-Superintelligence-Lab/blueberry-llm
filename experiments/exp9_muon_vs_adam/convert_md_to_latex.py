
import re
import sys

def convert_md_to_latex(input_path, output_path):
    with open(input_path, 'r') as f:
        lines = f.readlines()

    latex_lines = []
    
    # Preamble
    latex_lines.append(r'\documentclass{article}')
    latex_lines.append(r'\usepackage{graphicx}')
    latex_lines.append(r'\usepackage{hyperref}')
    latex_lines.append(r'\usepackage{listings}')
    latex_lines.append(r'\usepackage{xcolor}')
    latex_lines.append(r'\usepackage{booktabs}')
    latex_lines.append(r'\usepackage{geometry}')
    latex_lines.append(r'\usepackage{amsmath}')
    latex_lines.append(r'\usepackage{amssymb}')
    latex_lines.append(r'\usepackage{longtable}')
    latex_lines.append(r'\geometry{a4paper, margin=1in}')
    latex_lines.append(r'\lstset{basicstyle=\ttfamily, breaklines=true, frame=single}')
    latex_lines.append(r'')
    
    # Metadata extraction
    title = "Analysis and Design of Novel Optimizers for Neural Networks"
    author = "Vuk Rosić"
    date = "November 2025"
    
    # Try to parse metadata from the first few lines
    for line in lines[:20]:
        if line.startswith("**Author:**"):
            author = line.replace("**Author:**", "").strip()
        if line.startswith("**Date:**"):
            date = line.replace("**Date:**", "").strip()
    
    latex_lines.append(f'\\title{{{title}}}')
    latex_lines.append(f'\\author{{{author}}}')
    latex_lines.append(f'\\date{{{date}}}')
    latex_lines.append(r'\begin{document}')
    latex_lines.append(r'\maketitle')
    latex_lines.append(r'')

    in_code_block = False
    code_lang = ""
    in_list = False
    list_type = None # 'itemize' or 'enumerate'
    in_table = False
    table_lines = []
    in_abstract = False
    
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        original_line = lines[i] # Keep original for indentation check
        
        # Skip metadata lines we already processed or don't need in body
        if i < 20 and (line.startswith("**Author:**") or line.startswith("**Advisor:**") or 
                       line.startswith("**Date:**") or line.startswith("**Institution:**") or 
                       line.startswith("# Analysis and Design") or line.strip() == "---"):
            i += 1
            continue

        # End Abstract if we hit a separator or header
        if in_abstract and (line.strip() == "---" or line.startswith("#")):
            latex_lines.append(r'\end{abstract}')
            in_abstract = False
            if line.strip() == "---":
                i += 1
                continue

        # Code Blocks
        if line.strip().startswith("```"):
            if not in_code_block:
                in_code_block = True
                code_lang = line.strip().replace("```", "")
                if code_lang:
                    latex_lines.append(f'\\begin{{lstlisting}}[language={code_lang}]')
                else:
                    latex_lines.append(r'\begin{lstlisting}')
            else:
                in_code_block = False
                latex_lines.append(r'\end{lstlisting}')
            i += 1
            continue
        
        if in_code_block:
            latex_lines.append(line)
            i += 1
            continue

        # Tables
        if line.strip().startswith("|"):
            if not in_table:
                # Check if it's a table header (next line has |---)
                if i + 1 < len(lines) and lines[i+1].strip().startswith("|") and "-" in lines[i+1]:
                    in_table = True
                    table_lines = [line]
                else:
                    pass
            else:
                table_lines.append(line)
            i += 1
            continue
        else:
            if in_table:
                process_table(table_lines, latex_lines)
                in_table = False
                table_lines = []
        
        # Headers
        header_match = re.match(r'^(#+)\s+(.*)', line)
        if header_match:
            # If we were in a list, close it
            if in_list:
                latex_lines.append(f'\\end{{{list_type}}}')
                in_list = False

            level = len(header_match.group(1))
            text = header_match.group(2)
            
            if text.lower() == "abstract":
                latex_lines.append(r'\begin{abstract}')
                in_abstract = True
            elif text.lower() == "table of contents":
                latex_lines.append(r'\tableofcontents')
                latex_lines.append(r'\newpage')
                # Skip the manual TOC lines
                while i + 1 < len(lines) and (re.match(r'^\d+\.\s+\[', lines[i+1]) or lines[i+1].strip() == ""):
                    i += 1
            else:
                # Remove numbering from text if present
                text = re.sub(r'^\d+(\.\d+)*\.?\s+', '', text)
                
                if level == 1:
                    latex_lines.append(f'\\section{{{text}}}')
                elif level == 2:
                    latex_lines.append(f'\\section{{{text}}}')
                elif level == 3:
                    latex_lines.append(f'\\subsection{{{text}}}')
                elif level == 4:
                    latex_lines.append(f'\\subsubsection{{{text}}}')
                else:
                    latex_lines.append(f'\\paragraph{{{text}}}')
            i += 1
            continue

        # Lists
        is_itemize = line.strip().startswith("- ") or line.strip().startswith("* ")
        is_enumerate = re.match(r'^\d+\.\s+', line.strip())
        
        if is_itemize or is_enumerate:
            if not in_list:
                in_list = True
                list_type = 'itemize' if is_itemize else 'enumerate'
                latex_lines.append(f'\\begin{{{list_type}}}')
            
            # Switch list type
            if is_itemize and list_type == 'enumerate':
                latex_lines.append(r'\end{enumerate}')
                latex_lines.append(r'\begin{itemize}')
                list_type = 'itemize'
            elif is_enumerate and list_type == 'itemize':
                latex_lines.append(r'\end{itemize}')
                latex_lines.append(r'\begin{enumerate}')
                list_type = 'enumerate'
                
            content = line.strip()
            if is_itemize:
                # Handle both "- " and "* "
                if content.startswith("- "):
                    content = content[2:]
                elif content.startswith("* "):
                    content = content[2:]
            else:
                content = re.sub(r'^\d+\.\s+', '', content)
            
            content = format_inline(content)
            latex_lines.append(f'\\item {content}')
            i += 1
            continue
        else:
            if in_list:
                if line.strip() == "":
                    # Blank line. Check if next line is list item.
                    # Peek ahead
                    next_line = None
                    j = i + 1
                    while j < len(lines) and lines[j].strip() == "":
                        j += 1
                    if j < len(lines):
                        next_line = lines[j]
                    
                    if next_line:
                        is_next_item = next_line.strip().startswith("- ") or next_line.strip().startswith("* ") or re.match(r'^\d+\.\s+', next_line.strip())
                        is_next_header = next_line.strip().startswith("#")
                        next_indent = len(next_line) - len(next_line.lstrip())
                        
                        if is_next_item:
                            # List continues
                            pass
                        elif is_next_header:
                            latex_lines.append(f'\\end{{{list_type}}}')
                            in_list = False
                        elif next_indent >= 2: # Indented text
                            pass
                        else:
                            # Unindented text -> End of list
                            latex_lines.append(f'\\end{{{list_type}}}')
                            in_list = False
                    else:
                        # End of file
                        latex_lines.append(f'\\end{{{list_type}}}')
                        in_list = False
                    
                    latex_lines.append("") # Keep blank line
                    i += 1
                    continue
                else:
                    # Non-blank line.
                    # Check indentation
                    indent = len(original_line) - len(original_line.lstrip())
                    if indent >= 2:
                        # Continuation of item
                        content = format_inline(line.strip())
                        latex_lines.append(content)
                        i += 1
                        continue
                    else:
                        # End of list
                        latex_lines.append(f'\\end{{{list_type}}}')
                        in_list = False
                        # Fall through to regular text processing

        # Regular text
        if line.strip() == "":
            latex_lines.append("")
        else:
            formatted_line = format_inline(line)
            latex_lines.append(formatted_line)

        i += 1

    # Close any open environments
    if in_list:
        latex_lines.append(f'\\end{{{list_type}}}')
    if in_table:
        process_table(table_lines, latex_lines)
    if in_abstract:
        latex_lines.append(r'\end{abstract}')

    latex_lines.append(r'\end{document}')
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(latex_lines))

def process_table(table_lines, latex_lines):
    if len(table_lines) < 2:
        return
    
    header = table_lines[0]
    rows = table_lines[2:]
    
    cols = header.count('|') - 1
    if cols < 1: return
    
    latex_lines.append(r'\begin{center}')
    latex_lines.append(r'\begin{longtable}{|' + 'l|' * cols + '}')
    latex_lines.append(r'\hline')
    
    header_cells = [c.strip() for c in header.strip('|').split('|')]
    latex_lines.append(' & '.join([format_inline(c) for c in header_cells]) + r' \\ \hline')
    
    for row in rows:
        cells = [c.strip() for c in row.strip('|').split('|')]
        while len(cells) < cols:
            cells.append("")
        latex_lines.append(' & '.join([format_inline(c) for c in cells]) + r' \\ \hline')
        
    latex_lines.append(r'\end{longtable}')
    latex_lines.append(r'\end{center}')

def format_inline(text):
    # Escape backslash first
    text = text.replace('\\', r'\textbackslash')
    
    # Escape other chars
    text = text.replace('&', r'\&')
    text = text.replace('%', r'\%')
    text = text.replace('$', r'\$')
    text = text.replace('#', r'\#')
    text = text.replace('_', r'\_')
    text = text.replace('{', r'\{')
    text = text.replace('}', r'\}')
    text = text.replace('^', r'\^{}')
    text = text.replace('~', r'\~{}')

    # Inline code `...`
    text = re.sub(r'`(.*?)`', r'\\texttt{\1}', text)

    # Bold
    text = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', text)
    # Italic
    text = re.sub(r'\*(.*?)\*', r'\\textit{\1}', text)
    
    # Links
    text = re.sub(r'\[(.*?)\]\((.*?)\)', r'\\href{\2}{\1}', text)
    
    # Checkboxes
    text = text.replace('[x]', r'$\boxtimes$')
    text = text.replace('[ ]', r'$\square$')
    
    return text

if __name__ == "__main__":
    convert_md_to_latex(sys.argv[1], sys.argv[2])
