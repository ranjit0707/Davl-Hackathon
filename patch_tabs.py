import os, re, uuid

utils_dir = r"c:\Users\Ranjit\Downloads\DAVL_exam\DAVL_exam\utils"

for filename in os.listdir(utils_dir):
    if not filename.endswith(".py"): continue
    filepath = os.path.join(utils_dir, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Find the tabs definition handling newlines:
    match = re.search(r"(\w+(?:,\s*\w+)*)\s*=\s*st\.tabs\(\s*\[(.*?)\]\s*\)", content, flags=re.DOTALL)
    if not match:
        continue

    tabs_vars = [v.strip() for v in match.group(1).split(",")]
    
    tabs_names_raw = match.group(2)
    names = re.findall(r'("(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\')', tabs_names_raw)

    radio_key = f"radio_{filename[:-3]}_{uuid.uuid4().hex[:4]}"
    
    new_def = f"_sel_{filename[:-3]} = st.radio('Select View', [{', '.join(names)}], horizontal=True, label_visibility='collapsed', key='{radio_key}')"
    content = content.replace(match.group(0), new_def)
    
    for var, name in zip(tabs_vars, names):
        content = re.sub(rf"with\s+{re.escape(var)}\s*:", f"if _sel_{filename[:-3]} == {name}:", content)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
        
    print(f"Patched {filename}")
