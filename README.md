# Pre-requisuiets

pip install -r requirements.txt

for notebook:
pip install jupyterlab

# Run

jupyter lab

# Convert to HTML

jupyter nbconvert --to html --template .\scrythe.tpl .\Künstliche_Intelligenz.ipynb

# Convert to Markdown with no pictures

jupyter nbconvert --to markdown --template .\no_images.tpl .\Künstliche_Intelligenz.ipynb

# My Binder

https://mybinder.org/v2/gh/scrythe/Kuenstliche-Intelligenz-VWA/main?urlpath=%2Fdoc%2Ftree%2FK%C3%BCnstliche_Intelligenz.ipynb

# Run Notebook locally with jupyter lab

pip install jupyterlab
jupyter lab

# Convert notebook to pdf

pip install playwright pymupdf pikepdf
python create_html_and_pdf
