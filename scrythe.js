function hideTaggedCells() {
  document.querySelectorAll('.celltag_hide_input').forEach((cell) => {
    input_cells = cell.querySelector('.jp-Cell-inputWrapper');
    if (input_cells) input_cells.remove();
  });

  document.querySelectorAll('.celltag_hide_output').forEach((cell) => {
    output_cells = cell.querySelector('.jp-Cell-outputWrapper');
    if (output_cells) output_cells.remove();
  });
}

function hideCodeNumber() {
  document.querySelectorAll('.jp-InputPrompt').forEach((cell) => {
    cell.remove();
  });

  document.querySelectorAll('.jp-OutputPrompt').forEach((cell) => {
    cell.remove();
  });
}

function addNumberToHeaders() {
  toc = [];
  h1_number = 0;
  h2_number = 0;
  h3_number = 0;
  document.querySelectorAll('.jp-RenderedMarkdown').forEach((cell) => {
    h1 = cell.querySelector('h1');
    if (h1) {
      if (h1.innerText == 'Abstract') {
        return;
      }
      h1_number += 1;
      h2_number = 0;
      h3_number = 0;
      h1.innerHTML = `${h1_number} ${h1.innerText}`;
    }

    h2 = cell.querySelector('h2');
    if (h2) {
      h2_number += 1;
      h3_number = 0;
      h2.innerHTML = `${h1_number}.${h2_number} ${h2.innerText}`;
    }

    h3 = cell.querySelector('h3');
    if (h3) {
      h3_number += 1;
      h3.innerHTML = `${h1_number}.${h2_number}.${h3_number} ${h3.innerText}`;
    }
  });
}

function replaceEnQuotesToGerman() {
  document.querySelectorAll('.jp-RenderedMarkdown p').forEach((cell) => {
    if (cell.querySelector('a')) return;
    const regex = /"(.*?)"/g;
    const substitue = '„$1“';
    cell.textContent = cell.textContent.replace(regex, substitue);
  });
}

function generateAbb(abb_advanced = []) {
  const abbElement = document.querySelector('#abb');
  abbElement.innerHTML = '';
  const abbTitle = document.createElement('h1');
  const h1s = document.querySelectorAll('h1');
  const lastH1 = h1s[h1s.length - 1];
  const lastH1Number = lastH1.innerText.split(' ')[0];
  abbTitle.innerHTML = parseInt(lastH1Number) + 1 + ' Abbildungsverzeichnis';
  abbTitle.id = 'Abbildungsverzeichnis';
  abbElement.append(abbTitle);

  const abbs = Array.from(document.querySelectorAll('figcaption'));
  const abbKeywords = abbs.map((abb) => abb.innerText);
  abbs.forEach((abb, index) => {
    const abbLink = document.createElement('a');
    const abbTitle = document.createElement('span');
    const abbPage = document.createElement('span');

    abbLink.href = '#' + abb.id;
    abbTitle.innerHTML = abb.innerText;
    if (abb_advanced.length) {
      const page = abb_advanced[index][1];
      abbPage.innerHTML = page;
    }
    abbTitle.className = 'title';
    abbPage.className = 'page';

    abbLink.append(abbTitle);
    abbLink.append(abbPage);

    abbElement.append(abbLink);
  });
  return abbKeywords;
}

function generateTOC(toc = []) {
  const tocElement = document.querySelector('#toc');
  tocElement.innerHTML = '';
  const tocTitle = document.createElement('h1');
  tocTitle.innerHTML = 'Inhaltsverzeichnis';
  tocElement.append(tocTitle);

  const headings = Array.from(document.querySelectorAll('h1, h2, h3'));
  headings.splice(1, 1);
  const headerKeywords = headings.map((heading) => heading.innerText);
  headings.forEach((heading, index) => {
    const tocLink = document.createElement('a');
    const tocNumber = document.createElement('span');
    const tocTitle = document.createElement('span');
    const tocPage = document.createElement('span');

    let headingNumber = heading.innerText.split(' ')[0];
    let titleText = heading.innerText.substring(headingNumber.length);
    const headingLevel = parseInt(heading.tagName[1]);
    const indent = headingLevel - 1;

    tocNumber.style.paddingRight = `2em`;
    tocLink.style.paddingLeft = `${indent}em`;

    if (!titleText) {
      // when Abstract (no number)
      titleText = headingNumber;
      headingNumber = '';
      tocNumber.style.paddingRight = 'unset';
    }

    tocLink.href = '#' + heading.id;
    tocNumber.innerHTML = headingNumber;
    tocTitle.innerHTML = titleText;
    if (toc.length) {
      const page = toc[index][1];
      tocPage.innerHTML = page;
    }
    tocTitle.className = 'title';
    tocPage.className = 'page';

    tocLink.append(tocNumber);
    tocLink.append(tocTitle);
    tocLink.append(tocPage);

    tocElement.append(tocLink);
  });
  return headerKeywords;
}

hideTaggedCells();
hideCodeNumber();
replaceEnQuotesToGerman();
addNumberToHeaders();

const abbKeywords = generateAbb();
const headerKeywords = generateTOC();
