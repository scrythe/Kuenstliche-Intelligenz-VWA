document.querySelectorAll('.celltag_hide_input').forEach((cell) => {
  input_cells = cell.querySelector('.jp-Cell-inputWrapper');
  if (input_cells) input_cells.remove();
});

document.querySelectorAll('.celltag_hide_output').forEach((cell) => {
  output_cells = cell.querySelector('.jp-Cell-outputWrapper');
  if (output_cells) output_cells.remove();
});

document.querySelectorAll('.jp-InputPrompt').forEach((cell) => {
  cell.remove();
});

document.querySelectorAll('.jp-OutputPrompt').forEach((cell) => {
  cell.remove();
});

toc = [];
h1_number = 0;
h2_number = 0;
h3_number = 0;
document.querySelectorAll('.jp-RenderedMarkdown').forEach((cell) => {
  h1 = cell.querySelector('h1');
  if (h1) {
    h1_number += 1;
    h2_number = 0;
    h3_number = 0;
    h1.prepend(h1_number, ' ');
  }

  h2 = cell.querySelector('h2');
  if (h2) {
    h2_number += 1;
    h3_number = 0;
    h2.prepend(h1_number, '.', h2_number, ' ');
  }

  h3 = cell.querySelector('h3');
  if (h3) {
    h3_number += 1;
    h3.prepend(h1_number, '.', h2_number, '.', h3_number, ' ');
  }
});

document.querySelectorAll('.jp-RenderedMarkdown p').forEach((cell) => {
  if (cell.querySelector('a')) return;
  const regex = /"(.*?)"/g;
  const substitue = '„$1“';
  cell.textContent = cell.textContent.replace(regex, substitue);
});

function generateTOC() {
  const tocElement = document.querySelector('#toc');
  const tocTitle = document.createElement('h1');
  tocTitle.innerHTML = 'Inhaltsverzeichnis';
  tocElement.append(tocTitle);

  const headings = Array.from(document.querySelectorAll('h1, h2, h3'));
  headings.splice(0, 1);
  headings.forEach((heading) => {
    // const tocItem = document.createElement('li');
    const tocLink = document.createElement('a');
    const tocNumber = document.createElement('span');
    const tocTitle = document.createElement('span');
    const tocPage = document.createElement('span');

    const headingNumber = heading.innerText.split(' ')[0];
    const titleText = heading.innerText.substring(headingNumber.length);
    const headingLevel = parseInt(heading.tagName[1]);
    const indent = headingLevel - 1;

    tocLink.href = '#' + heading.id;
    tocNumber.innerHTML = headingNumber;
    tocTitle.innerHTML = titleText;
    tocPage.innerHTML = 'X';
    tocTitle.className = 'title';
    tocPage.className = 'page';

    tocNumber.style.paddingRight = `2em`;
    tocLink.style.paddingLeft = `${indent}em`;

    tocLink.append(tocNumber);
    tocLink.append(tocTitle);
    tocLink.append(tocPage);

    tocElement.append(tocLink);
  });
}

generateTOC();
