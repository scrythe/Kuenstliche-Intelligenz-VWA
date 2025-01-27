document.querySelectorAll('.celltag_hide_input').forEach((cell) => {
  cell.querySelector('.jp-Cell-inputWrapper').remove();
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

// document.querySelectorAll('h1').forEach((h1, h1_number) => {
//   h1_number = h1_number + 1;
//   h1.prepend(h1_number, ' ');

//   h1.querySelectorAll('h2').forEach((h2, h2_number) => {
//     console.log(h2);
//     h2_number = h2_number + 1;
//     h2.prepend(h1_number, '.', h2_number, ' ');
//   });
// });

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
    return;
  }

  h2 = cell.querySelector('h2');
  if (h2) {
    h2_number += 1;
    h3_number = 0;
    h2.prepend(h1_number, '.', h2_number, ' ');
    return;
  }

  h3 = cell.querySelector('h3');
  if (h3) {
    h3_number += 1;
    h3.prepend(h1_number, '.', h2_number, '.', h3_number, ' ');
    return;
  }
});
