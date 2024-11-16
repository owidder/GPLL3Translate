const htmlPdf = require('html-pdf');
const fs = require('fs');
const path = require('path');

const folderPath = '/Users/oliverwidder/dev/github/GPLL3Translate/tables'; // Replace with your folder path
const files = fs.readdirSync(folderPath).filter(file => file.endsWith('.html'));

let combinedHtml = '<html><head><style>table { width: 100%; table-layout: fixed; } td, th { word-wrap: break-word; } .page-break { page-break-after: always; }</style></head><body>';

for (const file of files) {
  const filePath = path.join(folderPath, file);
  const htmlContent = fs.readFileSync(filePath, 'utf8');
  combinedHtml += htmlContent + '<div class="page-break"></div>';
}

combinedHtml += '</body></html>';

const outputFilePath = path.join(folderPath, 'combined_tables.pdf');
htmlPdf.create(combinedHtml, { format: 'A4', orientation: 'landscape', border: '10mm' }).toFile(outputFilePath, (err, res) => {
  if (err) return console.log(err);
  console.log(res);
});
